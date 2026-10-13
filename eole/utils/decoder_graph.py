"""Public CUDA-graph replay for a compiled, inference-only decoder call."""

import torch

from eole.utils.logging import logger


def _tensor_signature(value):
    return value.shape, value.stride(), value.dtype, value.device


class DecoderGraph:
    """Bind inputs for one cache epoch; weights/state must keep their storage.

    The first two calls run normally, warming compilation and lazy kernels with
    real tokens. Capture records the third call without executing it; one replay
    then processes that token. Consequently recurrent/KV state advances exactly
    once per call, without extra warmup tokens or cache snapshots.

    Outputs (including decoder hidden states) alias replay storage and are
    overwritten by the next replay. Clone any result retained across replay.
    reset() drops bindings but preserves failed signatures for this runner's
    lifetime, avoiding repeated captures and warnings on later requests.
    """

    def __init__(self, function, fallback):
        self.function = function
        self.fallback = fallback
        self.failed_signatures = set()
        self.capture_stream = None
        self.reset()

    def reset(self):
        self.graph = None
        self.inputs = None
        self.outputs = None
        self.signature = None
        self.warm_calls = 0
        self.failed = False

    def __call__(self, emb, *, tgt_pad_mask):
        signature = (_tensor_signature(emb), _tensor_signature(tgt_pad_mask))
        if signature != self.signature:
            self.reset()
            self.signature = signature
        if signature in self.failed_signatures:
            self.failed = True
            return self.fallback(emb, tgt_pad_mask=tgt_pad_mask)
        if self.warm_calls < 2:
            result = self.function(emb, tgt_pad_mask=tgt_pad_mask)
            self.warm_calls += 1
            return result
        if self.graph is None:
            graph = None
            try:
                with torch.cuda.device(emb.device):
                    self.inputs = (emb.clone(), tgt_pad_mask.clone())
                    if self.capture_stream is None or self.capture_stream.device != emb.device:
                        self.capture_stream = torch.cuda.Stream(device=emb.device)
                    graph = torch.cuda.CUDAGraph()
                    # The outer context restores the caller's stream even if
                    # graph.__enter__/__exit__ itself raises during capture.
                    with torch.cuda.stream(torch.cuda.current_stream(emb.device)):
                        with torch.cuda.graph(graph, stream=self.capture_stream, capture_error_mode="thread_local"):
                            self.outputs = self.function(self.inputs[0], tgt_pad_mask=self.inputs[1])
            except RuntimeError as error:
                # Recording does not execute GPU state updates. Release the
                # partial graph/outputs before fallback, including capture OOM.
                self.reset()
                if graph is not None:
                    graph.reset()
                error.__traceback__ = None
                self.signature = signature
                self.failed = True
                self.failed_signatures.add(signature)
                logger.warning("Decoder CUDA graph capture failed; using uncaptured execution: %s", error)
                return self.fallback(emb, tgt_pad_mask=tgt_pad_mask)
            self.graph = graph
        else:
            self.inputs[0].copy_(emb)
            self.inputs[1].copy_(tgt_pad_mask)
        self.graph.replay()
        return self.outputs
