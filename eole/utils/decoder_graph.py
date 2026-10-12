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
    """

    def __init__(self, function, fallback):
        self.function = function
        self.fallback = fallback
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
        if self.failed:
            return self.fallback(emb, tgt_pad_mask=tgt_pad_mask)
        if self.warm_calls < 2:
            result = self.function(emb, tgt_pad_mask=tgt_pad_mask)
            self.warm_calls += 1
            return result
        if self.graph is None:
            self.inputs = (emb.clone(), tgt_pad_mask.clone())
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.device(emb.device), torch.cuda.graph(graph):
                    self.outputs = self.function(self.inputs[0], tgt_pad_mask=self.inputs[1])
            except torch.cuda.OutOfMemoryError:
                self.reset()
                raise
            except RuntimeError as error:
                # Failed capture does not execute recorded GPU operations.
                self.reset()
                self.signature = signature
                self.failed = True
                logger.warning("Decoder CUDA graph capture failed; using the compiled decoder: %s", error)
                return self.fallback(emb, tgt_pad_mask=tgt_pad_mask)
            self.graph = graph
        else:
            self.inputs[0].copy_(emb)
            self.inputs[1].copy_(tgt_pad_mask)
        self.graph.replay()
        return self.outputs
