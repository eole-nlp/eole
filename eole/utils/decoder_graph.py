"""Public CUDA-graph replay for a compiled, inference-only decoder call."""

from datetime import timedelta

import torch

from eole.utils.logging import logger


def initialize_cuda_graph_thread():
    """Initialize Inductor graph-tree state for a Python inference worker.

    PyTorch initializes this state only in the thread importing cudagraph_trees
    (or in autograd-created threads via inherited TLS). A new ThreadPoolExecutor
    worker has neither after a previous model imported it, so legacy compiled
    image/attention paths otherwise assert in Inductor's get_obj(). Keep each
    worker's managers independent, and preserve any inherited/existing state.
    """
    from collections import defaultdict
    import threading
    from torch._inductor import cudagraph_trees

    for name, factory in (
        ("tree_manager_containers", dict),
        ("tree_manager_locks", lambda: defaultdict(threading.Lock)),
    ):
        if not hasattr(cudagraph_trees.local, name):
            state = torch._C._get_obj_in_tls(name) if torch._C._is_key_in_tls(name) else factory()
            setattr(cudagraph_trees.local, name, state)
        if not torch._C._is_key_in_tls(name):
            torch._C._stash_obj_in_tls(name, getattr(cudagraph_trees.local, name))


def _tensor_signature(value):
    return value.shape, value.stride(), value.dtype, value.device


class CaptureCoordinator:
    """Agree on capture readiness without CUDA work or decoder collectives.

    This CPU group is shared by the decoder and verifier runners. It is used
    only during capture, never on the token replay path. All TP ranks must
    request the same program and cache epoch.
    """

    def __init__(self, timeout=60):
        import torch.distributed as dist

        self.failed = False
        self.group = dist.new_group(backend="gloo", timeout=timedelta(seconds=timeout))

    def all_success(self, success):
        import torch.distributed as dist

        with torch.inference_mode(False):
            status = torch.tensor(int(success), dtype=torch.int32, device="cpu")
            dist.all_reduce(status, op=dist.ReduceOp.MIN, group=self.group)
            return bool(status.item())


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

    Tensor-parallel allocation failures fall back on every rank. A failure
    after recording starts aborts all ranks: partial NCCL operation sequences
    cannot safely be reused, so inference workers must be reinitialized.
    """

    def __init__(self, function, fallback, *, coordinator=None):
        self.coordinator = coordinator
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
        if self.coordinator is not None and self.coordinator.failed:
            raise RuntimeError("Tensor-parallel capture failed; reinitialize the inference workers before retrying")
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
            error = None
            try:
                with torch.cuda.device(emb.device):
                    self.inputs = (emb.clone(), tgt_pad_mask.clone())
                    if self.capture_stream is None or self.capture_stream.device != emb.device:
                        self.capture_stream = torch.cuda.Stream(device=emb.device)
                    graph = torch.cuda.CUDAGraph()
            except RuntimeError as exception:
                error = exception
            # A peer may have failed allocation before entering capture. None
            # of the ranks should record NCCL collectives in that case.
            if not self._all_success(error is None):
                return self._capture_failed(signature, error, graph, emb, tgt_pad_mask)
            try:
                with torch.cuda.device(emb.device):
                    # The outer context restores the caller's stream even if
                    # graph.__enter__/__exit__ itself raises during capture.
                    with torch.cuda.stream(torch.cuda.current_stream(emb.device)):
                        with torch.cuda.graph(graph, stream=self.capture_stream, capture_error_mode="thread_local"):
                            self.outputs = self.function(self.inputs[0], tgt_pad_mask=self.inputs[1])
            except RuntimeError as exception:
                error = exception
            # Recording has not executed the token. Agree before any rank
            # replays a graph or takes the uncaptured fallback.
            if not self._all_success(error is None):
                return self._capture_failed(signature, error, graph, emb, tgt_pad_mask, recorded=True)
            self.graph = graph
        else:
            self.inputs[0].copy_(emb)
            self.inputs[1].copy_(tgt_pad_mask)
        self.graph.replay()
        return self.outputs

    def _all_success(self, success):
        return success if self.coordinator is None else self.coordinator.all_success(success)

    def _capture_failed(self, signature, error, graph, emb, tgt_pad_mask, *, recorded=False):
        # Release partial captures before fallback, including capture OOM.
        self.reset()
        if graph is not None:
            graph.reset()
        if error is not None:
            error.__traceback__ = None
        self.signature = signature
        self.failed = True
        self.failed_signatures.add(signature)
        if recorded and self.coordinator is not None:
            # A partial record may have advanced NCCL's host-side operation
            # sequence on only some ranks. It cannot be rolled back by dropping
            # the graph. Do not issue further decoder/cleanup collectives.
            self.coordinator.failed = True
            raise RuntimeError(
                "Tensor-parallel decoder CUDA graph capture failed; request aborted to preserve collective ordering"
            ) from error
        logger.warning(
            "Decoder CUDA graph capture failed; using uncaptured execution: %s",
            error if error is not None else "another tensor-parallel rank failed capture",
        )
        return self.fallback(emb, tgt_pad_mask=tgt_pad_mask)
