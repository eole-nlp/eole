"""GPU stages around the synchronous hybrid expert boundary.

Only the ordinary single-token Qwen-style residual layout is supported. Other
layouts and requests keep using the original eager decoder. No model math is
reimplemented here: routing, attention and shared experts call the same methods.
"""

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from eole.utils.logging import logger


class GPUStage:
    """Capture on the third real call; mutable state advances exactly once.

    State buffers are retained while captured and their addresses invalidate the
    graph if caches are reinitialized. Explicit arguments are copied into owned
    buffers. Outputs alias graph storage, overwritten on the next replay.
    """

    def __init__(self, function, states=lambda: (), stream=None):
        self.function, self.states = function, states
        self.stream = stream
        self.failed = set()
        self.captures = 0
        self.reset()

    def reset(self):
        self.graph = self.inputs = self.outputs = self.state_refs = None
        self.signature, self.warm_calls = None, 0

    def __call__(self, *args, **kwargs):
        leaves, spec = tree_flatten((args, kwargs))
        states = tuple(t for t in self.states() if isinstance(t, torch.Tensor))
        metadata = tuple(
            (tuple(t.shape), tuple(t.stride()), t.dtype, t.device) if isinstance(t, torch.Tensor) else repr(t)
            for t in leaves
        )
        signature = (tuple(t.data_ptr() for t in states), metadata, spec)
        if signature != self.signature:
            self.reset()
            self.signature = signature
        if metadata in self.failed or self.warm_calls < 2:
            self.warm_calls += 1
            return self.function(*args, **kwargs)
        if self.graph is None:
            graph = None
            try:
                self.inputs = [t.clone() if isinstance(t, torch.Tensor) else t for t in leaves]
                captured_args, captured_kwargs = tree_unflatten(self.inputs, spec)
                graph = torch.cuda.CUDAGraph()
                if self.stream is None:
                    self.stream = torch.cuda.Stream(device=args[0].device)
                stream = self.stream
                with torch.cuda.stream(torch.cuda.current_stream(args[0].device)):
                    with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
                        self.outputs = self.function(*captured_args, **captured_kwargs)
                self.graph, self.state_refs = graph, states
                self.captures += 1
            except RuntimeError as error:
                if graph is not None:
                    graph.reset()
                self.reset()
                self.failed.add(metadata)
                logger.warning("Hybrid GPU stage capture failed; using eager execution: %s", error)
                return self.function(*args, **kwargs)
        else:
            for target, source in zip(self.inputs, leaves):
                if isinstance(target, torch.Tensor):
                    target.copy_(source)
        self.graph.replay()
        return self.outputs


class HybridLayerGraph:
    def __init__(self, layer, stream=None):
        self.layer = layer
        self.enabled = True
        self.before = GPUStage(self.prefix, self.states, stream)
        self.after = GPUStage(self.suffix, stream=stream)

    def states(self):
        if self.layer.layer_type == "linear_attention":
            attention = self.layer.linear_attn
            return attention.conv_state, attention.recurrent_state
        attention = self.layer.self_attn
        return attention.kcache, attention.vcache, attention.cache_leftpad

    def allowed(self, x, kwargs):
        layer = self.layer
        return (
            self.enabled
            and not layer.training
            and not torch.is_grad_enabled()
            and x.shape[:2] == (1, 1)
            and kwargs.get("cache_slice") is None
            and not any(kwargs.get(k, False) for k in ("return_attn", "with_align", "collect_cross_attns"))
            and all(isinstance(t, torch.Tensor) for t in self.states())
        )

    def prefix(self, x, **kwargs):
        layer = self.layer
        _, attention, _ = layer._self_attention(x, **kwargs)
        attention.add_(x)
        ff = layer.post_attention_layernorm(attention)
        weights, ids = layer.mlp.route(ff.view(-1, ff.shape[-1]))
        return attention, ff, weights, ids

    def suffix(self, residual, ff, routed):
        return residual + self.layer.mlp.combine(ff, routed.view_as(ff))

    def __call__(self, x, **kwargs):
        residual, ff, weights, ids = self.before(x, **kwargs)
        routed = self.layer.mlp._executor.execute(ff.view(-1, ff.shape[-1]), weights, ids, training=False)
        return self.after(residual, ff, routed), None

    def close(self):
        self.before.reset()
        self.after.reset()


class HybridCacheStorage:
    """Own static attention state across requests, independently of active caches.

    Decoder teardown may detach its active caches. Compatible requests restore
    these buffers and reset their contents in place; captured addresses survive.
    A shape/dtype/device change releases graphs before replacing any buffers.
    """

    def __init__(self, graphs):
        self.graphs = graphs
        self.signature = None
        self.tensors = {}

    def begin(self, signature):
        if signature != self.signature:
            self.close()
            self.signature = signature

    def zeros(self, key, shape, *, dtype, device):
        if key not in self.tensors:
            self.tensors[key] = torch.empty(shape, dtype=dtype, device=device)
        tensor = self.tensors[key]
        # Shapes are fixed for an epoch. A caller must begin a new epoch before
        # changing them, rather than silently replacing captured storage.
        if tensor.shape != shape or tensor.dtype != dtype or tensor.device != device:
            raise ValueError("Hybrid cache shape changed without a new cache epoch")
        return tensor.zero_()

    def close(self):
        for graph in self.graphs:
            graph.close()
        self.tensors.clear()
        self.signature = None
