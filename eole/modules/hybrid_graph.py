"""GPU stages around the synchronous hybrid expert boundary.

Only the ordinary single-token Qwen-style residual layout is supported. Other
layouts and requests keep using the original eager decoder. No model math is
reimplemented here: routing, attention and shared experts call the same methods.
"""

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from eole.utils.logging import logger


class GraphPool:
    """One ordered, non-concurrent pool for a model's prefix/suffix stages.

    Capture starts at stage zero after every stage has warmed up. A changed
    signature, interrupted request or capture failure abandons the whole epoch;
    eager execution continues until the next request boundary. Expert graphs
    are independent and never allocate from this pool.
    """

    def __init__(self, device, stream=None):
        self.device, self.stream = device, stream
        self.stages = []
        self.handle = None
        self.next = 0
        self.armed = self.disabled = self.started = False

    def register(self, stage):
        stage.pool_index = len(self.stages)
        self.stages.append(stage)

    def enter(self, stage):
        self.started = True
        if self.disabled:
            return False
        if stage.pool_index != self.next:
            self.abort()
            return False
        if self.next == 0 and not self.armed and all(s.warm_calls >= 2 for s in self.stages):
            self.handle = torch.cuda.graph_pool_handle()
            self.armed = True
        self.next = (self.next + 1) % len(self.stages)
        return self.armed

    def reset(self):
        try:
            if self.started:
                torch.cuda.synchronize(self.device)
        finally:
            for stage in self.stages:
                stage.reset()
            self.handle = None
            self.next = 0
            self.armed = self.disabled = self.started = False

    def abort(self):
        try:
            self.reset()
        finally:
            self.disabled = True

    def begin(self):
        if self.disabled or self.next:
            self.reset()


class GPUStage:
    """Capture on the third real call; mutable state advances exactly once.

    State buffers are retained while captured and their addresses invalidate the
    graph if caches are reinitialized. Explicit arguments are copied into owned
    buffers. Outputs alias graph storage, overwritten on the next replay.
    """

    def __init__(self, function, states=lambda: (), stream=None, pool=None):
        self.function, self.states = function, states
        self.stream, self.pool = stream, pool
        if pool is not None:
            pool.register(self)
        self.failed = set()
        self.captures = 0
        self.reset()

    def reset(self):
        self.graph = self.inputs = self.outputs = self.state_refs = None
        self.signature, self.warm_calls = None, 0

    def __call__(self, *args, **kwargs):
        leaves, spec = tree_flatten((args, kwargs))
        states = tuple(t for t in self.states() if isinstance(t, torch.Tensor))
        # Size-one dimensions have no observable stride. A final one-token
        # prefill slice can differ from decode here without changing the layout
        # of the owned capture buffers (arguments are copied before replay).
        metadata = tuple(
            (
                (tuple(t.shape), tuple(s if n > 1 else 0 for n, s in zip(t.shape, t.stride())), t.dtype, t.device)
                if isinstance(t, torch.Tensor)
                else repr(t)
            )
            for t in leaves
        )
        signature = (tuple(t.data_ptr() for t in states), metadata, spec)
        if signature != self.signature:
            if self.pool is not None and self.signature is not None:
                self.pool.abort()
            self.reset()
            self.signature = signature
        capture_allowed = self.pool.enter(self) if self.pool is not None else True
        if metadata in self.failed or self.warm_calls < 2 or not capture_allowed:
            self.warm_calls += 1
            return self.function(*args, **kwargs)
        if self.graph is None:
            graph = None
            try:
                self.inputs = [t.clone() if isinstance(t, torch.Tensor) else t for t in leaves]
                captured_args, captured_kwargs = tree_unflatten(self.inputs, spec)
                graph = torch.cuda.CUDAGraph()
                if self.pool is not None:
                    if self.pool.stream is None:
                        self.pool.stream = self.stream or torch.cuda.Stream(device=args[0].device)
                    self.stream = self.pool.stream
                elif self.stream is None:
                    self.stream = torch.cuda.Stream(device=args[0].device)
                stream = self.stream
                with torch.cuda.stream(torch.cuda.current_stream(args[0].device)):
                    with torch.cuda.graph(
                        graph,
                        stream=stream,
                        pool=self.pool.handle if self.pool is not None else None,
                        capture_error_mode="thread_local",
                    ):
                        self.outputs = self.function(*captured_args, **captured_kwargs)
                self.graph, self.state_refs = graph, states
                self.captures += 1
            except RuntimeError as error:
                if graph is not None:
                    graph.reset()
                if self.pool is not None:
                    self.pool.abort()
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
    def __init__(self, layer, stream=None, pool=None):
        self.layer, self.pool = layer, pool
        self.enabled = True
        self.before = GPUStage(self.prefix, self.states, stream, pool)
        self.after = GPUStage(self.suffix, stream=stream, pool=pool)

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
        try:
            residual, ff, weights, ids = self.before(x, **kwargs)
            routed = self.layer.mlp._executor.execute(ff.view(-1, ff.shape[-1]), weights, ids, training=False)
            return self.after(residual, ff, routed), None
        except BaseException:
            if self.pool is not None:
                self.pool.abort()
            raise

    def close(self):
        if self.pool is not None and (self.before.signature is not None or self.after.signature is not None):
            self.pool.abort()
        else:
            self.before.reset()
            self.after.reset()


class HybridCacheStorage:
    """Own static attention state across requests, independently of active caches.

    Decoder teardown may detach its active caches. Compatible requests restore
    these buffers and reset their contents in place; captured addresses survive.
    A shape/dtype/device change releases graphs before replacing any buffers.
    """

    def __init__(self, graphs, pool=None):
        self.graphs, self.pool = graphs, pool
        self.signature = None
        self.tensors = {}

    def begin(self, signature):
        if signature != self.signature:
            self.close()
            self.signature = signature
        elif self.pool is not None:
            self.pool.begin()

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
        if self.pool is not None:
            self.pool.reset()
        else:
            for graph in self.graphs:
                graph.close()
        self.tensors.clear()
        self.signature = None
