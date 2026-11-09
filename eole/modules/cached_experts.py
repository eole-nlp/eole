"""Single-request hybrid GPTQ executor with a bounded GPU expert cache.

CPU and GPU execute complementary routes using the same published assignment.
Admissions complete before execution; there is deliberately no asynchronous
cache mutation, speculative prefetch or routing approximation here.
"""

from threading import Lock

import torch

from eole.modules.expert_cache import FrequencyExpertCache
from eole.modules.host_gptq_experts import MarlinExpertStore


class ExpertGPUPhase:
    """Owned decode graph; prefill scratch never replaces captured storage.

    The lookup and weight banks have stable addresses. Only their contents may
    change between replays. Returned route values are overwritten next call.
    """

    def __init__(self, tensors, hidden, intermediate, experts, stream=None):
        from eole.modules.marlin_utils import marlin_make_workspace

        self.tensors = tensors
        self.hidden, self.intermediate = hidden, intermediate
        self.lookup = torch.full((experts,), -1, dtype=torch.long, device=tensors[0].device)
        self.workspace = marlin_make_workspace(tensors[0].device)
        self.empty = torch.empty(0, device=tensors[0].device, dtype=torch.int32)
        self.stream = stream
        self.graph = self.inputs = self.output = None
        self.failed = False
        self.prefill_scratch = None

    def _scratch(self, x, ids):
        pairs = ids.numel()
        return (
            torch.empty(pairs * max(2 * self.intermediate, self.hidden), device=x.device, dtype=x.dtype),
            torch.empty(pairs * self.intermediate, device=x.device, dtype=x.dtype),
        )

    def _run(self, x, ids, scratch):
        from eole.modules.fused_moe_marlin import fused_experts_marlin_impl
        from eole.modules.moe_quant_utils import MARLIN_UINT4B8_TYPE_ID

        slots = self.lookup[ids]
        # Misses run through a valid dummy slot and are masked before combining
        # CPU results. Logical IDs never leave the device to build GPU routing.
        mapped = slots.clamp_min(0)
        w1, s1, w2, s2 = self.tensors
        parts = fused_experts_marlin_impl(
            x,
            w1,
            s1,
            self.empty,
            self.empty,
            self.empty,
            w2,
            s2,
            self.empty,
            self.empty,
            self.empty,
            self.workspace,
            torch.ones_like(ids, dtype=torch.float32),
            mapped,
            MARLIN_UINT4B8_TYPE_ID,
            w1.shape[0],
            cache13=scratch[0],
            cache2=scratch[1],
            return_per_route=True,
        )
        # A dummy expert may overflow even when the actual CPU expert does
        # not. Assignment masking also suppresses NaN/Inf (multiplying by zero
        # would leak them into an otherwise valid CPU result).
        return parts.masked_fill_((slots < 0).unsqueeze(-1), 0)

    def __call__(self, x, ids, capture=True):
        if x.shape[0] != 1 or not capture or self.failed:
            need = ids.numel() * max(2 * self.intermediate, self.hidden)
            if self.prefill_scratch is None or self.prefill_scratch[0].numel() < need:
                self.prefill_scratch = self._scratch(x, ids)
            return self._run(x, ids, self.prefill_scratch)
        if self.graph is not None and (x.shape != self.inputs[0].shape or ids.shape != self.inputs[1].shape):
            raise ValueError("Hybrid expert decode shape changed; reload with the new routing configuration")
        if self.graph is None:
            inputs = (x.clone(), ids.clone())
            scratch = self._scratch(x, ids)
            if self.stream is None:
                self.stream = torch.cuda.Stream(device=x.device)
            stream = self.stream
            current = torch.cuda.current_stream(x.device)
            stream.wait_stream(current)
            graph = None
            try:
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        self._run(*inputs, scratch)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.stream(current):
                    with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
                        output = self._run(*inputs, scratch)
                current.wait_stream(stream)
                self.graph, self.inputs, self.output = graph, inputs, output
                self.decode_scratch = scratch
            except torch.cuda.OutOfMemoryError:
                raise
            except RuntimeError as error:
                current.wait_stream(stream)
                from eole.utils.logging import logger

                if graph is not None:
                    graph.reset()
                logger.warning("Hybrid expert graph capture failed; using eager GPU experts: %s", error)
                self.failed = True
                return self._run(x, ids, scratch)
        else:
            self.inputs[0].copy_(x)
            self.inputs[1].copy_(ids)
        self.graph.replay()
        return self.output

    def close(self):
        self.graph = self.inputs = self.output = self.prefill_scratch = None
        self.decode_scratch = None
        self.tensors = ()
        self.lookup = self.workspace = self.empty = self.stream = None


class CachedGPTQExpertExecutor:
    def __init__(self, experts, runtime, config, device):
        self.runtime = runtime
        self.lock = Lock()
        self.store = MarlinExpertStore(experts, device, getattr(config, "moe_expert_cache_dir", None))
        self.gpu = None
        self.closed = False
        try:
            capacity = config.moe_gpu_experts
            self.policy = FrequencyExpertCache(
                len(experts), capacity, config.moe_expert_cache_interval, config.moe_expert_cache_swaps
            )
            self.mapping = {i: i for i in range(capacity)}
            self.resident_ids = frozenset(self.mapping)
            self.spare = capacity
            banks = [
                torch.empty((capacity + 1, *t.shape[1:]), device=device, dtype=t.dtype) for t in self.store.tensors
            ]
            self.gpu = ExpertGPUPhase(
                banks, self.store.hidden, self.store.intermediate, len(experts), runtime.graph_stream(device)
            )
            for expert, slot in self.mapping.items():
                self._upload(slot, expert)
            self._publish()
            self.fully_resident = capacity == len(experts)
            self.adaptive = config.moe_expert_cache_policy == "frequency" and not self.fully_resident
            self.capture = config.moe_expert_cuda_graph
            self.routes = self.lookup_hits = self.gpu_hits = self.migrations = 0
        except BaseException:
            self.close()
            raise

    def _upload(self, slot, expert):
        for target, source in zip(self.gpu.tensors, self.store.tensors):
            target[slot].copy_(source[expert])

    def _publish(self):
        lookup = torch.full((len(self.store.cpu_experts),), -1, dtype=torch.long)
        for expert, slot in self.mapping.items():
            lookup[expert] = slot
        self.gpu.lookup.copy_(lookup)
        self.resident_ids = frozenset(self.mapping)

    def _admit(self, host_ids):
        if not self.adaptive:
            return
        self.policy.observe(host_ids)
        for old, new in self.policy.replacements():
            # Upload into an unpublished spare. On failure, old weights and
            # both lookup tables stay intact. Publish after each completed swap.
            self._upload(self.spare, new)
            old_slot = self.mapping.pop(old)
            self.mapping[new] = self.spare
            self.spare = old_slot
            self.policy.commit(old, new)
            self._publish()
            self.migrations += 1

    @torch.inference_mode()
    def _run_cpu(self, x, ids, resident):
        parts = torch.zeros(*ids.shape, x.shape[-1], dtype=torch.bfloat16)
        if x.shape[0] == 1:
            for position, expert in enumerate(ids[0].tolist()):
                if expert not in resident:
                    parts[0, position] = self.store.cpu_experts[expert](x)[0]
        else:
            for expert in torch.unique(ids).tolist():
                if expert not in resident:
                    token, position = torch.where(ids == expert)
                    parts[token, position] = self.store.cpu_experts[expert](x[token])
        return parts

    @torch.compiler.disable
    def execute(self, x, weights, ids, *, training=False):
        if self.closed:
            raise RuntimeError("Hybrid expert executor is closed")
        if training or torch.is_grad_enabled():
            raise ValueError("Hybrid experts support inference only")
        # Lock covers worker completion and cache publication. This executor is
        # not a concurrent request scheduler; model caches are request-local too.
        with self.lock:
            if self.fully_resident:
                # No CPU experts or possible admissions: avoid a routing
                # readback when the complete expert set fits on this GPU.
                self.routes += ids.numel()
                self.lookup_hits += ids.numel()
                self.gpu_hits += ids.numel()
                parts = self.gpu(x, ids, self.capture)
                return (parts * weights.unsqueeze(-1)).sum(1)
            host_ids = ids.cpu()
            selected = host_ids.reshape(-1).tolist()
            self.routes += len(selected)
            self.lookup_hits += sum(i in self.mapping for i in selected)
            self._admit(host_ids)
            gpu_hits = sum(i in self.mapping for i in selected)
            self.gpu_hits += gpu_hits
            future = None
            if gpu_hits != len(selected):
                future = self.runtime.pool.submit(self._run_cpu, x.cpu(), host_ids, self.resident_ids)
            try:
                parts = self.gpu(x, ids, self.capture)
                if future is not None:
                    parts.add_(future.result().to(x.device))
                return (parts * weights.unsqueeze(-1)).sum(1)
            except BaseException:
                if future is not None:
                    try:
                        future.result()
                    except Exception:
                        pass
                raise

    def _maybe_init_quant_weights(self, device, dtype):
        pass  # CPU layouts and bounded GPU banks already belong to this executor.

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.gpu is not None:
                self.gpu.close()
        finally:
            self.store.close()
