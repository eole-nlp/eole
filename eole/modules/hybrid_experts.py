"""Experimental full-routing CPU INT8 / GPU dense expert execution.

Only activations, routing and results cross PCIe. Expert weights stay on their
assigned device. This first executor uses a fixed expert partition, not an
adaptive cache, and stays outside whole-decoder torch.compile. Supported GPU phases use CUDA graphs.
"""

from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn.functional as F


class CPUInt8Linear:
    """Per-channel INT8 weights and BF16 activations using PyTorch's CPU kernel.

    Keep the backend call here so a future native/torchao implementation can
    replace it without changing routing, placement or the CPU/GPU scheduler.
    """

    def __init__(self, linear):
        if type(linear) is not torch.nn.Linear or linear.weight.device.type != "cpu":
            raise ValueError("CPU INT8 experts require plain CPU Linear weights")
        weight = linear.weight.detach().float()
        scales = weight.abs().amax(dim=1).clamp_min(1e-8) / 127
        self.weight = (weight / scales[:, None]).round().clamp(-127, 127).to(torch.int8).contiguous()
        self.scales = scales.to(torch.bfloat16)
        self.bias = linear.bias.detach().to(torch.bfloat16) if linear.bias is not None else None

    def __call__(self, x):
        result = torch.ops.aten._weight_int8pack_mm(x.to(torch.bfloat16).contiguous(), self.weight, self.scales)
        return result + self.bias if self.bias is not None else result


class CPUInt8Expert:
    def __init__(self, expert):
        if expert.activation is not F.silu:
            raise ValueError("CPU INT8 experts currently require gated SiLU")
        self.gate = CPUInt8Linear(expert.gate_up_proj)
        self.up = CPUInt8Linear(expert.up_proj)
        self.down = CPUInt8Linear(expert.down_proj)

    def __call__(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class HybridExpertRuntime:
    """One CPU worker shared by all layers of a single-request model."""

    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="eole-experts")
        self.executors = []
        self._graph_streams = {}

    def graph_stream(self, device):
        # BLAS workspaces are stream-specific. Share a capture stream across
        # stages/layers, but never share graph memory pools (outputs overlap in
        # lifetime across the attention -> expert -> shared-MLP boundary).
        key = torch.device(device)
        if key not in self._graph_streams:
            self._graph_streams[key] = torch.cuda.Stream(device=key)
        return self._graph_streams[key]

    def close(self):
        self.pool.shutdown(wait=True)
        for executor in self.executors:
            executor.close()
        self.executors.clear()
        self._graph_streams.clear()


class HybridExpertExecutor:
    def __init__(self, experts, resident, runtime):
        self.experts = experts
        self.resident = resident
        self.runtime = runtime
        self.cpu_experts = {}
        for index in range(resident, len(experts)):
            self.cpu_experts[index] = CPUInt8Expert(experts[index])
            # Free BF16 source weights as each expert is packed. Keep ModuleList
            # indices stable, without retaining a second copy of host weights.
            experts[index] = torch.nn.Identity()

    @torch.inference_mode()
    def _run_cpu(self, x, ids):
        if x.shape[0] == 1:
            # Top-k selects distinct experts for one token. Avoid regrouping,
            # per-expert nonzero indices and copies of the same input row.
            # CPUInt8Expert returns BF16; keeping that dtype also avoids an
            # FP32 round trip before execute() uploads BF16 results to CUDA.
            parts = torch.zeros(1, ids.shape[1], x.shape[1], dtype=torch.bfloat16)
            for position, expert_id in enumerate(ids[0].tolist()):
                if expert_id >= self.resident:
                    parts[0, position] = self.cpu_experts[expert_id](x)[0]
            return parts
        parts = torch.zeros(ids.shape[0], ids.shape[1], x.shape[1], dtype=torch.float32)
        for expert_id in torch.unique(ids).tolist():
            if expert_id < self.resident:
                continue
            token, position = torch.where(ids == expert_id)
            parts[token, position] = self.cpu_experts[expert_id](x[token]).float()
        return parts

    @torch.compiler.disable
    def execute(self, x, weights, ids, *, training=False):
        if training or torch.is_grad_enabled():
            raise ValueError("Hybrid experts support inference only")
        # One routing handoff per layer. The worker receives owned CPU tensors;
        # it never touches CUDA and cannot race buffer reuse on the next layer.
        host_ids = ids.cpu()
        host_x = x.cpu()
        future = self.runtime.pool.submit(self._run_cpu, host_x, host_ids)
        parts = torch.zeros(*ids.shape, x.shape[-1], device=x.device, dtype=x.dtype)
        try:
            for expert_id in torch.unique(host_ids).tolist():
                if expert_id >= self.resident:
                    continue
                token, position = torch.where(ids == expert_id)
                parts[token, position] = self.experts[expert_id](x[token])
            # Finish GPU work while the CPU worker runs, then upload only its
            # outputs. Keep the canonical routed order for the final reduction.
            parts += future.result().to(device=x.device, dtype=x.dtype)
        except BaseException:
            # Preserve the initiating exception while draining any CPU work.
            try:
                future.result()
            except Exception:
                pass
            raise
        return (parts * weights.unsqueeze(-1)).sum(dim=1)

    def _maybe_init_quant_weights(self, device, dtype):
        pass  # Packed once during startup, never stack CPU experts onto CUDA.


def prepare_hybrid_experts(model, config):
    if getattr(config, "moe_expert_backend", "resident") == "resident":
        return
    from eole.modules.moe import MoE
    from eole.modules.expert_placement import validate_expert_placement

    validate_expert_placement(config)
    if getattr(model, "_expert_runtime", None) is not None:
        return
    runtime = HybridExpertRuntime()
    try:
        from eole.utils.logging import logger

        logger.info(
            "Preparing hybrid MoE: %s, %d GPU experts per layer", config.moe_expert_backend, config.moe_gpu_experts
        )
        for module in model.modules():
            if isinstance(module, MoE):
                module._executor = HybridExpertExecutor(module.experts, config.moe_gpu_experts, runtime)
        if getattr(config, "moe_expert_cuda_graph", True):
            from eole.modules.hybrid_graph import HybridCacheStorage, HybridLayerGraph

            decoder = getattr(model, "decoder", None)
            if decoder is not None and not decoder.dynamic_shapes:
                graphs = []
                for layer in decoder.transformer_layers:
                    if (
                        hasattr(layer.mlp, "_executor")
                        and not layer.context_attn
                        and not layer.ffn_layernorm
                        and not layer.parallel_residual
                        and layer.dropout_p == 0
                        and decoder.self_attn_backend == "flash"
                    ):
                        graph = HybridLayerGraph(layer, runtime.graph_stream(layer.mlp.gate.weight.device))
                        layer._hybrid_graph = graph
                        runtime.executors.append(graph)
                        graphs.append(graph)
                if graphs:
                    decoder._hybrid_cache_storage = HybridCacheStorage(graphs)
                    runtime.executors.append(decoder._hybrid_cache_storage)
        model._expert_runtime = runtime
    except BaseException:
        runtime.close()
        raise
