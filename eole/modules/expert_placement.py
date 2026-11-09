"""Inference placement policy: checkpoint format does not decide execution device."""

import torch


def validate_expert_placement(config):
    if getattr(config, "moe_expert_backend", "resident") == "resident":
        return
    from eole import EOLE_TORCH_COMPILE

    if EOLE_TORCH_COMPILE:
        raise ValueError("Hybrid MoE currently requires EOLE_TORCH_COMPILE=0; host execution cannot use decoder graphs")
    if getattr(config, "parallel_gpu", 1) != 1 or getattr(config, "world_size", 1) != 1:
        raise ValueError("Hybrid MoE currently requires one GPU")
    if getattr(config, "compute_dtype", torch.bfloat16) != torch.bfloat16:
        raise ValueError("Hybrid MoE currently requires compute_dtype=bf16")
    if getattr(config, "moe_expert_backend", "resident") == "hybrid_gptq":
        if getattr(config, "quant_type", "") != "autoround":
            raise ValueError("Hybrid GPTQ requires a GPTQ/autoround INT4 checkpoint")
        if getattr(config, "moe_gpu_experts", 16) < 1:
            raise ValueError("Hybrid GPTQ requires at least one GPU cache slot")
    elif getattr(config, "quant_type", ""):
        raise ValueError("Hybrid MoE CPU INT8 requires an unquantized source checkpoint")
    if getattr(config, "self_speculative_decoding", False) or getattr(config, "dflash2_model", None):
        raise ValueError("Hybrid MoE currently requires ordinary decoding")


class ExpertWeightPlacement:
    """Keep host experts off CUDA while loading, before any runtime packing."""

    def __init__(self, model, config, device):
        self.device = device
        self.host_experts = set()
        if getattr(config, "moe_expert_backend", "resident") == "resident":
            return
        validate_expert_placement(config)
        if device.type != "cuda":
            raise ValueError("Hybrid MoE requires a CUDA execution device")
        from eole.modules.moe import MoE

        host_experts = set()
        quantized = config.moe_expert_backend == "hybrid_gptq"
        resident = 0 if quantized else config.moe_gpu_experts
        for name, module in model.named_modules():
            if isinstance(module, MoE):
                if module.activation_function != "silu" or module.experts[0].up_proj is None:
                    raise ValueError("Hybrid MoE currently requires gated SiLU experts")
                if quantized:
                    module._defer_expert_post_init = True
                    if config.moe_gpu_experts > len(module.experts):
                        raise ValueError("moe_gpu_experts exceeds the number of experts")
                if resident >= len(module.experts):
                    raise ValueError("moe_gpu_experts must leave at least one CPU expert per layer")
                root = f".{name}.experts" if name else ".experts"
                host_experts.update(f"{root}.{i}" for i in range(resident, len(module.experts)))
        if not host_experts:
            raise ValueError("Hybrid MoE requires a model with routed experts")
        self.host_experts = host_experts

    def device_for(self, name):
        prefix, separator, suffix = ("." + name).partition(".experts.")
        if separator and f"{prefix}.experts.{suffix.split('.', 1)[0]}" in self.host_experts:
            return torch.device("cpu")
        return self.device

    def dtype_for(self, name, compute_dtype):
        # Host weights are packed later; preserve their source dtype here to
        # avoid a full FP16->BF16 CPU copy during checkpoint loading.
        return None if self.device_for(name) != self.device else compute_dtype
