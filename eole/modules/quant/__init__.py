"""eole.modules.quant: quantized linear backends and the quant_type registry.

Backends (autoround/awq/bnb/gguf/compressed_float_linear, marlin_*, int8_gemv,
dense_quantization, quantization_loading/selection, vocabulary_quantization)
are selected by ``InferenceConfig.quant_type`` through :data:`QUANT_BACKENDS`:
each handler receives ``(model, running_config, quant_targets,
nonlora_to_quant)`` and replaces matching linears in place. Register a new
backend with :func:`register_quant_backend`.
"""

import torch


def _target_prefix(model, target):
    return next(name for name, module in model.named_modules() if module is target)


def _apply_bnb(model, running_config, quant_targets, nonlora_to_quant):
    try:
        from eole.modules.quant.bnb_linear import replace_bnb_linear
    except ImportError:
        raise ImportError("Install bitsandbytes to use 4/8bit compression")
    # try to do this inplace, not sure it'll work
    for quant_target in quant_targets:
        replace_bnb_linear(
            quant_target,
            module_to_convert=nonlora_to_quant,
            q_type=running_config.quant_type,
            module_to_not_convert=getattr(running_config, "quant_exclude_modules", []),
            prefix=_target_prefix(model, quant_target),
        )


def _apply_awq(model, running_config, quant_targets, nonlora_to_quant):
    try:
        from eole.modules.quant.awq_linear import replace_awq_linear
    except ImportError:
        raise ImportError("Install AutoAWQ to use awq quantized model")
    # try to do this inplace, not sure it'll work
    for quant_target in quant_targets:
        replace_awq_linear(
            quant_target,
            module_to_convert=nonlora_to_quant,
            w_bit=running_config.w_bit,
            group_size=running_config.group_size,
            q_type=running_config.quant_type,
            module_to_not_convert=getattr(running_config, "quant_exclude_modules", []),
            prefix=_target_prefix(model, quant_target),
        )


def _apply_autoround(model, running_config, quant_targets, nonlora_to_quant):
    try:
        from eole.modules.quant.autoround_linear import replace_autoround_linear
    except ImportError:
        raise ImportError("Install auto-round to use autoround quantized model")
    for quant_target in quant_targets:
        replace_autoround_linear(
            quant_target,
            module_to_convert=nonlora_to_quant,
            w_bit=running_config.w_bit,
            group_size=running_config.group_size,
            packing_format=getattr(running_config, "autoround_packing_format", "auto_round:auto_gptq"),
            sym=getattr(running_config, "autoround_sym", True),
            module_to_not_convert=getattr(running_config, "quant_exclude_modules", []),
            prefix=_target_prefix(model, quant_target),
        )


def _apply_nvfp4(model, running_config, quant_targets, nonlora_to_quant):
    from eole.modules.quant.compressed_float_linear import (
        CompressedFloatLinear,
        replace_compressed_float_linear,
    )
    from eole.modules.quant.quantization_selection import is_selected

    # Vision checkpoints keep a native head unless conversion explicitly mapped it as packed.
    if "generator" in nonlora_to_quant and is_selected(
        "generator", nonlora_to_quant, getattr(running_config, "quant_exclude_modules", [])
    ):
        if model.share_decoder_embeddings:
            raise ValueError("Compressed vocabulary heads cannot share decoder embeddings")
        model.generator = CompressedFloatLinear(
            model.generator.in_features, model.generator.out_features, model.generator.bias is not None
        )
    if getattr(running_config, "parallel_gpu", 1) != 1:
        raise ValueError("NVFP4 inference currently requires a single GPU")
    if running_config.compute_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("NVFP4 inference requires compute_dtype=fp16 or bf16")
    for quant_target in quant_targets:
        replace_compressed_float_linear(
            quant_target,
            nonlora_to_quant,
            getattr(running_config, "quant_exclude_modules", []),
            _target_prefix(model, quant_target),
        )


def _apply_gguf(model, running_config, quant_targets, nonlora_to_quant):
    try:
        from eole.modules.quant.gguf_linear import replace_gguf_linear
    except ImportError:
        raise ImportError("Install gguf to use GGUF quantized models: pip install gguf")
    for quant_target in quant_targets:
        replace_gguf_linear(
            quant_target,
            module_to_convert=nonlora_to_quant,
            module_to_not_convert=getattr(running_config, "quant_exclude_modules", []),
            prefix=_target_prefix(model, quant_target),
        )


# Keyed by InferenceConfig.quant_type. Handlers import their backend lazily so
# importing eole.modules.quant never requires bitsandbytes/AutoAWQ/gguf.
QUANT_BACKENDS = {
    "bnb_8bit": _apply_bnb,
    "bnb_FP4": _apply_bnb,
    "bnb_NF4": _apply_bnb,
    "awq_gemm": _apply_awq,
    "awq_gemv": _apply_awq,
    "autoround": _apply_autoround,
    "nvfp4": _apply_nvfp4,
    "gguf": _apply_gguf,
}


def register_quant_backend(quant_type, handler):
    """Register a handler for the ``quant_type`` config value."""
    QUANT_BACKENDS[quant_type] = handler


def apply_quant_backend(quant_type, model, running_config, quant_targets, nonlora_to_quant):
    """Apply the registered backend for ``quant_type``; return False if unknown."""
    handler = QUANT_BACKENDS.get(quant_type)
    if handler is None:
        return False
    handler(model, running_config, quant_targets, nonlora_to_quant)
    return True
