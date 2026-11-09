"""Opt-in weight-only INT8 projections, excluding expert routing and control math."""

import torch
from torch import nn

from eole.modules.vocabulary_quantization import quantize_linear
from eole.utils.logging import logger


# Select projection roles, rather than every Linear: MoE routers, GDN A/B,
# embeddings, vocabulary heads and routed experts must retain their own policy.
_PROJECTIONS = {
    "linear_keys",
    "linear_values",
    "linear_query",
    "linear_kvq",
    "final_linear",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_qkvz",
    "out_proj",
    "gate_up_proj",
    "up_proj",
    "down_proj",
}


def quantize_decoder_layer(layer):
    """Replace eligible decoder projections before ordinary inference fusion.

    Deferred Marlin post_init lets existing QKV/Z and gate/up packing operate on
    the unrepacked layout. The inference setup finalizes each layer immediately,
    releasing its original floating-point projections before processing the next.
    """
    converted = 0
    skipped = []
    for name, projection in list(layer.named_modules()):
        parts = name.split(".")
        if parts[-1] not in _PROJECTIONS or "experts" in parts or type(projection) is not nn.Linear:
            continue
        if (
            projection.weight.device.type != "cuda"
            or projection.weight.dtype not in (torch.float16, torch.bfloat16)
            or projection.bias is not None
            or projection.in_features % 128
            or projection.out_features % 64
        ):
            skipped.append(name)
            continue
        parent, _, leaf = name.rpartition(".")
        replacement = quantize_linear(projection, 8, 32, defer_post_init=True)
        replacement._runtime_dense_int8 = True
        setattr(layer.get_submodule(parent), leaf, replacement)
        converted += 1
    if skipped:
        logger.warning("Dense INT8 retained unsupported projections in native precision: %s", ", ".join(skipped))
    return converted
