"""Runtime round-to-nearest quantization shared by vocabulary and dense projections."""

import torch
from torch import nn


@torch.no_grad()
def quantize_vocabulary_head(head, bits, group_size=128):
    """Build a Marlin head without modifying the checkpoint or tied embeddings."""
    return quantize_linear(head, bits, group_size)


@torch.no_grad()
def quantize_linear(head, bits, group_size=128, *, defer_post_init=False):
    """Quantize a floating-point projection; optionally leave its GPTQ layout for fusion."""
    if not isinstance(head, nn.Linear):
        raise ValueError("Runtime quantization requires an unquantized nn.Linear projection")
    if head.weight.device.type != "cuda" or head.weight.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Runtime quantization requires CUDA and FP16/BF16 weights")
    if head.bias is not None:
        raise ValueError("Runtime quantization currently requires a bias-free projection")
    if bits not in (4, 8):
        raise ValueError("Runtime quantization supports 4 or 8 bits")
    if head.in_features % group_size:
        raise ValueError("Projection input size must be divisible by the quantization group size")
    from eole.modules.autoround_linear import make_runtime_marlin_linear

    result = make_runtime_marlin_linear(head, bits, group_size)
    # Quantize output rows in bounded CPU chunks, avoiding a full FP32 GPU copy.
    for start in range(0, head.out_features, 256):
        stop = min(start + 256, head.out_features)
        weight = head.weight[start:stop].detach().cpu()
        packed, scales = quantize_rows(weight, bits, group_size)
        result.qweight[:, start:stop].copy_(packed.to(head.weight.device))
        result.scales[:, start:stop].copy_(scales.to(head.weight.device))
    result.qzeros.zero_()  # Unused for symmetric Marlin quantization.
    if defer_post_init:
        result._projection_post_init_deferred = True
    else:
        result.post_init()
    return result


def quantize_rows(weight, bits, group_size):
    """Return GPTQ input-axis packing and rounded scales for output-major rows."""
    groups = weight.float().reshape(weight.size(0), -1, group_size)
    limit = (1 << (bits - 1)) - 1
    scales = (groups.abs().amax(-1) / limit).clamp_min(torch.finfo(weight.dtype).tiny).to(weight.dtype)
    unsigned = (groups / scales.float().unsqueeze(-1)).round().clamp(-limit, limit).to(torch.int64)
    unsigned = (unsigned + (1 << (bits - 1))).reshape(weight.shape).t().contiguous()
    pack = 32 // bits
    shifts = torch.arange(pack, device=weight.device) * bits
    packed = (unsigned.reshape(-1, pack, weight.size(0)) << shifts[None, :, None]).sum(1).to(torch.int32)
    return packed, scales.t().contiguous()
