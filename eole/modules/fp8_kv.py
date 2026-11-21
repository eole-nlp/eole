"""Prototype fixed-capacity FP8 KV attention for unpadded causal sequences.

The caller owns the positions, including advancing and rolling back speculative
writes. The FA2 masking path reads FP8 directly but schedules the full capacity.
"""

import torch


@torch.library.custom_op("eole::fp8_kv_attention", mutates_args={"key_cache", "value_cache"})
def fp8_kv_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    lengths: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    import flashinfer

    steps = torch.arange(query.size(1), device=query.device)
    positions = lengths[:, None] + steps
    slots = torch.arange(key_cache.size(1), device=query.device)
    output = []
    for row in range(query.size(0)):
        for cache, update in ((key_cache, key), (value_cache, value)):
            # Byte views allow writes on PyTorch builds without FP8 index_copy.
            quantized = update[row].clamp(-448, 448).to(torch.float8_e4m3fn)
            cache[row].view(torch.uint8).index_copy_(0, positions[row].long(), quantized.view(torch.uint8))
        visible = slots[None, :] <= positions[row, :, None]
        mask = flashinfer.packbits(visible.flatten(), bitorder="little")
        output.append(
            flashinfer.single_prefill_with_kv_cache(
                query[row], key_cache[row], value_cache[row], packed_custom_mask=mask, backend="fa2", sm_scale=scale
            )
        )
    return torch.stack(output)


@fp8_kv_attention.register_fake
def _(query, key_cache, value_cache, key, value, lengths, scale):
    return torch.empty_like(query)
