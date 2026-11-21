"""Prototype fixed-capacity FP8 KV attention for unpadded causal sequences.

The caller owns the positions, including advancing and rolling back speculative
writes. Each layer supplies separate positive K and V scales. Lengths are
int32 device positions and must leave room for the complete input block.
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
    key_scale: float = 1.0,
    value_scale: float = 1.0,
) -> torch.Tensor:
    from math import isfinite

    from eole.triton.fp8_kv import append, split_attention

    if (
        key_cache.dtype != torch.float8_e4m3fn
        or value_cache.dtype != key_cache.dtype
        or not key_cache.is_contiguous()
        or not value_cache.is_contiguous()
    ):
        raise ValueError("FP8 attention requires contiguous E4M3 cache tensors")
    if any(not isfinite(x) or x <= 0 for x in (key_scale, value_scale)):
        raise ValueError("FP8 cache scales must be finite and positive")
    batch, tokens, heads, dim = query.shape
    kv_heads = key_cache.size(2)
    if (
        query.dtype not in (torch.bfloat16, torch.float16)
        or not 1 <= tokens <= 16
        or dim not in (64, 128, 256)
        or not kv_heads
        or heads % kv_heads != 0
        or key_cache.shape != value_cache.shape
        or key_cache.shape != (batch, key_cache.size(1), kv_heads, dim)
        or key.shape != (batch, tokens, kv_heads, dim)
        or value.shape != key.shape
        or lengths.shape != (batch,)
        or lengths.dtype != torch.int32
        or not lengths.is_contiguous()
    ):
        raise ValueError("Unsupported FP8 decode attention layout")
    append(key, value, key_cache, value_cache, lengths, key_scale, value_scale)
    return split_attention(query.contiguous(), key_cache, value_cache, lengths, scale, key_scale, value_scale)


@fp8_kv_attention.register_fake
def _(query, key_cache, value_cache, key, value, lengths, scale, key_scale=1.0, value_scale=1.0):
    return torch.empty_like(query)
