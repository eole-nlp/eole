"""Fixed-capacity FP8 KV attention for unpadded causal sequences.

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
    return torch.empty_like(query, memory_format=torch.contiguous_format)


def configure_fp8_cache(decoder, parameter, scales):
    """Configure the target attention layers before allocating a KV cache."""
    from math import isfinite

    if parameter.device.type != "cuda":
        raise ValueError("FP8 KV requires CUDA")
    if decoder.dynamic_shapes or decoder.with_cross_attn or decoder._prefill_cache is not None:
        raise ValueError("FP8 KV requires a fixed decoder-only cache with prefill_cache_size=0")
    if torch.cuda.get_device_capability(parameter.device) < (8, 9):
        raise ValueError("FP8 KV requires CUDA capability 8.9 or newer")
    if not decoder._can_use_flash_cache(parameter.dtype, parameter.device):
        raise ValueError("FP8 KV requires a usable FlashAttention KV-cache backend and BF16 or FP16 compute")
    import flashinfer  # noqa: F401

    if scales is not None and (not isinstance(scales, list) or len(scales) != len(decoder.transformer_layers)):
        raise ValueError("Specify one K/V scale pair per decoder layer (null for GDN layers)")
    configured = []
    for i, layer in enumerate(decoder.transformer_layers):
        pair = scales[i] if scales is not None else None
        if layer.layer_type == "linear_attention":
            if pair is not None:
                raise ValueError("GDN layers do not accept FP8 KV scales")
            continue
        attention = layer.self_attn
        if attention.window_size != (-1, -1) or attention.dim_per_head not in (64, 128, 256):
            raise ValueError("FP8 KV requires full-window attention with head dimension 64, 128 or 256")
        pair = (1.0, 1.0) if pair is None else pair
        if (
            not isinstance(pair, (tuple, list))
            or len(pair) != 2
            or any(type(x) not in (int, float) or not isfinite(x) or x <= 0 for x in pair)
        ):
            raise ValueError(f"Invalid FP8 K/V scales for decoder layer {i}")
        configured.append((i, attention, tuple(map(float, pair))))
    for _, attention, pair in configured:
        attention._fp8_kv_scales = pair
    decoder._verifier_cache_config["fp8_kv"] = [(i, pair) for i, _, pair in configured]


def prefill(query, key_cache, value_cache, key, value, lengths, scale, key_scale, value_scale):
    """Chunked prompt prefill uses FlashInfer's causal FP8 kernel on the live prefix."""
    import flashinfer

    from eole.triton.fp8_kv import append

    if query.size(0) != 1:
        raise ValueError("FP8 prompt prefill requires one unpadded request")
    end = int(lengths[0]) + query.size(1)
    append(key, value, key_cache, value_cache, lengths, key_scale, value_scale)
    return flashinfer.single_prefill_with_kv_cache(
        query[0],
        key_cache[0, :end],
        value_cache[0, :end],
        causal=True,
        backend="fa2",
        sm_scale=scale,
        k_scale=key_scale,
        v_scale=value_scale,
    ).unsqueeze(0)
