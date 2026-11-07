"""Logical FlashAttention cache capacity without moving its backing storage."""

import torch


def bounded_kv_cache(k_cache, v_cache, bound=None, block_table=None):
    """Apply a request-wide CPU bound inside the opaque attention operation.

    Reading a CPU scalar here adds no GPU synchronization and keeps the bound's
    value outside the compiled decoder graph. CUDA capture records the selected
    view; callers must reset graph bindings before changing the bound.
    """
    if bound is None or block_table is not None:
        return k_cache, v_cache
    if bound.device.type != "cpu" or bound.numel() != 1 or bound.dtype != torch.int64:
        raise ValueError("KV-cache view bound must be one CPU int64 value")
    capacity = int(bound.item())
    if capacity <= 0:
        raise ValueError("KV-cache view bound must be positive")
    capacity = min(capacity, k_cache.size(1))
    return k_cache[:, :capacity], v_cache[:, :capacity]
