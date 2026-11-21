"""Split-KV attention with device-side lengths and fixed CUDA graph geometry."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["KS", "VS"])
def _append(
    K,
    V,
    KC,
    VC,
    L,
    T: tl.constexpr,
    C: tl.constexpr,
    HD: tl.constexpr,
    KS,
    VS,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    row = token // T
    length = tl.load(L + row)
    position = length.to(tl.int64) + token % T
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    key = tl.load(K + token * HD + d, d < HD, 0).to(tl.float32) / KS
    value = tl.load(V + token * HD + d, d < HD, 0).to(tl.float32) / VS
    dst = (row * C + position) * HD + d
    valid = (d < HD) & (length >= 0) & (length <= C - T)
    tl.store(KC + dst, tl.minimum(tl.maximum(key, -448.0), 448.0), valid)
    tl.store(VC + dst, tl.minimum(tl.maximum(value, -448.0), 448.0), valid)


def append(key, value, key_cache, value_cache, lengths, key_scale, value_scale):
    batch, tokens, heads, dim = key.shape
    width = heads * dim
    block = min(1024, triton.next_power_of_2(width))
    _append[(batch * tokens, triton.cdiv(width, block))](
        key.contiguous(),
        value.contiguous(),
        key_cache,
        value_cache,
        lengths,
        tokens,
        key_cache.size(1),
        width,
        key_scale,
        value_scale,
        block,
    )


@triton.jit(do_not_specialize=["SM", "KS", "VS"])
def _split(
    Q,
    K,
    V,
    L,
    PART,
    LSE,
    T: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
    S: tl.constexpr,
    CHUNK: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    SM,
    KS,
    VS,
):
    row = tl.program_id(0)
    head = tl.program_id(1) % HK
    query_tile = tl.program_id(1) // HK
    split = tl.program_id(2)
    start = split * CHUNK
    length = tl.load(L + row)
    if length >= 0 and length <= C - T and start < length + T:
        group: tl.constexpr = H // HK
        m = query_tile * M + tl.arange(0, M)
        n = tl.arange(0, N)
        d = tl.arange(0, D)
        token = m // group
        qhead = head * group + m % group
        out_row = (row * T + token) * H + qhead
        q = tl.load(Q + out_row[:, None] * D + d[None, :], token[:, None] < T, 0)
        maximum = tl.full((M,), -float("inf"), tl.float32)
        denom = tl.full((M,), 0, tl.float32)
        acc = tl.full((M, D), 0, tl.float32)
        for i in range(start, tl.minimum(start + CHUNK, length + T), N):
            pos = i + n
            offset = ((row.to(tl.int64) * C + pos[:, None]) * HK + head) * D + d[None, :]
            valid = (pos < length + T) & (pos < C)
            k = tl.where(valid[:, None], tl.load(K + offset, valid[:, None]).to(tl.float32), 0.0) * KS
            v = tl.where(valid[:, None], tl.load(V + offset, valid[:, None]).to(tl.float32), 0.0) * VS
            scores = tl.dot(q, tl.trans(k.to(q.dtype))) * SM
            scores = tl.where(valid[None, :] & (pos[None, :] <= length + token[:, None]), scores, -float("inf"))
            new_max = tl.maximum(maximum, tl.max(scores, 1))
            safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
            old = tl.exp(maximum - safe_max)
            p = tl.exp(scores - safe_max[:, None])
            acc = acc * old[:, None] + tl.dot(p.to(q.dtype), v.to(q.dtype))
            denom = denom * old + tl.sum(p, 1)
            maximum = new_max
        idx = out_row * S + split
        tl.store(PART + idx[:, None] * D + d[None, :], acc / tl.maximum(denom[:, None], 1e-20), token[:, None] < T)
        tl.store(LSE + idx, maximum + tl.log(denom), token < T)


@triton.jit
def _merge(
    PART,
    LSE,
    L,
    OUT,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK_S: tl.constexpr,
    C: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // H % T
    length = tl.load(L + row // (T * H))
    s = tl.arange(0, BLOCK_S)
    d = tl.arange(0, D)
    valid = (s < S) & (s * CHUNK <= length + token) & (length >= 0) & (length <= C - T)
    offset = row * S + s
    lse = tl.load(LSE + offset, valid, -float("inf"))
    max_lse = tl.max(lse, 0)
    weights = tl.exp(lse - tl.where(max_lse == -float("inf"), 0.0, max_lse))
    partial = tl.load(PART + offset[:, None] * D + d[None, :], valid[:, None], 0)
    result = tl.sum(partial * weights[:, None], 0) / tl.maximum(tl.sum(weights, 0), 1e-20)
    tl.store(OUT + row * D + d, result)


def split_attention(query, key_cache, value_cache, lengths, scale, key_scale, value_scale):
    batch, tokens, heads, dim = query.shape
    capacity, kv_heads = key_cache.shape[1:3]
    chunk = 512
    splits = triton.cdiv(capacity, chunk)
    partial = torch.empty((batch, tokens, heads, splits, dim), device=query.device, dtype=torch.float32)
    lse = torch.empty(partial.shape[:-1], device=query.device, dtype=torch.float32)
    result = torch.empty_like(query)
    query_rows = tokens * (heads // kv_heads)
    tile = min(32, max(16, triton.next_power_of_2(query_rows)))
    _split[(batch, kv_heads * triton.cdiv(query_rows, tile), splits)](
        query,
        key_cache,
        value_cache,
        lengths,
        partial,
        lse,
        tokens,
        heads,
        kv_heads,
        dim,
        capacity,
        splits,
        chunk,
        tile,
        32,
        scale,
        key_scale,
        value_scale,
    )
    _merge[(batch * tokens * heads,)](
        partial, lse, lengths, result, tokens, heads, dim, splits, chunk, triton.next_power_of_2(splits), capacity
    )
    return result
