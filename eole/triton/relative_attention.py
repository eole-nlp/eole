"""Single-token relative attention over a fixed KV cache."""

import torch
import triton
import triton.language as tl


@triton.jit
def _attention_partials(
    Q,
    K,
    V,
    B,
    P,
    T,
    C: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    R: tl.constexpr,
    W: tl.constexpr,
    S: tl.constexpr,
    BK: tl.constexpr,
    BD: tl.constexpr,
):
    h, part = tl.program_id(0), tl.program_id(1)
    pos = tl.load(P)
    j = part * BK + tl.arange(0, BK)
    x = tl.arange(0, BD)
    # Enumerate live absolute positions, then map sliding positions to ring slots.
    start = tl.maximum(0, pos - W + 1) if W else 0
    p = start + j
    ok = (p <= pos) & (j < C)
    slot = p % C if W else p
    base = (h // R) * C * D
    q = tl.load(Q + h * D + x, x < D, 0).to(tl.float32)
    k = tl.load(K + base + slot[:, None] * D + x[None, :], ok[:, None] & (x[None, :] < D), 0).to(tl.float32)
    # Keep the BF16 rounding of the score path before the FP32 softmax.
    dot = tl.sum(k * q[None, :], 1).to(tl.bfloat16).to(tl.float32)
    distance = pos - p
    bias = tl.load(B + h * E + distance, ok & (distance < E), 0).to(tl.float32)
    score = ((dot * S).to(tl.bfloat16).to(tl.float32) + bias).to(tl.bfloat16).to(tl.float32)
    score = tl.where(ok, score, -float("inf"))
    largest = tl.max(score, 0)
    stable = tl.where(largest == -float("inf"), 0.0, largest)
    weights = tl.exp(score - stable)
    v = tl.load(V + base + slot[:, None] * D + x[None, :], ok[:, None] & (x[None, :] < D), 0).to(tl.float32)
    numerator = tl.sum(weights[:, None] * v, 0)
    off = (h * tl.num_programs(1) + part) * (D + 2)
    tl.store(T + off + x, numerator, x < D)
    tl.store(T + off + D, largest)
    tl.store(T + off + D + 1, tl.sum(weights, 0))


@triton.jit
def _combine_partials(T, SINK, G, OUT, D: tl.constexpr, NS: tl.constexpr, BS: tl.constexpr, BD: tl.constexpr):
    h = tl.program_id(0)
    x = tl.arange(0, BD)
    s = tl.arange(0, BS)
    b = (h * NS + s) * (D + 2)
    largest = tl.load(T + b + D, s < NS, -float("inf"))
    mass = tl.load(T + b + D + 1, s < NS, 0)
    sink = tl.load(SINK + h).to(tl.float32)
    reference = tl.maximum(tl.max(largest, 0), sink)
    weights = tl.exp(largest - reference)
    numerator = tl.load(T + b[:, None] + x[None, :], (s[:, None] < NS) & (x[None, :] < D), 0)
    # The sink contributes to the normalization and has no value vector.
    denom = tl.sum(weights * mass, 0) + tl.exp(sink - reference)
    gate = tl.load(G + h * D + x, x < D, 0).to(tl.float32)
    y = (tl.sum(numerator * weights[:, None], 0) / denom).to(tl.bfloat16).to(tl.float32) * tl.sigmoid(gate)
    tl.store(OUT + h * D + x, y, x < D)


def relative_attention_decode(q, k, v, profiles, sink, gate, pos, window):
    """Decode one BF16 token against contiguous [1, KV heads, capacity, head dim] buffers."""
    h, d = q.shape[-2:]
    c = k.size(-2)
    bk = 64
    ns = triton.cdiv(min(c, window) if window else c, bk)
    tmp = torch.empty((h, ns, d + 2), device=q.device, dtype=torch.float32)
    out = q.new_empty((1, 1, h * d))
    _attention_partials[(h, ns)](
        q,
        k,
        v,
        profiles,
        pos,
        tmp,
        c,
        d,
        profiles.size(-1),
        h // k.size(1),
        window or 0,
        d**-0.5,
        bk,
        triton.next_power_of_2(d),
    )
    _combine_partials[(h,)](tmp, sink, gate, out, d, ns, triton.next_power_of_2(ns), triton.next_power_of_2(d))
    return out


@triton.jit
def _write_kv(
    K,
    V,
    KC,
    VC,
    P,
    C: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    CLIP: tl.constexpr,
    EPS: tl.constexpr,
    B: tl.constexpr,
):
    h = tl.program_id(0)
    x = tl.arange(0, B)
    p = tl.load(P)
    if W:
        p = p % C
    k = tl.load(K + h * D + x, x < D, 0).to(tl.float32)
    v = tl.load(V + h * D + x, x < D, 0).to(tl.float32)
    k = k * tl.minimum(1.0, CLIP / (tl.sqrt(tl.sum(k * k, 0)) + 1e-6))
    v = v * tl.rsqrt(tl.sum(v * v, 0) / D + EPS)
    tl.store(KC + (h * C + p) * D + x, k, x < D)
    tl.store(VC + (h * C + p) * D + x, v, x < D)


def write_normalized_kv(k, v, kc, vc, pos, window, clip, eps):
    """Clip the convolved keys and normalize values while writing the current cache slot."""
    heads, dim = k.shape[-2:]
    _write_kv[(heads,)](
        k, v, kc, vc, pos, kc.size(-2), dim, window or 0, clip, eps, triton.next_power_of_2(dim), num_warps=1
    )
