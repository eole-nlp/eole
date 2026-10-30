"""Inference-only DFlash2 Q/K normalization and split-half RoPE."""

import torch
import triton
import triton.language as tl


@triton.jit
def _norm_rope(
    X,
    W,
    C,
    S,
    Y,
    SX0: tl.constexpr,
    SX1: tl.constexpr,
    SX2: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    t = row % T
    h = (row // T) % H
    b = row // (T * H)
    d = tl.arange(0, D)
    x = tl.load(X + b * SX0 + h * SX1 + t * SX2 + d).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    # Keep every BF16 rounding boundary of the eager implementation. Norm
    # reduction stays FP32; weight multiplication follows the first cast.
    z = (x * inv).to(Y.dtype.element_ty)
    z = (z.to(tl.float32) * tl.load(W + d).to(tl.float32)).to(Y.dtype.element_ty)
    partner = tl.gather(z, (d + D // 2) % D, 0).to(tl.float32)
    partner = tl.where(d < D // 2, -partner, partner)
    c = tl.load(C + t * D + d).to(tl.float32)
    s = tl.load(S + t * D + d).to(tl.float32)
    a = (z.to(tl.float32) * c).to(Y.dtype.element_ty)
    v = (partner * s).to(Y.dtype.element_ty)
    tl.store(Y + row * D + d, a.to(tl.float32) + v.to(tl.float32))


def norm_rope(x, weight, eps, tables):
    """Normalize [B,H,T,D] in FP32 and apply already-rounded RoPE tables."""
    b, h, t, d = x.shape
    out = torch.empty((b, h, t, d), dtype=x.dtype, device=x.device)
    _norm_rope[(b * h * t,)](
        x,
        weight,
        tables[0],
        tables[1],
        out,
        *x.stride()[:3],
        h,
        t,
        d,
        eps,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


@triton.jit
def put_positions(P, C, W: tl.constexpr, T: tl.constexpr):
    t = tl.arange(0, T)
    start = tl.load(C + 2)
    count = tl.load(C + 3)
    tl.store(P + (start + t) % W, start + t, t < count)


@triton.jit
def put_kv(
    K,
    V,
    KC,
    VC,
    C,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    VS0: tl.constexpr,
    VS1: tl.constexpr,
    VS2: tl.constexpr,
    W: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
):
    row = tl.program_id(0)
    h = row // T
    t = row % T
    d = tl.arange(0, D)
    start = tl.load(C + 2)
    count = tl.load(C + 3)
    kval = tl.load(K + h * KS1 + t * KS2 + d)
    vval = tl.load(V + h * VS1 + t * VS2 + d)
    dst = (h * W + (start + t) % W) * D + d
    tl.store(KC + dst, kval, t < count)
    tl.store(VC + dst, vval, t < count)
