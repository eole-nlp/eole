"""Group-32 W8A8 projection using DP4A; identical per-row math for every batch size."""

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize(X, Q, S, K: tl.constexpr):
    g, row = tl.program_id(0), tl.program_id(1)
    i = tl.arange(0, 32)
    x = tl.load(X + row * K + g * 32 + i).to(tl.float32)
    scale = tl.maximum(tl.div_rn(tl.max(tl.abs(x)), 127.0), 1.0e-30)
    q = tl.extra.cuda.libdevice.nearbyint(tl.div_rn(x, scale)).to(tl.int32)
    q = tl.minimum(tl.maximum(q, -127), 127) & 255
    packed = tl.sum(tl.reshape(q, (8, 4)) << (tl.arange(0, 4)[None, :] * 8), axis=1)
    tl.store(Q + row * (K // 4) + g * 8 + tl.arange(0, 8), packed)
    tl.store(S + row * (K // 32) + g, scale)


@triton.jit
def _gemv(Q, XS, W, WS, Y, K: tl.constexpr, N: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr):
    row = tl.program_id(1)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK // 4)
    x = tl.load(Q + row * (K // 4) + k, k < K // 4, other=0)
    w = tl.load(W + n[:, None] * (K // 4) + k[None, :], (n[:, None] < N) & (k[None, :] < K // 4), other=0)
    dots = tl.inline_asm_elementwise(
        "dp4a.s32.s32 $0, $1, $2, 0;",
        constraints="=r,r,r",
        args=[w, x[None, :]],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    sums = tl.sum(tl.reshape(dots, (BN, BK // 32, 8)), axis=2)
    g = tl.arange(0, BK // 32)
    xs = tl.load(XS + row * (K // 32) + g, g < K // 32, other=0)
    ws = tl.load(WS + n[:, None] * (K // 32) + g[None, :], (n[:, None] < N) & (g[None, :] < K // 32), other=0)
    y = tl.sum(sums.to(tl.float32) * xs[None, :] * ws.to(tl.float32), axis=1)
    tl.store(Y + row * N + n, y, n < N)


@triton.jit
def _round_inputs(X, Y, K: tl.constexpr):
    g = tl.program_id(0) * 8 + tl.arange(0, 8)
    i = tl.arange(0, 32)
    row = tl.program_id(1)
    offsets = row * K + g[:, None] * 32 + i[None, :]
    x = tl.load(X + offsets, g[:, None] < K // 32, other=0).to(tl.float32)
    scale = tl.maximum(tl.div_rn(tl.max(tl.abs(x), axis=1), 127.0), 1.0e-30)
    q = tl.extra.cuda.libdevice.nearbyint(tl.div_rn(x, scale[:, None]))
    q = tl.minimum(tl.maximum(q, -127), 127)
    tl.store(Y + offsets, q * scale[:, None], g[:, None] < K // 32)


def round_inputs(inputs):
    output = torch.empty_like(inputs)
    k = inputs.shape[-1]
    _round_inputs[(triton.cdiv(k // 32, 8), inputs.numel() // k)](inputs, output, k, num_warps=4)
    return output


def project(inputs, weight, scales):
    k, n = inputs.shape[-1], weight.shape[0]
    rows = inputs.numel() // k
    output = inputs.new_empty((*inputs.shape[:-1], n))
    if not rows:
        return output
    q = torch.empty((rows, k // 4), device=inputs.device, dtype=torch.int32)
    xs = torch.empty((rows, k // 32), device=inputs.device, dtype=torch.float32)
    _quantize[(k // 32, rows)](inputs, q, xs, k, num_warps=4)
    _gemv[(triton.cdiv(n, 4), rows)](q, xs, weight, scales, output, k, n, triton.next_power_of_2(k), 4, num_warps=4)
    return output
