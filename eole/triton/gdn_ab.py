"""Paired BF16 GDN A/B projections for single-token decode and verification."""

import torch
import triton
import triton.language as tl


@triton.jit
def _partial(X, W, P, K: tl.constexpr, N: tl.constexpr):
    # Independent K slices expose parallelism that the skinny GEMM lacks.
    n = tl.program_id(0) * 32 + tl.arange(0, 32)
    m = tl.arange(0, 16)
    k = tl.program_id(1) * 128 + tl.arange(0, 128)
    x = tl.load(X + m[:, None] * K + k[None, :], m[:, None] < 4, other=0)
    w = tl.load(W + n[None, :] * K + k[:, None])
    acc = tl.dot(x, w, out_dtype=tl.float32)
    tl.store(P + tl.program_id(1) * (4 * N) + m[:, None] * N + n[None, :], acc, m[:, None] < 4)


@triton.jit
def _reduce(P, Y, N: tl.constexpr, S: tl.constexpr, R: tl.constexpr):
    i = tl.program_id(0) * 16 + tl.arange(0, 16)
    r = tl.arange(0, R)
    values = tl.load(P + r[:, None] * (4 * N) + i[None, :], r[:, None] < S, other=0)
    tl.store(Y + i, tl.sum(values, axis=0))


@triton.jit
def _single_row(X, W, Y, K: tl.constexpr, BK: tl.constexpr):
    n = tl.program_id(0)
    k = tl.arange(0, BK)
    x = tl.load(X + k, k < K, other=0).to(tl.float32)
    w = tl.load(W + n * K + k, k < K, other=0).to(tl.float32)
    tl.store(Y + n, tl.sum(x * w, axis=0))


def project_ab(inputs, weight):
    n, k = weight.shape
    rows = inputs.shape[1]
    output = torch.empty((1, rows, n), device=inputs.device, dtype=inputs.dtype)
    if rows == 1:
        # One CTA per output, without a split-K scratch buffer.
        _single_row[(n,)](inputs, weight, output, k, triton.next_power_of_2(k), num_warps=4)
    else:
        splits = k // 128
        partials = torch.empty((splits, 4, n), device=inputs.device, dtype=torch.float32)
        _partial[(n // 32, splits)](inputs, weight, partials, k, n, num_warps=4)
        _reduce[(4 * n // 16,)](partials, output, n, splits, triton.next_power_of_2(splits), num_warps=4)
    return output
