"""Paired BF16 GDN A/B projections for single-token decode and verification."""

import torch
import triton
import triton.language as tl


@triton.jit
def _partial(X, W, P):
    # Forty independent K slices expose parallelism that the skinny GEMM lacks.
    n = tl.program_id(0) * 32 + tl.arange(0, 32)
    m = tl.arange(0, 16)
    k = tl.program_id(1) * 128 + tl.arange(0, 128)
    x = tl.load(X + m[:, None] * 5120 + k[None, :], m[:, None] < 4, other=0)
    w = tl.load(W + n[None, :] * 5120 + k[:, None])
    acc = tl.dot(x, w, out_dtype=tl.float32)
    tl.store(P + tl.program_id(1) * 384 + m[:, None] * 96 + n[None, :], acc, m[:, None] < 4)


@triton.jit
def _reduce(P, Y):
    i = tl.program_id(0) * 16 + tl.arange(0, 16)
    r = tl.arange(0, 64)
    values = tl.load(P + r[:, None] * 384 + i[None, :], r[:, None] < 40, other=0)
    tl.store(Y + i, tl.sum(values, axis=0))


@triton.jit
def _single_row(X, W, Y):
    n = tl.program_id(0)
    k = tl.arange(0, 8192)
    x = tl.load(X + k, k < 5120, other=0).to(tl.float32)
    w = tl.load(W + n * 5120 + k, k < 5120, other=0).to(tl.float32)
    tl.store(Y + n, tl.sum(x * w, axis=0))


def project_ab(inputs, weight):
    if inputs.shape[1] == 1:
        output = torch.empty((1, 1, 96), device=inputs.device, dtype=inputs.dtype)
        # One CTA per output exposes more parallelism than two skinny GEMVs,
        # without a second launch or a split-K scratch buffer.
        _single_row[(96,)](inputs, weight, output, num_warps=4)
        return output
    output = torch.empty((1, 4, 96), device=inputs.device, dtype=inputs.dtype)
    partials = torch.empty((40, 4, 96), device=inputs.device, dtype=torch.float32)
    _partial[(3, 40)](inputs, weight, partials, num_warps=4)
    _reduce[(24,)](partials, output, num_warps=4)
    return output
