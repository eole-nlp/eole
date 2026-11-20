"""Replay accepted GDN inputs directly into persistent per-layer state."""

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp


@triton.jit(do_not_specialize=["n_kept"])
def _recurrent_commit(pointers, n_kept, HEADS: tl.constexpr, BV: tl.constexpr):
    layer = tl.program_id(2)
    head = tl.program_id(1)
    # Tensor arguments carry alignment metadata; pointers loaded from a table
    # do not. Dispatch guarantees 64-byte addresses (16 FP32 elements), so
    # restore this information on the typed pointers. This enables vectorized
    # access and preserves FLA's FP32 reduction layout.
    keys = tl.multiple_of(tl.load(pointers + layer * 7).to(tl.pointer_type(tl.bfloat16)), 16)
    values = tl.multiple_of(tl.load(pointers + layer * 7 + 1).to(tl.pointer_type(tl.bfloat16)), 16)
    decay = tl.multiple_of(tl.load(pointers + layer * 7 + 2).to(tl.pointer_type(tl.float32)), 16)
    betas = tl.multiple_of(tl.load(pointers + layer * 7 + 3).to(tl.pointer_type(tl.bfloat16)), 16)
    state = tl.multiple_of(tl.load(pointers + layer * 7 + 4).to(tl.pointer_type(tl.bfloat16)), 16)
    k = tl.arange(0, 128)
    v = tl.program_id(0) * BV + tl.arange(0, BV)
    offsets = head * 128 * 128 + k[:, None] * 128 + v[None, :]
    h = tl.load(state + offsets).to(tl.float32)
    for token in tl.range(0, n_kept):
        key = tl.load(keys + (token * HEADS + head) * 128 + k).to(tl.float32)
        key = key / tl.sqrt(tl.sum(key * key) + 1e-6)
        value = tl.load(values + (token * HEADS + head) * 128 + v).to(tl.float32)
        g = tl.load(decay + token * HEADS + head).to(tl.float32)
        beta = tl.load(betas + token * HEADS + head).to(tl.float32)
        h *= exp(g)
        delta = beta * (value - tl.sum(h * key[:, None], 0))
        h += key[:, None] * delta
    # One BF16 rounding after replay, matching the existing FP32 replay and
    # final cast. No query normalization or attention output is needed here.
    tl.store(state + offsets, h)


@triton.jit
def _conv_commit(pointers, N: tl.constexpr, CHANNELS: tl.constexpr, KERNEL: tl.constexpr):
    layer = tl.program_id(0)
    i = tl.program_id(1) * 256 + tl.arange(0, 256)
    source = tl.multiple_of(tl.load(pointers + layer * 7 + 5).to(tl.pointer_type(tl.bfloat16)), 16)
    target = tl.multiple_of(tl.load(pointers + layer * 7 + 6).to(tl.pointer_type(tl.bfloat16)), 16)
    values = tl.load(source + (i // KERNEL) * (KERNEL + 4) + N + i % KERNEL, i < CHANNELS * KERNEL, other=0)
    tl.store(target + i, values, i < CHANNELS * KERNEL)


def commit_states(pointers, layers, n_kept, heads, channels, kernel):
    with torch.cuda.device(pointers.device):
        _recurrent_commit[(16, heads, layers)](pointers, n_kept, heads, 8, num_warps=1, num_stages=3)
        _conv_commit[(layers, triton.cdiv(channels * kernel, 256))](pointers, n_kept, channels, kernel, num_warps=4)
