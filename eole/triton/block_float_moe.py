"""Weight-only NVFP4 and MXFP8 expert projections for single-token decode."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gemv(
    X,
    W,
    S,
    OUTER,
    IDX,
    OUT,
    N: tl.constexpr,
    K: tl.constexpr,
    NVFP4: tl.constexpr,
    SHARED_X: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    route = tl.program_id(1)
    expert = tl.load(IDX + route)
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    k = tl.arange(0, COLS)
    valid = (rows[:, None] < N) & (k[None, :] < K)
    x = tl.load(X + (0 if SHARED_X else route * K) + k, k < K, 0).to(tl.float32)
    weight_row = (expert * N + rows)[:, None]
    if NVFP4:
        p = tl.load(W + weight_row * (K // 2) + k[None, :] // 2, valid, 0)
        code = ((p >> ((k & 1) * 4)[None, :]) & 15).to(tl.int32)
        magnitude_code = code & 7
        # Positive E2M1 normal values map to equally spaced FP32 bit patterns.
        # The first two codes represent zero and the subnormal 0.5.
        normal = ((126 << 23) + (magnitude_code << 22)).to(tl.float32, bitcast=True)
        magnitude = tl.where(magnitude_code < 2, magnitude_code.to(tl.float32) * 0.5, normal)
        fp4 = tl.where((code & 8) != 0, -magnitude, magnitude)
        scale = (
            tl.load(S + weight_row * (K // 16) + k[None, :] // 16, valid, 0)
            .to(tl.float8e4nv, bitcast=True)
            .to(tl.float32)
        )
        w = (fp4 * scale * tl.load(OUTER + expert)).to(tl.bfloat16)
    else:
        fp8 = tl.load(W + weight_row * K + k[None, :], valid, 0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        shift = tl.load(S + weight_row * (K // 32) + k[None, :] // 32, valid, 127).to(tl.float32) - 127
        w = (fp8 * tl.exp2(shift)).to(tl.bfloat16)
    out = tl.sum(w.to(tl.float32) * x[None, :], axis=1)
    tl.store(OUT + route * N + rows, out, rows < N)


def expert_gemv(projection, inputs, expert_ids, *, shared_input):
    """Read the selected packed weights directly and accumulate in FP32."""
    output_size = projection.weight.size(1)
    input_size = inputs.size(-1)
    output = inputs.new_empty((expert_ids.numel(), output_size))
    rows = 8 if not projection.nvfp4 and not shared_input else 4
    warps = 8 if not projection.nvfp4 and shared_input else 4
    _gemv[(triton.cdiv(output_size, rows), expert_ids.numel())](
        inputs.contiguous(),
        projection.weight.view(torch.uint8),
        projection.scales.view(torch.uint8),
        projection.outer_scale if projection.nvfp4 else projection.scales,
        expert_ids,
        output,
        output_size,
        input_size,
        projection.nvfp4,
        shared_input,
        rows,
        triton.next_power_of_2(input_size),
        num_warps=warps,
    )
    return output
