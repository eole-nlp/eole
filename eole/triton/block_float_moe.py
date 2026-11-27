"""Weight-only NVFP4 and MXFP8 expert projections for prefill and decode."""

import torch
import triton
import triton.language as tl


@triton.jit
def _load_weight_tile(W, S, weight_row, k, valid, outer_scale, K: tl.constexpr, NVFP4: tl.constexpr):
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
        return (fp4 * scale * outer_scale).to(tl.bfloat16)
    else:
        fp8 = tl.load(W + weight_row * K + k[None, :], valid, 0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        shift = tl.load(S + weight_row * (K // 32) + k[None, :] // 32, valid, 127).to(tl.float32) - 127
        return (fp8 * tl.exp2(shift)).to(tl.bfloat16)


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
    outer_scale = tl.load(OUTER + expert) if NVFP4 else 1.0
    w = _load_weight_tile(W, S, weight_row, k, valid, outer_scale, K, NVFP4)
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


@triton.jit(do_not_specialize=["num_routes"])
def _grouped_gemm(
    X,
    W,
    S,
    OUTER,
    SORTED_ROUTES,
    EXPERT_IDS,
    PADDED_COUNT,
    OUT,
    num_routes,
    TOP_K: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NVFP4: tl.constexpr,
    SHARED_X: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # One program row handles one expert's padded block of token routes.
    # Read the active count on device to keep launch sizing free of CPU syncs.
    block_start = tl.program_id(0) * BLOCK_M
    if block_start >= tl.load(PADDED_COUNT):
        return

    route = tl.load(SORTED_ROUTES + block_start + tl.arange(0, BLOCK_M))
    expert = tl.load(EXPERT_IDS + tl.program_id(0))
    outer_scale = tl.load(OUTER + expert) if NVFP4 else 1.0
    n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    cols = tl.arange(0, BLOCK_K)
    input_row = route // TOP_K if SHARED_X else route
    weight_row = (expert * N + n)[:, None]
    accum = tl.full((BLOCK_M, BLOCK_N), 0, tl.float32)
    for step in range(tl.cdiv(K, BLOCK_K)):
        k = step * BLOCK_K + cols
        x = tl.load(X + input_row[:, None] * K + k[None, :], (route[:, None] < num_routes) & (k[None, :] < K), 0)
        valid = (n[:, None] < N) & (k[None, :] < K)
        w = _load_weight_tile(W, S, weight_row, k, valid, outer_scale, K, NVFP4)
        accum = tl.dot(x, tl.trans(w), accum)

    tl.store(OUT + route[:, None] * N + n[None, :], accum, (route[:, None] < num_routes) & (n[None, :] < N))


def grouped_experts(experts, inputs, weights, indices, *, small_batch=False):
    """Run the two expert projections over GPU-grouped token routes."""
    from eole.modules.moe_quant_utils import _moe_align_block_size

    num_experts = experts.gate_up.weight.size(0)
    num_tokens, top_k = indices.shape
    num_routes = num_tokens * top_k
    block_m = 16 if num_routes <= 16 * num_experts else 32
    rows, expert_ids, padded = _moe_align_block_size(indices, block_m, num_experts)

    def project(layer, x, shared_input):
        n, k = layer.weight.size(1), x.size(1)
        out = x.new_empty((num_routes, n))
        # Full prefill amortizes traffic across more token rows. On Ada the
        # short verifier tiles benefit from a longer inner reduction.
        if small_batch:
            block_n, block_k, stages = (32, 128, 2) if layer.nvfp4 else (64, 128, 3)
        else:
            block_n, block_k, stages = 64, 64, 3
        _grouped_gemm[(expert_ids.numel(), triton.cdiv(n, block_n))](
            x.contiguous(),
            layer.weight.view(torch.uint8),
            layer.scales.view(torch.uint8),
            layer.outer_scale if layer.nvfp4 else layer.scales,
            rows,
            expert_ids,
            padded,
            out,
            num_routes,
            top_k,
            n,
            k,
            layer.nvfp4,
            shared_input,
            block_m,
            block_n,
            block_k,
            num_stages=stages,
        )
        return out

    activated = experts._activate(project(experts.gate_up, inputs, shared_input=True))
    routes = project(experts.down, activated, shared_input=False).view(num_tokens, top_k, -1)
    return (routes.float() * weights[..., None]).sum(dim=1).to(inputs.dtype)
