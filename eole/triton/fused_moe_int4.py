"""
Int4 MoE Triton kernels supporting GPTQ/AutoRound (K-packed) and AWQ (N-packed) formats,
plus a Marlin MoE path using moe_wna16_marlin_gemm kernel.

Layout conventions
------------------
GPTQ / AutoRound (KPACKED=True)
  qweight : (E, K//8,          N)  int32  – 8 int4 values packed along K (input) dim
  scales  : (E, K//group_size, N)  fp16
  qzeros  : (E, K//group_size, N//8) int32 – 8 int4 zero-points packed along N dim

AWQ (KPACKED=False)
  qweight : (E, K,             N//8) int32 – 8 int4 values packed along N (output) dim
  scales  : (E, K//group_size, N)   fp16
  qzeros  : (E, K//group_size, N//8) int32 – same as GPTQ zeros

where  K = in_features (H for W1, I for W2)
       N = out_features (2*I for W1, H for W2)
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# ─────────────────────────────────────────────────────────────────────────────
# Import fused ops and Marlin GEMM from eole.ops (supports torch.compile)
# ─────────────────────────────────────────────────────────────────────────────
from eole.ops import silu_and_mul
from eole.ops import gelu_and_mul
from eole.ops import moe_wna16_marlin_gemm

from eole.modules.moe_quant_utils import _moe_align_block_size  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Buffer view helper
# ─────────────────────────────────────────────────────────────────────────────


def _view_cache(flat: torch.Tensor, *shape: int) -> torch.Tensor:
    """Return a view of *flat* with the given *shape*.

    The total number of elements in *shape* must be ≤ flat.numel().  This lets
    two differently-shaped views share the same underlying storage (e.g. the W1
    GEMM output and the W2 GEMM output both live in the same flat buffer since
    they are never live at the same time).
    """
    n = 1
    for s in shape:
        n *= s
    return flat[:n].view(*shape)


# ---------------------------------------------------------------------------
# W1 kernel: int4 matmul + gated activation (gate and up in one pass)
# ---------------------------------------------------------------------------


@triton.jit
def _w1_int4_act_kernel(
    # ── inputs ──────────────────────────────────────────────────────────────
    X_ptr,  # (M, H)         fp16/bf16  – input tokens
    Qw_ptr,  # see layout doc above
    Sc_ptr,  # (E, K//gs, 2I) fp16       – scales
    Qz_ptr,  # (E, K//gs, 2I//8) int32   – packed zeros
    Y_ptr,  # (num_pairs, I) fp16/bf16  – intermediate output
    expert_ids_ptr,  # (num_pairs,)   int32
    token_ids_ptr,  # (num_pairs,)   int32
    # ── strides ─────────────────────────────────────────────────────────────
    sx_m,
    sx_k,  # X
    sq_e,
    sq_r,
    sq_c,  # Qw  (e / row / col)
    ss_e,
    ss_r,
    ss_c,  # Sc
    sz_e,
    sz_r,
    sz_c,  # Qz
    sy_m,
    sy_n,  # Y
    # ── problem dims (constexpr → compiled-in, no recompile per batch) ──────
    H: tl.constexpr,  # input hidden size
    I: tl.constexpr,  # intermediate size (gate OR up, so W1 width = 2*I)
    group_size: tl.constexpr,  # quantisation group size (e.g. 128)
    # ── layout / activation flags ───────────────────────────────────────────
    KPACKED: tl.constexpr,  # True = GPTQ/AutoRound, False = AWQ
    ACTIVATION: tl.constexpr,  # 0 = SiLU (default), 1 = GELU approx, 2 = ReLU
    # ── tile sizes ──────────────────────────────────────────────────────────
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    # ── packing constants ────────────────────────────────────────────────────
    KPACK: tl.constexpr,  # 8  (int4: 8 values per int32)
    NPACK: tl.constexpr,  # 8  (zero-points also packed ×8 along N)
    # ── dynamic scalar ───────────────────────────────────────────────────────
    num_pairs,  # NOT constexpr – avoids recompile per batch size
):
    pid_pair = tl.program_id(0)
    pid_n = tl.program_id(1)

    if pid_pair >= num_pairs:
        return

    eid = tl.load(expert_ids_ptr + pid_pair).to(tl.int64)
    tid = tl.load(token_ids_ptr + pid_pair).to(tl.int64)

    # Output indices for the gate (first I outputs); up uses offs_n + I
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < I

    acc_gate = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_N,), dtype=tl.float32)

    x_base = X_ptr + tid * sx_m

    for k0 in range(0, H, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < H

        x = tl.load(x_base + offs_k * sx_k, mask=mask_k, other=0.0).to(tl.float32)

        sc_row = offs_k // group_size  # (BLOCK_K,) – which scale row

        # ── load & unpack int4 weights ───────────────────────────────────────
        if KPACKED:
            # GPTQ: qweight[e, k//8, n] – packed along K
            qw_row = offs_k // KPACK  # (BLOCK_K,)
            k_shift = ((offs_k % KPACK) * 4).to(tl.int32)  # (BLOCK_K,)

            # Load (BLOCK_K, BLOCK_N) int32 for gate outputs
            qw_gate = tl.load(
                Qw_ptr + eid * sq_e + qw_row[:, None] * sq_r + offs_n[None, :] * sq_c,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0,
            )
            # Load (BLOCK_K, BLOCK_N) int32 for up outputs (starts at column I)
            qw_up = tl.load(
                Qw_ptr + eid * sq_e + qw_row[:, None] * sq_r + (offs_n[None, :] + I) * sq_c,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0,
            )

            gate_nib = ((qw_gate >> k_shift[:, None]) & 0xF).to(tl.float32)
            up_nib = ((qw_up >> k_shift[:, None]) & 0xF).to(tl.float32)

        else:
            # AWQ: qweight[e, k, n//8] – packed along N
            qw_col_g = offs_n // NPACK  # (BLOCK_N,)
            qw_col_u = (offs_n + I) // NPACK
            n_shift = ((offs_n % NPACK) * 4).to(tl.int32)  # (BLOCK_N,)

            qw_gate = tl.load(
                Qw_ptr + eid * sq_e + offs_k[:, None] * sq_r + qw_col_g[None, :] * sq_c,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0,
            )
            qw_up = tl.load(
                Qw_ptr + eid * sq_e + offs_k[:, None] * sq_r + qw_col_u[None, :] * sq_c,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0,
            )

            gate_nib = ((qw_gate >> n_shift[None, :]) & 0xF).to(tl.float32)
            up_nib = ((qw_up >> n_shift[None, :]) & 0xF).to(tl.float32)

        # ── scales (BLOCK_K, BLOCK_N) ────────────────────────────────────────
        sc_gate = tl.load(
            Sc_ptr + eid * ss_e + sc_row[:, None] * ss_r + offs_n[None, :] * ss_c,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)

        sc_up = tl.load(
            Sc_ptr + eid * ss_e + sc_row[:, None] * ss_r + (offs_n[None, :] + I) * ss_c,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)

        # ── zero-points (always N-packed, both layouts share this convention) ──
        z_col = offs_n // NPACK  # (BLOCK_N,)
        z_shft = ((offs_n % NPACK) * 4).to(tl.int32)  # (BLOCK_N,)
        z_col_u = (offs_n + I) // NPACK

        qz_gate = tl.load(
            Qz_ptr + eid * sz_e + sc_row[:, None] * sz_r + z_col[None, :] * sz_c,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0,
        )
        qz_up = tl.load(
            Qz_ptr + eid * sz_e + sc_row[:, None] * sz_r + z_col_u[None, :] * sz_c,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0,
        )

        z_gate = ((qz_gate >> z_shft[None, :]) & 0xF).to(tl.float32)
        z_up = ((qz_up >> z_shft[None, :]) & 0xF).to(tl.float32)

        # ── dequantize and accumulate ─────────────────────────────────────────
        w_gate = (gate_nib - z_gate) * sc_gate  # (BLOCK_K, BLOCK_N)
        w_up = (up_nib - z_up) * sc_up

        acc_gate += tl.sum(x[:, None] * w_gate, axis=0)
        acc_up += tl.sum(x[:, None] * w_up, axis=0)

    # ── gated activation ─────────────────────────────────────────────────────
    if ACTIVATION == 0:  # SiLU (default for most MoE models)
        act = acc_gate * tl.sigmoid(acc_gate) * acc_up
    elif ACTIVATION == 1:  # GELU approximation
        sqrt_2_over_pi: tl.constexpr = 0.7978845608028654
        g3 = acc_gate * acc_gate * acc_gate
        tanh_arg = sqrt_2_over_pi * (acc_gate + 0.044715 * g3)
        act = 0.5 * acc_gate * (1.0 + tl.math.tanh(tanh_arg)) * acc_up
    else:  # ReLU
        act = tl.where(acc_gate > 0, acc_gate, 0.0) * acc_up

    tl.store(
        Y_ptr + pid_pair * sy_m + offs_n * sy_n,
        act.to(X_ptr.dtype.element_ty),
        mask=mask_n,
    )


# ---------------------------------------------------------------------------
# W2 kernel: int4 matmul + weighted atomic-add reduce into output
# ---------------------------------------------------------------------------


@triton.jit
def _w2_int4_reduce_kernel(
    X_ptr,  # (num_pairs, I) fp16/bf16  – intermediate (from W1 kernel)
    Qw_ptr,  # GPTQ: (E, I//8, H)  AWQ: (E, I, H//8)  int32
    Sc_ptr,  # (E, I//group_size, H)  fp16
    Qz_ptr,  # (E, I//group_size, H//8) int32
    Y_ptr,  # (M, H)         fp16/bf16  – output (atomic accumulation)
    expert_ids_ptr,
    token_ids_ptr,
    weights_ptr,  # (num_pairs,)   fp16/bf16  – routing weights
    sx_m,
    sx_k,
    sq_e,
    sq_r,
    sq_c,
    ss_e,
    ss_r,
    ss_c,
    sz_e,
    sz_r,
    sz_c,
    sy_m,
    sy_n,
    H: tl.constexpr,
    I: tl.constexpr,
    group_size: tl.constexpr,
    KPACKED: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    KPACK: tl.constexpr,
    NPACK: tl.constexpr,
    num_pairs,
):
    pid_pair = tl.program_id(0)
    pid_n = tl.program_id(1)

    if pid_pair >= num_pairs:
        return

    eid = tl.load(expert_ids_ptr + pid_pair).to(tl.int64)
    tid = tl.load(token_ids_ptr + pid_pair).to(tl.int64)
    weight = tl.load(weights_ptr + pid_pair).to(tl.float32)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # output (hidden) indices
    mask_n = offs_n < H

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    x_base = X_ptr + pid_pair * sx_m

    for k0 in range(0, I, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < I

        x = tl.load(x_base + offs_k * sx_k, mask=mask_k, other=0.0).to(tl.float32)

        sc_row = offs_k // group_size

        if KPACKED:
            # GPTQ: qweight[e, k//8, n]  shape (E, I//8, H)
            qw_row = offs_k // KPACK
            k_shift = ((offs_k % KPACK) * 4).to(tl.int32)

            qw = tl.load(
                Qw_ptr + eid * sq_e + qw_row[:, None] * sq_r + offs_n[None, :] * sq_c,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0,
            )
            nib = ((qw >> k_shift[:, None]) & 0xF).to(tl.float32)

        else:
            # AWQ: qweight[e, k, n//8]  shape (E, I, H//8)
            qw_col = offs_n // NPACK
            n_shift = ((offs_n % NPACK) * 4).to(tl.int32)

            qw = tl.load(
                Qw_ptr + eid * sq_e + offs_k[:, None] * sq_r + qw_col[None, :] * sq_c,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0,
            )
            nib = ((qw >> n_shift[None, :]) & 0xF).to(tl.float32)

        sc = tl.load(
            Sc_ptr + eid * ss_e + sc_row[:, None] * ss_r + offs_n[None, :] * ss_c,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)

        z_col = offs_n // NPACK
        z_shft = ((offs_n % NPACK) * 4).to(tl.int32)
        qz = tl.load(
            Qz_ptr + eid * sz_e + sc_row[:, None] * sz_r + z_col[None, :] * sz_c,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0,
        )
        z = ((qz >> z_shft[None, :]) & 0xF).to(tl.float32)

        w = (nib - z) * sc
        acc += tl.sum(x[:, None] * w, axis=0)

    weighted = acc * weight
    y_ptrs = Y_ptr + tid * sy_m + offs_n * sy_n
    tl.atomic_add(y_ptrs, weighted.to(X_ptr.dtype.element_ty), mask=mask_n)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

_ACTIVATION_MAP = {"silu": 0, "gelu": 1, "relu": 2}


def fused_experts_int4_impl(
    hidden_states: torch.Tensor,
    w1_qweight: torch.Tensor,
    w1_scales: torch.Tensor,
    w1_qzeros: torch.Tensor,
    w2_qweight: torch.Tensor,
    w2_scales: torch.Tensor,
    w2_qzeros: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    group_size: int = 128,
    kpacked: bool = True,
    activation: str = "silu",
    # ── Pre-allocated buffers (grown lazily in moe.py) ────────────────────────
    # Eliminates 5 GPU allocations per MoE layer per decode step.
    # When None the function falls back to dynamic allocation.
    intermediate_buf: "torch.Tensor | None" = None,  # flat, numel ≥ num_pairs * I
    output_buf: "torch.Tensor | None" = None,  # (M, H) – zeroed in-place before use
    expert_ids_buf: "torch.Tensor | None" = None,  # (num_pairs,) int32
    token_ids_buf: "torch.Tensor | None" = None,  # (num_pairs,) int32
    weights_buf: "torch.Tensor | None" = None,  # (num_pairs,) dtype
) -> torch.Tensor:
    """MoE forward pass with int4-quantised weights (GPTQ/AWQ).

    Parameters
    ----------
    hidden_states : (M, H)
    w1_qweight    : (E, H//8, 2I) kpacked  or  (E, H, 2I//8) awq   – int32
    w1_scales     : (E, H//gs, 2I)                                   – fp16/bf16
    w1_qzeros     : (E, H//gs, 2I//8)                                – int32
    w2_qweight    : (E, I//8, H)  kpacked  or  (E, I, H//8)  awq   – int32
    w2_scales     : (E, I//gs, H)                                    – fp16/bf16
    w2_qzeros     : (E, I//gs, H//8)                                 – int32
    topk_weights  : (M, K)
    topk_ids      : (M, K)
    group_size    : quantisation group size (default 128)
    kpacked       : True = GPTQ/AutoRound layout, False = AWQ layout
    activation    : "silu" | "gelu" | "relu"
    intermediate_buf / output_buf / expert_ids_buf / token_ids_buf / weights_buf:
                    optional pre-allocated buffers (see moe.py _maybe_grow_int4_buffers)
    """
    M, H = hidden_states.shape
    K = topk_ids.shape[1]
    num_pairs = M * K

    if kpacked:
        I = w1_qweight.shape[2] // 2  # (E, H//8, 2I) → col = 2I  # noqa E741
    else:
        I = w1_qweight.shape[2] * 8 // 2  # (E, H, 2I//8) → col*8 = 2I  # noqa E741

    device = hidden_states.device
    dtype = hidden_states.dtype
    act_code = _ACTIVATION_MAP.get(activation.lower(), 0)
    BLOCK_N, BLOCK_K = 64, 64

    # ── Routing vectors ───────────────────────────────────────────────────────
    if expert_ids_buf is not None and expert_ids_buf.numel() >= num_pairs:
        expert_ids = expert_ids_buf[:num_pairs]
        expert_ids.copy_(topk_ids.flatten().to(torch.int32))
    else:
        expert_ids = topk_ids.flatten().to(torch.int32)

    if token_ids_buf is not None and token_ids_buf.numel() >= num_pairs:
        token_ids = token_ids_buf[:num_pairs]
        # Fill with repeat-interleaved arange: [0,0,...,1,1,...,M-1,M-1,...]
        token_ids.view(M, K).copy_(torch.arange(M, device=device, dtype=torch.int32).unsqueeze(1).expand(M, K))
    else:
        token_ids = torch.arange(M, device=device, dtype=torch.int32).repeat_interleave(K)

    if weights_buf is not None and weights_buf.numel() >= num_pairs:
        weights = weights_buf[:num_pairs]
        weights.copy_(topk_weights.flatten())
    else:
        weights = topk_weights.flatten()

    # ── Intermediate buffer ───────────────────────────────────────────────────
    if intermediate_buf is not None and intermediate_buf.numel() >= num_pairs * I:
        intermediate = intermediate_buf[: num_pairs * I].view(num_pairs, I)
    else:
        intermediate = torch.empty((num_pairs, I), device=device, dtype=dtype)

    # ── Output buffer — must be zeroed; W2 uses atomic_add ───────────────────
    if output_buf is not None and output_buf.numel() >= M * H:
        final_output = output_buf[:M, :H] if output_buf.dim() == 2 else output_buf[: M * H].view(M, H)
        final_output.zero_()
    else:
        final_output = torch.zeros((M, H), device=device, dtype=dtype)

    # ── W1: gate+up projection + activation → intermediate ───────────────────
    grid_w1 = (num_pairs, triton.cdiv(I, BLOCK_N))
    _w1_int4_act_kernel[grid_w1](
        hidden_states,
        w1_qweight,
        w1_scales,
        w1_qzeros,
        intermediate,
        expert_ids,
        token_ids,
        hidden_states.stride(0),
        hidden_states.stride(1),
        w1_qweight.stride(0),
        w1_qweight.stride(1),
        w1_qweight.stride(2),
        w1_scales.stride(0),
        w1_scales.stride(1),
        w1_scales.stride(2),
        w1_qzeros.stride(0),
        w1_qzeros.stride(1),
        w1_qzeros.stride(2),
        intermediate.stride(0),
        intermediate.stride(1),
        H=H,
        I=I,
        group_size=group_size,
        KPACKED=kpacked,
        ACTIVATION=act_code,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        KPACK=8,
        NPACK=8,
        num_pairs=num_pairs,
    )

    # ── W2: down projection + weighted reduce → output ────────────────────────
    grid_w2 = (num_pairs, triton.cdiv(H, BLOCK_N))
    _w2_int4_reduce_kernel[grid_w2](
        intermediate,
        w2_qweight,
        w2_scales,
        w2_qzeros,
        final_output,
        expert_ids,
        token_ids,
        weights,
        intermediate.stride(0),
        intermediate.stride(1),
        w2_qweight.stride(0),
        w2_qweight.stride(1),
        w2_qweight.stride(2),
        w2_scales.stride(0),
        w2_scales.stride(1),
        w2_scales.stride(2),
        w2_qzeros.stride(0),
        w2_qzeros.stride(1),
        w2_qzeros.stride(2),
        final_output.stride(0),
        final_output.stride(1),
        H=H,
        I=I,
        group_size=group_size,
        KPACKED=kpacked,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        KPACK=8,
        NPACK=8,
        num_pairs=num_pairs,
    )

    return final_output


# ---------------------------------------------------------------------------
# Marlin MoE path using moe_wna16_marlin_gemm kernel
# ---------------------------------------------------------------------------


def _marlin_gemm(
    a: torch.Tensor,
    c: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    g_idx: torch.Tensor,
    perm: torch.Tensor,
    workspace: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    moe_block_size: int,
    top_k: int,
    size_m: int,
    size_n: int,
    size_k: int,
    b_q_type_id: int,
) -> None:
    """Thin wrapper around moe_wna16_marlin_gemm with fixed non-varying args."""
    moe_wna16_marlin_gemm(
        a,
        c,
        None,
        qweight,
        None,  # b_bias
        scales,
        None,  # b_act_input
        None,  # b_global_scale
        qzeros if qzeros.numel() > 0 else None,
        g_idx if g_idx.numel() > 0 else None,
        perm if perm.numel() > 0 else None,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size,
        top_k,
        False,  # mul_topk_weights – weights applied in final bmm
        b_q_type_id,
        size_m,
        size_n,
        size_k,
        True,  # is_k_full
        False,  # use_atomic_add
        True,  # use_fp32_reduce
        False,  # is_zp_float
    )


def fused_experts_marlin_impl(
    hidden_states: torch.Tensor,
    w1_qweight: torch.Tensor,
    w1_scales: torch.Tensor,
    w1_qzeros: torch.Tensor,
    w1_g_idx: torch.Tensor,
    w1_perm: torch.Tensor,
    w2_qweight: torch.Tensor,
    w2_scales: torch.Tensor,
    w2_qzeros: torch.Tensor,
    w2_g_idx: torch.Tensor,
    w2_perm: torch.Tensor,
    workspace: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    b_q_type_id: int,
    num_experts: int,
    activation: str = "silu",
    cache13: "torch.Tensor | None" = None,
    cache2: "torch.Tensor | None" = None,
    out: "torch.Tensor | None" = None,
    topk_ids_i32: "torch.Tensor | None" = None,
) -> torch.Tensor:
    """MoE forward pass using moe_wna16_marlin_gemm fused kernel.

    Parameters
    ----------
    hidden_states : (M, K)
    w1_qweight    : (E, K//16, 4*I)    Marlin-tiled gate+up weight
    w1_scales     : (E, K//gs, 2*I)
    w1_qzeros     : (E, 0)             empty for symmetric quant
    w1_g_idx      : (E, 0)             empty
    w1_perm       : (E, 0)             empty
    w2_qweight    : (E, I//16, 2*K)
    w2_scales     : (E, I//gs, K)
    w2_qzeros/g_idx/perm: (E, 0)       empty
    workspace     : (sms*4,) int32
    topk_weights  : (M, topk) float32
    topk_ids      : (M, topk) int32/int64
    b_q_type      : ScalarType
    num_experts   : int
    activation    : "silu" | "gelu" | "relu"
    cache13/cache2/out/topk_ids_i32: pre-allocated buffers (optional, grown lazily in moe.py)

    Returns
    -------
    (M, K) output tensor (= *out* when provided)
    """
    M, K = hidden_states.shape
    topk = topk_ids.shape[1]
    M_topk = M * topk
    device = hidden_states.device
    dtype = hidden_states.dtype

    # I = intermediate size from W2 shape: w2_qweight (E, I//16, 2*K)
    I = w2_qweight.size(1) * 16  # noqa: E741

    topk_weights_f32 = topk_weights if topk_weights.dtype == torch.float32 else topk_weights.float()

    # ── moe_block_size heuristic ──────────────────────────────
    tpe = M * topk / max(num_experts, 1)
    moe_block_size = next((bs for bs in [8, 16, 32, 48, 64] if tpe / bs < 0.9), 64)

    # ── Routing structures ────────────────────────────────────
    sorted_token_ids, expert_ids, num_tokens_post_padded = _moe_align_block_size(
        topk_ids.view(M, topk),
        moe_block_size,
        num_experts,
        pre_topk_ids_i32=topk_ids_i32,
    )

    # ── Intermediate buffers ──────────────────────────────────
    need13 = M_topk * max(2 * I, K)
    need2 = M_topk * I
    if cache13 is None or cache13.numel() < need13:
        cache13 = torch.empty(need13, device=device, dtype=dtype)
    if cache2 is None or cache2.numel() < need2:
        cache2 = torch.empty(need2, device=device, dtype=dtype)

    cache_w1 = _view_cache(cache13, M_topk, 2 * I)
    cache_w2 = _view_cache(cache13, M_topk, K)
    cache_act = _view_cache(cache2, M_topk, I)

    # ── W1: gate+up projection ─────────────────────────────────────────────
    _marlin_gemm(
        hidden_states,
        cache_w1,
        w1_qweight,
        w1_scales,
        w1_qzeros,
        w1_g_idx,
        w1_perm,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights_f32,
        moe_block_size,
        topk,
        M,
        2 * I,
        K,
        b_q_type_id,
    )

    # ── Gated activation ───────────────────────────────────────────────────
    act_str = activation.lower()
    if act_str == "silu":
        silu_and_mul(cache_act, cache_w1)
    elif act_str == "gelu":
        gelu_and_mul(cache_act, cache_w1)
    else:
        gate, up = cache_w1[:, :I], cache_w1[:, I:]
        torch.mul(F.relu(gate), up, out=cache_act)

    # ── W2: down projection ────────────────────────────────────────────────
    _marlin_gemm(
        cache_act,
        cache_w2,
        w2_qweight,
        w2_scales,
        w2_qzeros,
        w2_g_idx,
        w2_perm,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights_f32,
        moe_block_size,
        1,
        M_topk,
        K,
        I,
        b_q_type_id,
    )

    # ── Weighted reduction: (M*topk, K) → (M, K) ──────────────────────────
    # Routing weights are applied here via bmm rather than inside the kernel,
    # avoiding the kernel indexing topk_weights with padding sentinels (OOB).
    if out is None:
        out = torch.empty((M, K), device=device, dtype=dtype)

    torch.bmm(
        topk_weights_f32.to(dtype).unsqueeze(1),  # (M, 1, topk)
        cache_w2.reshape(M, topk, K),  # (M, topk, K)
        out=out.unsqueeze(1),  # (M, 1, K) → same storage as out
    )

    return out
