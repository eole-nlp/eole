"""Marlin MoE path using the moe_wna16_marlin_gemm kernel."""

import torch
import torch.nn.functional as F

from eole.modules.moe_quant_utils import _moe_align_block_size
from eole.ops import gelu_and_mul
from eole.ops import moe_wna16_marlin_gemm
from eole.ops import silu_and_mul


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
