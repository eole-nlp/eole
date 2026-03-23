"""
moe_quant_utils.py
Utilities for detecting quantisation type and stacking per-expert int4
weights into batch tensors compatible with fused_moe_int4.py.

Supported backends
------------------
* AutoGPTQ / AutoRound  – QuantLinear   (K-packed int4, kpacked=True)
* AutoAWQ               – WQLinear_GEMM (N-packed int4, kpacked=False)
* Marlin-compatible     – MarlinQuantLinear  (Marlin tiled layout)

Tensor layout after stacking (E = num_experts, gs = group_size)
----------------------------------------------------------------
GPTQ / AutoRound (kpacked=True):
  w1_qweight  (E, K//8,  2*I)     int32
  w1_scales   (E, K//gs, 2*I)     fp16/bf16
  w1_qzeros   (E, K//gs, 2*I//8)  int32
  w2_qweight  (E, I//8,  H)       int32
  w2_scales   (E, I//gs, H)       fp16/bf16
  w2_qzeros   (E, I//gs, H//8)    int32

AWQ (kpacked=False):
  w1_qweight  (E, K,     2*I//8)  int32
  w1_scales   (E, K//gs, 2*I)     fp16/bf16
  w1_qzeros   (E, K//gs, 2*I//8)  int32
  w2_qweight  (E, I,     H//8)    int32
  w2_scales   (E, I//gs, H)       fp16/bf16
  w2_qzeros   (E, I//gs, H//8)    int32

Marlin (after MarlinQuantLinear.post_init(), for use with moe_wna16_marlin_gemm):
  w1_qweight  (E, K//16, 4*I)     int32   – gate+up concatenated along N dim
  w1_scales   (E, K//gs, 2*I)     fp16    – gate+up scales concatenated
  w2_qweight  (E, I//16, 2*K)     int32
  w2_scales   (E, I//gs, K)       fp16
  (qzeros, g_idx, perm are empty tensors for symmetric quantization)

where K = in_features (= H for W1, = I for W2)
      I = per-stream intermediate size (W1 out_features = 2*I for gated)
"""

from __future__ import annotations
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Named constants
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_SM_COUNT_FALLBACK = 160

# Marlin scalar type IDs (must match eole/csrc/quantization/marlin/eole_scalar_type.hpp and
# eole/modules/marlin_scalar_type.py).
MARLIN_UINT4B8_TYPE_ID = 4  # uint4b8: 4-bit unsigned with bias 8 (symmetric int4)
MARLIN_UINT8B128_TYPE_ID = 5  # uint8b128: 8-bit unsigned with bias 128 (symmetric int8)

# ─────────────────────────────────────────────────────────────────────────────
# Marlin workspace / block-size helpers
# ─────────────────────────────────────────────────────────────────────────────


def _marlin_make_workspace(device: torch.device, max_blocks_per_sm: int = 4) -> torch.Tensor:
    try:
        sms = torch.cuda.get_device_properties(device).multi_processor_count
    except Exception:
        sms = _DEFAULT_SM_COUNT_FALLBACK
    return torch.zeros(sms * max_blocks_per_sm, dtype=torch.int32, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# Routing structure builder
# ─────────────────────────────────────────────────────────────────────────────


def _moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    pre_topk_ids_i32: "torch.Tensor | None" = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build Marlin MoE routing structures using eole's own CUDA kernel.

    Parameters
    ----------
    topk_ids     : (M, topk) int32/int64 – expert assignments
    block_size   : Marlin moe_block_size
    num_experts  : total number of experts
    pre_topk_ids_i32 : optional pre-allocated int32 buffer used only to avoid
                       casting allocations when topk_ids arrives as int64

    Returns
    -------
    sorted_token_ids      : (num_tokens_post_padded,) int32
    expert_ids            : (num_blocks,) int32
    num_tokens_post_padded: (1,) int32
    """
    from eole.ops import moe_align_block_size as _align_fn

    flat = topk_ids.flatten()
    if flat.dtype != torch.int32:
        if pre_topk_ids_i32 is not None and pre_topk_ids_i32.numel() >= flat.numel():
            pre_topk_ids_i32.view(-1)[: flat.numel()].copy_(flat)
            flat = pre_topk_ids_i32.view(-1)[: flat.numel()]
        else:
            flat = flat.to(torch.int32)

    # Return full worst-case sized tensors — no .item() / no slicing.
    # The Marlin kernel reads ntpp[0] on the GPU; sentinel slots are ignored.
    return _align_fn(flat, num_experts, block_size)


# ─────────────────────────────────────────────────────────────────────────────
# Detection
# ─────────────────────────────────────────────────────────────────────────────


def detect_expert_quant_type(experts) -> str:
    """
    Inspect the first expert gate_up_proj and return one of:
      'gptq'   – AutoGPTQ / AutoRound  (K-packed int4)
      'awq'    – AutoAWQ               (N-packed int4)
      'marlin' – MarlinQuantLinear moe_wna16_marlin_gemm
                 (uses fused Marlin MoE kernel)
      'fp16'   – plain nn.Linear / anything else
    """
    if not experts:
        return "fp16"

    layer = experts[0].gate_up_proj
    module_path = type(layer).__module__
    classname = type(layer).__name__

    if classname == "MarlinQuantLinear":
        return "marlin"

    if classname == "QuantLinear" and hasattr(layer, "qweight"):
        in_f = (
            layer.infeatures
            if hasattr(layer, "infeatures")
            else layer.in_features if hasattr(layer, "in_features") else None
        )
        if in_f is not None:
            if layer.qweight.shape[0] == in_f:
                return "awq"
            if layer.qweight.shape[0] == in_f // 8:
                return "gptq"
            return "fp16"
        return "fp16"

    if "WQLinear" in classname and hasattr(layer, "qweight"):
        return "awq"

    if "auto_round" in module_path and hasattr(layer, "qweight"):
        return "gptq"

    return "fp16"


# ─────────────────────────────────────────────────────────────────────────────
# Low-level tensor extractors
# ─────────────────────────────────────────────────────────────────────────────


def _gptq_tensors(layer):
    """Return (qweight, scales, qzeros, group_size) from a GPTQ QuantLinear.

    qweight : (K//8, N)      int32
    scales  : (K//gs, N)     fp16
    qzeros  : (K//gs, N//8)  int32
    """
    qw, sc, qz = layer.qweight, layer.scales, layer.qzeros
    gs = (
        layer.group_size
        if (hasattr(layer, "group_size") and layer.group_size > 0)
        else (qw.shape[0] * 8) // sc.shape[0]
    )
    return qw, sc, qz, gs


def _awq_tensors(layer):
    """Return (qweight, scales, qzeros, group_size) from an AWQ WQLinear.

    qweight : (K, N//8)      int32
    scales  : (K//gs, N)     fp16
    qzeros  : (K//gs, N//8)  int32
    """
    qw, sc, qz = layer.qweight, layer.scales, layer.qzeros
    if hasattr(layer, "group_size") and layer.group_size > 0:
        gs = layer.group_size
    else:
        gs = qw.shape[0] // sc.shape[0]
    return qw, sc, qz, gs


def _cat_n(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """Concatenate along the last (output / N) dimension."""
    return torch.cat([t1, t2], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Weight freeing helpers
# ─────────────────────────────────────────────────────────────────────────────


def _free_expert_weights_single(expert):
    """Null out weight tensors for one expert without flushing the CUDA allocator."""
    for proj_name in ("gate_up_proj", "up_proj", "down_proj"):
        proj = getattr(expert, proj_name, None)
        if proj is None:
            continue
        for attr in ("qweight", "scales", "qzeros"):
            if hasattr(proj, attr):
                setattr(proj, attr, None)
        if hasattr(proj, "weight") and proj.weight is not None:
            proj.weight = None


def free_expert_weights(experts):
    """Free all experts' weight tensors then flush the CUDA allocator once."""
    for e in experts:
        _free_expert_weights_single(e)
    torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Weight stacking
# ─────────────────────────────────────────────────────────────────────────────


def stack_gptq_moe_weights(experts, device: torch.device):
    """
    Stack per-expert GPTQ / AutoRound int4 weights into batched tensors.

    Returns
    -------
    w1_qweight  (E, K//8,  2*I)     int32
    w1_scales   (E, K//gs, 2*I)     fp16
    w1_qzeros   (E, K//gs, 2*I//8)  int32
    w2_qweight  (E, I//8,  H)       int32
    w2_scales   (E, I//gs, H)       fp16
    w2_qzeros   (E, I//gs, H//8)    int32
    group_size  int
    """
    w1_qw, w1_sc, w1_qz = [], [], []
    w2_qw, w2_sc, w2_qz = [], [], []
    group_size = None

    for e in experts:
        qw_g, sc_g, qz_g, gs = _gptq_tensors(e.gate_up_proj)
        if group_size is None:
            group_size = gs

        up = getattr(e, "up_proj", None)
        if up is not None:
            qw_u, sc_u, qz_u, _ = _gptq_tensors(up)
            qw1 = _cat_n(qw_g, qw_u)
            sc1 = _cat_n(sc_g, sc_u)
            qz1 = _cat_n(qz_g, qz_u)
        else:
            qw1, sc1, qz1 = qw_g, sc_g, qz_g

        w1_qw.append(qw1.unsqueeze(0))
        w1_sc.append(sc1.unsqueeze(0))
        w1_qz.append(qz1.unsqueeze(0))

        qw_d, sc_d, qz_d, _ = _gptq_tensors(e.down_proj)
        w2_qw.append(qw_d.unsqueeze(0))
        w2_sc.append(sc_d.unsqueeze(0))
        w2_qz.append(qz_d.unsqueeze(0))

    free_expert_weights(experts)

    return (
        torch.cat(w1_qw, dim=0).to(device),
        torch.cat(w1_sc, dim=0).to(device),
        torch.cat(w1_qz, dim=0).to(device),
        torch.cat(w2_qw, dim=0).to(device),
        torch.cat(w2_sc, dim=0).to(device),
        torch.cat(w2_qz, dim=0).to(device),
        group_size,
    )


def stack_awq_moe_weights(experts, device: torch.device):
    """
    Stack per-expert AWQ int4 weights into batched tensors.

    Returns the same 7-tuple as stack_gptq_moe_weights but N-packed:
    w1_qweight  (E, K,     2*I//8)  int32
    w1_scales   (E, K//gs, 2*I)     fp16
    w1_qzeros   (E, K//gs, 2*I//8)  int32
    w2_qweight  (E, I,     H//8)    int32
    w2_scales   (E, I//gs, H)       fp16
    w2_qzeros   (E, I//gs, H//8)    int32
    group_size  int
    """
    w1_qw, w1_sc, w1_qz = [], [], []
    w2_qw, w2_sc, w2_qz = [], [], []
    group_size = None

    for e in experts:
        qw_g, sc_g, qz_g, gs = _awq_tensors(e.gate_up_proj)
        if group_size is None:
            group_size = gs

        up = getattr(e, "up_proj", None)
        if up is not None:
            qw_u, sc_u, qz_u, _ = _awq_tensors(up)
            qw1 = _cat_n(qw_g, qw_u)
            sc1 = _cat_n(sc_g, sc_u)
            qz1 = _cat_n(qz_g, qz_u)
        else:
            qw1, sc1, qz1 = qw_g, sc_g, qz_g

        w1_qw.append(qw1.unsqueeze(0))
        w1_sc.append(sc1.unsqueeze(0))
        w1_qz.append(qz1.unsqueeze(0))

        qw_d, sc_d, qz_d, _ = _awq_tensors(e.down_proj)
        w2_qw.append(qw_d.unsqueeze(0))
        w2_sc.append(sc_d.unsqueeze(0))
        w2_qz.append(qz_d.unsqueeze(0))

    free_expert_weights(experts)

    return (
        torch.cat(w1_qw, dim=0).to(device),
        torch.cat(w1_sc, dim=0).to(device),
        torch.cat(w1_qz, dim=0).to(device),
        torch.cat(w2_qw, dim=0).to(device),
        torch.cat(w2_sc, dim=0).to(device),
        torch.cat(w2_qz, dim=0).to(device),
        group_size,
    )


def stack_fp16_moe_weights(experts, device: torch.device):
    """Stack plain fp16/bf16 expert weights for vectorized MoE forward."""
    w1, w2 = [], []

    for e in experts:
        w1.append(e.gate_up_proj.weight.data.unsqueeze(0))
        w2.append(e.down_proj.weight.data.unsqueeze(0))

    free_expert_weights(experts)

    return (
        torch.cat(w1, dim=0).to(device),
        torch.cat(w2, dim=0).to(device),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Marlin MoE weight stacking
# ─────────────────────────────────────────────────────────────────────────────


def stack_marlin_moe_weights(experts, device: torch.device):
    """Stack per-expert MarlinQuantLinear weights for moe_wna16_marlin_gemm.

    After ``post_init()`` each ``MarlinQuantLinear`` stores weights in the
    Marlin tile format (``gptq_marlin_repack`` applied).  For int4 symmetric
    quantization the layout per expert is::

        gate_up_proj.qweight  (K//16, 2*I)  int32   – Marlin-tiled K→I weight
        up_proj.qweight       (K//16, 2*I)  int32   – Marlin-tiled K→I weight
        down_proj.qweight     (I//16, 2*K)  int32   – Marlin-tiled I→K weight

    The gate and up projections are concatenated along their last (N) dimension
    before stacking, producing the combined W1 weight expected by
    ``moe_wna16_marlin_gemm`` with ``size_n = 2*I``.  This concatenation is
    valid because Marlin tiling is column-independent: repacking (K, I)
    separately and concatenating along N gives the same result as repacking
    the combined (K, 2I) weight matrix in one shot.

    Returns
    -------
    w1_qweight  (E, K//16, 4*I)   int32   – combined gate+up Marlin weight
    w1_scales   (E, K//gs, 2*I)   fp16    – combined gate+up scales
    w1_qzeros   (E, 0)            int32   – empty (symmetric quantization)
    w1_g_idx    (E, 0)            int32   – empty (no activation reordering)
    w1_perm     (E, 0)            int32   – empty
    w2_qweight  (E, I//16, 2*K)   int32
    w2_scales   (E, I//gs, K)     fp16
    w2_qzeros   (E, 0)            int32   – empty
    w2_g_idx    (E, 0)            int32   – empty
    w2_perm     (E, 0)            int32   – empty
    workspace   (sms * 4,)        int32   – shared Marlin workspace
    num_bits    int               – quantization bit-width (4 or 8)
    scalar_type_id  int           – Marlin scalar type ID (e.g. uint4b8.id)
    group_size  int               – quantization group size
    """
    w1_qw, w1_sc = [], []
    w2_qw, w2_sc = [], []
    num_bits = None
    scalar_type_id = None
    group_size = None

    for e in experts:
        gate_layer = e.gate_up_proj
        up_layer = getattr(e, "up_proj", None)
        down_layer = e.down_proj

        if num_bits is None:
            num_bits = getattr(gate_layer, "bits", 4)
            group_size = getattr(gate_layer, "group_size", 128)
            # Retrieve the Marlin scalar type ID (e.g. uint4b8.id)
            scalar_type_id = MARLIN_UINT4B8_TYPE_ID if num_bits == 4 else MARLIN_UINT8B128_TYPE_ID

        if up_layer is not None:
            qw1 = torch.cat([gate_layer.qweight, up_layer.qweight], dim=-1)
            sc1 = torch.cat([gate_layer.scales, up_layer.scales], dim=-1)
        else:
            # gate_up_proj already contains the fused gate+up weight
            qw1 = gate_layer.qweight
            sc1 = gate_layer.scales

        w1_qw.append(qw1.unsqueeze(0))
        w1_sc.append(sc1.unsqueeze(0))
        w2_qw.append(down_layer.qweight.unsqueeze(0))
        w2_sc.append(down_layer.scales.unsqueeze(0))

    free_expert_weights(experts)

    # Empty tensors for symmetric (no zero-points, no g_idx / perm)
    empty = torch.empty(0, dtype=torch.int32, device=device)

    return (
        torch.cat(w1_qw, dim=0).to(device),
        torch.cat(w1_sc, dim=0).to(device),
        empty,  # w1_qzeros
        empty,  # w1_g_idx
        empty,  # w1_perm
        torch.cat(w2_qw, dim=0).to(device),
        torch.cat(w2_sc, dim=0).to(device),
        empty,  # w2_qzeros
        empty,  # w2_g_idx
        empty,  # w2_perm
        _marlin_make_workspace(device),
        num_bits,
        scalar_type_id,
        group_size,
    )
