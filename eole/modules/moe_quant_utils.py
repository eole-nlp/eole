"""
moe_quant_utils.py
Utilities for detecting quantisation type and stacking per-expert int4
weights into batch tensors compatible with fused_moe_int4.py.

Supported backends
------------------
* AutoGPTQ / AutoRound  – QuantLinear   (K-packed int4, kpacked=True)
* AutoAWQ               – WQLinear_GEMM (N-packed int4, kpacked=False)

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

where K = in_features (= H for W1, = I for W2)
      I = per-stream intermediate size (W1 out_features = 2*I)
"""

from __future__ import annotations
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Detection
# ─────────────────────────────────────────────────────────────────────────────


def detect_expert_quant_type(experts) -> str:
    """
    Inspect the first expert gate_up_proj and return one of:
      'gptq'  – AutoGPTQ / AutoRound  (K-packed int4)
      'awq'   – AutoAWQ               (N-packed int4)
      'fp16'  – plain nn.Linear / anything else (including Marlin)

    Note on gptqmodel Marlin layers
    --------------------------------
    gptqmodel's Marlin backend repacks the standard GPTQ qweight (shape
    ``(in_features//8, out_features)``) into its own tiled layout during
    ``post_init()``.  After repacking the shape no longer matches either
    the GPTQ (K-packed) or AWQ (N-packed) conventions.  We detect this case
    via two guards and fall through to 'fp16' so that vectorized_moe is used
    instead, which calls each expert's own ``forward()`` (i.e. the fast
    Marlin CUDA kernel) rather than our custom Triton kernel.
    """
    if not experts:
        return "fp16"

    layer = experts[0].gate_up_proj
    module_path = type(layer).__module__
    classname = type(layer).__name__

    # gptqmodel Marlin layers live under the "gptqmodel" namespace.
    # Their qweight has been repacked into Marlin tiles and is incompatible
    # with our int4 Triton kernel.  Let each expert's own forward() handle it.
    if "gptqmodel" in module_path:
        return "fp16"

    # AutoGPTQ / AutoRound: QuantLinear with qweight.
    # Distinguish from AWQ's QuantLinear by packing axis:
    #   GPTQ : qweight.shape[0] == in_features // 8  (K-packed)
    #   AWQ  : qweight.shape[0] == in_features        (N-packed)
    # Any other shape (e.g. Marlin-repacked qweight from a wrapper that does
    # not live under "gptqmodel") is treated as an unrecognised format and
    # falls through to 'fp16'.
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
            # Unrecognised packing (e.g. Marlin-repacked qweight exposed via a
            # non-gptqmodel wrapper) – fall through to per-expert forward().
            return "fp16"
        # Cannot determine in_features; cannot safely validate the weight layout,
        # so fall back to per-expert forward() rather than risk a kernel crash.
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

    return (
        torch.cat(w1_qw, dim=0).to(device),
        torch.cat(w1_sc, dim=0).to(device),
        torch.cat(w1_qz, dim=0).to(device),
        torch.cat(w2_qw, dim=0).to(device),
        torch.cat(w2_sc, dim=0).to(device),
        torch.cat(w2_qz, dim=0).to(device),
        group_size,
    )


def _free_expert_weights(experts):
    """
    Delete per-expert weight tensors after they have been stacked into the
    fast-path cache. This recovers the VRAM that would otherwise be held
    by the original nn.ModuleList parameters.

    Note: this makes the experts unusable for forward() calls, which is
    intentional — inference always goes through the Triton kernel after this.
    """
    for e in experts:
        for proj_name in ("gate_up_proj", "up_proj", "down_proj"):
            proj = getattr(e, proj_name, None)
            if proj is None:
                continue
            # int4 quantised layers: free packed weight buffers
            for attr in ("qweight", "scales", "qzeros"):
                if hasattr(proj, attr):
                    setattr(proj, attr, None)
            # fp16 plain linear
            if hasattr(proj, "weight") and proj.weight is not None:
                proj.weight = None
    torch.cuda.empty_cache()
