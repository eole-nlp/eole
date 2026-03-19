"""
eole/modules/marlin_linear.py

Standalone MarlinQuantLinear module for GPTQ-Marlin inference.
Adapted from gptqmodel/nn_modules/qlinear/marlin.py (Apache-2.0).
Original source: https://github.com/ModelCloud/GPTQModel

This implementation has no dependency on the gptqmodel Python package.
It uses eole's own CUDA kernels (via eole._ops and eole.ops) for both
weight repacking (gptq_marlin_repack) and inference (gptq_marlin_gemm
via the dedicated dense Marlin CUDA kernel — no MoE routing overhead).
"""

from typing import Optional

import torch
import torch.nn as nn

from eole.ops import gptq_marlin_repack
from .marlin_scalar_type import scalar_types
from .marlin_utils import (
    _transform_param,
    apply_gptq_marlin_linear,
    marlin_is_k_full,
    marlin_make_empty_g_idx,
    marlin_make_workspace,
    marlin_permute_scales,
    marlin_sort_g_idx,
    replace_parameter,
)

# Supported quantization configs: (bits, sym) → ScalarType
_TYPE_MAP = {
    (4, True): scalar_types.uint4b8,
    (8, True): scalar_types.uint8b128,
}

# Marlin requires in_features divisible by 128 (tile_k_size * 8 for 4-bit)
# and out_features divisible by 64 (tile_n_size).
_MIN_IN_FEATURES_MULT = 128
_MIN_OUT_FEATURES_MULT = 64


class MarlinQuantLinear(nn.Module):
    """Quantized linear layer using the Marlin kernel.

    Weights are stored in GPTQ column-packed format initially and repacked
    into Marlin tiled format during ``post_init()``.  Inference is performed
    via the MoE Marlin GEMM kernel with trivial (single-expert) routing,
    which is already compiled into ``eole._ops``.

    Parameters
    ----------
    bits:       4 or 8
    group_size: quantization group size (-1 = per-channel)
    desc_act:   whether activation reordering (act_order) is used
    sym:        symmetric quantization (must be True for Marlin)
    in_features, out_features: linear layer dimensions
    bias:       whether to include a bias term
    """

    SUPPORTS_BITS = [4, 8]
    SUPPORTS_GROUP_SIZE = [-1, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True]

    def __init__(
        self,
        bits: int,
        group_size: int,
        desc_act: bool,
        sym: bool,
        in_features: int,
        out_features: int,
        bias: bool = False,
        **kwargs,
    ):
        super().__init__()

        if bits not in self.SUPPORTS_BITS:
            raise ValueError(f"MarlinQuantLinear: unsupported bits={bits}. Must be 4 or 8.")
        if sym not in self.SUPPORTS_SYM:
            raise ValueError("MarlinQuantLinear: only symmetric quantization (sym=True) is supported.")
        if (bits, sym) not in _TYPE_MAP:
            raise ValueError(f"Unsupported quantization config: bits={bits}, sym={sym}")

        # Validate shape requirements
        if out_features % _MIN_OUT_FEATURES_MULT != 0:
            raise NotImplementedError(
                f"MarlinQuantLinear: out_features={out_features} must be divisible by "
                f"{_MIN_OUT_FEATURES_MULT} (Marlin tile size). "
                f"For layers with non-aligned dimensions, use the PyTorch or Triton backend "
                f"by setting sym=False or force_pytorch=True."
            )

        if desc_act and group_size == -1:
            desc_act = False

        self.bits = bits
        self.group_size = group_size if group_size != -1 else in_features
        self.desc_act = desc_act
        self.in_features = in_features
        self.out_features = out_features
        self.pack_factor = 32 // bits

        self.weight_type = _TYPE_MAP[(bits, sym)]

        scales_and_zp_size = in_features // self.group_size

        # Quantized weights (GPTQ column-packed format, before post_init)
        self.register_parameter(
            "qweight",
            nn.Parameter(
                torch.empty(in_features // self.pack_factor, out_features, dtype=torch.int32),
                requires_grad=False,
            ),
        )

        # Activation-order index
        self.register_parameter(
            "g_idx",
            nn.Parameter(
                torch.empty(in_features, dtype=torch.int32),
                requires_grad=False,
            ),
        )

        # Quantization scales
        self.register_parameter(
            "scales",
            nn.Parameter(
                torch.empty(scales_and_zp_size, out_features, dtype=torch.float16),
                requires_grad=False,
            ),
        )

        # Quantized zero-points (zeroed out for symmetric quant; repacked to empty in post_init)
        self.register_parameter(
            "qzeros",
            nn.Parameter(
                torch.empty(scales_and_zp_size, out_features // self.pack_factor, dtype=torch.int32),
                requires_grad=False,
            ),
        )

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

        # Set after post_init
        self.is_k_full: bool = True
        self.workspace: Optional[torch.Tensor] = None
        self.g_idx_sort_indices: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # post_init: repack weights into Marlin tiled format
    # ------------------------------------------------------------------

    def post_init(self) -> None:
        """Repack GPTQ weights into Marlin tiled layout and allocate workspace.

        Must be called after the model weights have been loaded and the
        module has been moved to a CUDA device.
        """
        device = self.qweight.device
        if device.type != "cuda":
            raise RuntimeError("MarlinQuantLinear.post_init() requires CUDA.")

        self.is_k_full = marlin_is_k_full(self.desc_act, is_row_parallel=False)

        # Allocate workspace (sms * 4 to cover both decode and large prefill)
        self.workspace = marlin_make_workspace(device, max_blocks_per_sm=4)

        def transform_w_q(x: nn.Parameter) -> nn.Parameter:
            x.data = gptq_marlin_repack(
                x.data.contiguous(),
                perm=self.g_idx_sort_indices,
                size_k=self.in_features,
                size_n=self.out_features,
                num_bits=self.bits,
            )
            return x

        def transform_w_s(x: nn.Parameter) -> nn.Parameter:
            x.data = marlin_permute_scales(
                x.data.contiguous(),
                size_k=self.in_features,
                size_n=self.out_features,
                group_size=self.group_size,
            )
            return x

        # Sort g_idx for activation reordering
        if self.desc_act:
            g_idx, g_idx_sort_indices = marlin_sort_g_idx(self.g_idx)
            _transform_param(self, "g_idx", lambda _: g_idx)
            self.g_idx_sort_indices = g_idx_sort_indices
        else:
            replace_parameter(self, "g_idx", marlin_make_empty_g_idx(device))
            self.g_idx_sort_indices = marlin_make_empty_g_idx(device)

        # qzeros are not used for symmetric quantization
        replace_parameter(self, "qzeros", marlin_make_empty_g_idx(device))

        _transform_param(self, "qweight", transform_w_q)
        _transform_param(self, "scales", transform_w_s)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 0:
            return torch.empty((0, self.out_features), dtype=x.dtype, device=x.device)

        # Sync scales dtype with input
        if x.dtype != self.scales.dtype:
            replace_parameter(self, "scales", self.scales.to(dtype=x.dtype))

        return apply_gptq_marlin_linear(
            input=x,
            weight=self.qweight,
            weight_scale=self.scales,
            weight_zp=self.qzeros,
            g_idx=self.g_idx,
            g_idx_sort_indices=self.g_idx_sort_indices,
            workspace=self.workspace,
            wtype=self.weight_type,
            output_size_per_partition=self.out_features,
            input_size_per_partition=self.in_features,
            is_k_full=self.is_k_full,
            bias=self.bias,
            use_fp32_reduce=True,
            use_atomics=False,
        )


__all__ = ["MarlinQuantLinear"]
