"""Group-32 dynamic INT8 projections with one retained weight layout.

One row-major layout serves decoding, verification and prefill. Importing this
module does not require Triton; its CUDA implementation is loaded on first use.
"""

import torch
from torch import nn


@torch.library.custom_op("eole::int8_projection", mutates_args=())
def int8_projection(inputs: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    k = inputs.shape[-1]
    if (
        not inputs.is_cuda
        or inputs.dtype not in (torch.bfloat16, torch.float16)
        or not inputs.is_contiguous()
        or k < 128
        or k % 128
        or weight.ndim != 2
        or weight.shape[1] != k // 4
        or scales.shape != (weight.shape[0], k // 32)
        or weight.dtype != torch.int32
        or scales.dtype != inputs.dtype
        or weight.device != inputs.device
        or scales.device != inputs.device
        or not weight.is_contiguous()
        or not scales.is_contiguous()
    ):
        raise ValueError("W8A8 requires contiguous CUDA FP16/BF16 inputs and packed INT8/group-32 weights")
    from eole.triton.int8_gemv import project

    return project(inputs, weight, scales)


@int8_projection.register_fake
def _fake_int8_projection(inputs, weight, scales):
    return inputs.new_empty((*inputs.shape[:-1], weight.shape[0]))


@torch.library.custom_op("eole::round_int8_inputs", mutates_args=())
def round_int8_inputs(inputs: torch.Tensor) -> torch.Tensor:
    if (
        not inputs.is_cuda
        or not inputs.is_contiguous()
        or inputs.dtype not in (torch.float16, torch.bfloat16)
        or inputs.shape[-1] < 128
        or inputs.shape[-1] % 128
    ):
        raise ValueError("W8A8 activation rounding requires contiguous CUDA FP16/BF16 inputs")
    from eole.triton.int8_gemv import round_inputs

    return round_inputs(inputs)


@round_int8_inputs.register_fake
def _fake_round_inputs(inputs):
    return torch.empty_like(inputs)


class W8A8Linear(nn.Module):
    """Packed signed INT8 weights and native-dtype group scales; no duplicate layout."""

    def __init__(self, source):
        super().__init__()
        if (
            source.bits != 8
            or source.group_size != 32
            or source.desc_act
            or source.bias is not None
            or source.workspace is not None
            or source.qweight.device.type != "cuda"
        ):
            raise ValueError("W8A8 conversion requires unrepacked symmetric INT8/group-32 projections")
        from eole.modules.marlin_utils import marlin_make_workspace

        self.register_buffer("workspace", marlin_make_workspace(source.qweight.device))
        self.register_buffer("empty_index", torch.empty(0, device=source.qweight.device, dtype=torch.int32))
        self.in_features = source.in_features
        self.out_features = source.out_features
        # GPTQ stores four unsigned (code + 128) bytes per int32. Flipping each
        # sign bit gives signed bytes for DP4A without requantizing the weights.
        self.register_buffer("packed", source.qweight.t().contiguous() ^ -2139062144)
        self.register_buffer("scales", source.scales.t().contiguous())

    def forward(self, inputs):
        if inputs.shape[-1] != self.in_features:
            raise ValueError("W8A8 input width does not match the projection")
        inputs = inputs.contiguous()
        # Larger batches amortize a temporary repack; never retain another
        # model-sized layout merely to accelerate prefill.
        if inputs.numel() // self.in_features >= 64:
            from eole.modules.autoround_linear import w8a8_prefill

            return w8a8_prefill(round_int8_inputs(inputs), self.packed, self.scales, self.workspace, self.empty_index)
        return int8_projection(inputs, self.packed, self.scales)
