"""Opaque inference operator; importing this module does not require Triton."""

import torch


@torch.library.custom_op("eole::gdn_control_pair", mutates_args=())
def gdn_control_pair(inputs: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if (
        inputs.shape != (1, 4, 5120)
        or weight.shape != (96, 5120)
        or inputs.dtype != torch.bfloat16
        or weight.dtype != inputs.dtype
        or not inputs.is_cuda
        or inputs.device != weight.device
        or not inputs.is_contiguous()
        or not weight.is_contiguous()
    ):
        raise ValueError("GDN control kernel requires contiguous CUDA BF16 [1,4,5120] inputs and [96,5120] weights")
    from eole.triton.gdn_control import control_pair

    return control_pair(inputs, weight)


@gdn_control_pair.register_fake
def _fake_control_pair(inputs, weight):
    return inputs.new_empty((1, 4, 96))
