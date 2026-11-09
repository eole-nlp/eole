"""Paired A/B projections, packed B then A; importing does not require Triton."""

import torch


@torch.library.custom_op("eole::gdn_ab_projections", mutates_args=())
def gdn_ab_projections(inputs: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if (
        weight.shape not in ((96, 5120), (64, 2048))
        or inputs.shape not in ((1, 1, weight.shape[1]), (1, 4, weight.shape[1]))
        or inputs.dtype != torch.bfloat16
        or weight.dtype != inputs.dtype
        or not inputs.is_cuda
        or inputs.device != weight.device
        or not inputs.is_contiguous()
        or not weight.is_contiguous()
    ):
        raise ValueError(
            "GDN A/B projection kernel requires contiguous CUDA BF16 [1,1|4,K] inputs "
            "and [96,5120] or [64,2048] weights"
        )
    from eole.triton.gdn_ab import project_ab

    return project_ab(inputs, weight)


@gdn_ab_projections.register_fake
def _fake_ab_projections(inputs, weight):
    return inputs.new_empty((*inputs.shape[:2], weight.shape[0]))
