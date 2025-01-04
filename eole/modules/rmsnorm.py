"""RMSnorm."""

import torch
import torch.nn as nn


try:
    import awq_ext

    AWQ_EXT = True
except ImportError:
    AWQ_EXT = False


class RMSNorm(torch.nn.Module):
    """RMSNorm: https://arxiv.org/abs/1910.07467
    Args:
        hidden_size (int): layer hidden_size dimension.
        eps: variance epsilon.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile(dynamic=True)
    def compute_rms(self, hidden_states, dtype):
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states = hidden_states.to(dtype)
        return hidden_states * self.weight

    def forward(self, hidden_states):
        inp_dtype = hidden_states.dtype
        if AWQ_EXT and not self.training:
            # cuda kernel support only fp16 - need to cast
            output = torch.empty_like(hidden_states).to(torch.float16)
            if hidden_states.dim() == 2:  # patch for multi experts
                hidden_states = hidden_states.unsqueeze(0)
            awq_ext.layernorm_forward_cuda(hidden_states.half(), self.weight.half(), output, self.eps)
            if hidden_states.dim() == 2:  # patch for multi experts
                output = output.unsqueeze(0)
            return output.to(inp_dtype)
        else:
            return self.compute_rms(hidden_states, inp_dtype)
