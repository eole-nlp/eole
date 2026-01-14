"""RMSnorm."""

import torch
import torch.nn as nn
from eole.constants import _eole_ops

if _eole_ops:
    import eole.ops


class RMSNorm(torch.nn.Module):
    """RMSNorm: https://arxiv.org/abs/1910.07467
    Args:
        hidden_size (int): layer hidden_size dimension.
        eps: variance epsilon.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(hidden_size))

    @torch.compile(dynamic=True)
    def compute_rms(self, hidden_states):
        inp_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return hidden_states.to(inp_dtype) * self.weight

    def forward(self, hidden_states):
        if self.training or not _eole_ops:
            return self.compute_rms(hidden_states)
        out = torch.empty_like(hidden_states)
        eole.ops.rms_norm(out, hidden_states, self.weight, self.eps)
        return out

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class GemmaRMSNorm(RMSNorm):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__(hidden_size, eps)

    @torch.compile(dynamic=True)
    def compute_rms(self, hidden_states):
        dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states = hidden_states * (1.0 + self.weight.float())
        return hidden_states.to(dtype)

    def forward(self, hidden_states):
        if self.training or not _eole_ops:
            return self.compute_rms(hidden_states)
        out = torch.empty_like(hidden_states)
        eole.ops.rms_norm(out, hidden_states, self.weight, self.eps, gemma=True)
        return out

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"
