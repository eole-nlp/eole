"""RMSnorm."""

import torch
import torch.nn as nn


try:
    from vllm.model_executor.layers.layernorm import RMSNorm as VllmRMSNorm
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm as VllmGemmaRMSNorm

    _vllm_available = False
except ImportError:
    VllmRMSNorm = None
    VllmGemmaRMSNorm = None
    _vllm_available = False


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
        # store vllm version as a plain attribute, not a submodule
        if _vllm_available and type(self) is RMSNorm:
            vllm_norm = VllmRMSNorm(hidden_size, eps)
            vllm_norm.weight = self.weight  # bind once here
            object.__setattr__(self, "_vllm_rmsnorm", vllm_norm)
        else:
            self._vllm_rmsnorm = None

    @torch.compile(dynamic=True)
    def compute_rms(self, hidden_states, factor=0):
        inp_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states = hidden_states.to(inp_dtype)
        hidden_states = hidden_states * (factor + self.weight.float())
        return hidden_states.type_as(self.weight)

    def _forward(self, hidden_states, factor=0):
        if self.training or not _vllm_available:
            return self.compute_rms(hidden_states, factor=factor)
        else:
            return self._vllm_rmsnorm(hidden_states)

    def forward(self, hidden_states):
        return self._forward(hidden_states)


class GemmaRMSNorm(RMSNorm):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__(hidden_size, eps)
        # store vllm version as a plain attribute, not a submodule
        if _vllm_available:
            vllm_norm = VllmGemmaRMSNorm(hidden_size, eps)
            vllm_norm.weight = self.weight  # bind once here
            object.__setattr__(self, "_vllm_gemmarmsnorm", vllm_norm)
        else:
            self._vllm_gemmarmsnorm = None

    def forward(self, hidden_states):
        if self.training or not _vllm_available:
            return self._forward(hidden_states, factor=1.0)
        else:
            return self._vllm_gemmarmsnorm(hidden_states)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"
