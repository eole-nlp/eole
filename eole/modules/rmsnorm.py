"""RMSnorm."""

import torch
import torch.nn as nn
from eole.ops import _CPP_OPS_AVAILABLE, rms_norm


class RMSNormFunc(torch.autograd.Function):
    """Custom autograd Function implementing RMSNorm with a manual backward pass.

    This wraps the low-level ``eole.ops.rms_norm`` kernel in the forward pass and
    defines the corresponding gradient computation in :meth:`backward` so that
    RMSNorm can be used efficiently during training.
    """

    @staticmethod
    def forward(ctx, hidden_states, weight, eps, gemma=False):
        """Apply RMS normalization using the fused ``eole.ops.rms_norm`` kernel.

        Args:
            ctx: Autograd context used to stash tensors for the backward pass.
            hidden_states (torch.Tensor): Input activations of shape
                ``(..., hidden_size)`` to be normalized along the last dimension.
            weight (torch.Tensor): Learnable scaling weights of shape
                ``(hidden_size,)`` applied after normalization.
            eps (float): Small epsilon value added to the variance for numerical
                stability before taking the reciprocal square root.
            gemma (bool, optional): If ``True``, use Gemma-style RMSNorm where
                the effective weight is ``(1.0 + weight)`` instead of ``weight``.
                Defaults to ``False``.

        Returns:
            torch.Tensor: The RMS-normalized tensor with the same shape and dtype
            as ``hidden_states``.
        """
        # Save these for the manual backward pass
        ctx.save_for_backward(hidden_states, weight)
        ctx.eps = eps
        ctx.gemma = gemma

        out = torch.empty_like(hidden_states)
        rms_norm(out, hidden_states, weight, eps, gemma=gemma)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, weight = ctx.saved_tensors
        eps, gemma = ctx.eps, ctx.gemma

        # 1. Setup Dtypes and Shapes
        orig_dtype = hidden_states.dtype
        # High precision for normalization math to avoid numerical instability
        x = hidden_states.to(torch.float32)
        grad_y = grad_output.to(torch.float32)
        w = weight.to(torch.float32)

        # 2. Recompute the statistics needed for the gradient
        var = x.pow(2).mean(-1, keepdim=True)
        rstd = torch.rsqrt(var + eps)

        # Handle Gemma offset: actual_w = (1 + w) for Gemma, else just w
        actual_w = (1.0 + w) if gemma else w

        # 3. Compute Gradient w.r.t Weight (dL/dW)
        x_norm = x * rstd
        # Sum over all dimensions except the last one (hidden_size)
        grad_weight = (grad_y * x_norm).sum(dim=tuple(range(grad_y.dim() - 1)))

        # 4. Compute Gradient w.r.t Input (dL/dX)
        # Following the derivation: dx = (1/N) * rstd * (N * dy_w - x_norm * sum(dy_w * x_norm))
        dy_w = grad_y * actual_w
        sum_dyw_xnorm = (dy_w * x_norm).sum(-1, keepdim=True)

        grad_input = rstd * (dy_w - x_norm * sum_dyw_xnorm / x.shape[-1])

        return grad_input.to(orig_dtype), grad_weight.to(orig_dtype), None, None


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

    def forward(self, hidden_states):
        if _CPP_OPS_AVAILABLE and not torch.compiler.is_compiling():
            if self.training:
                return RMSNormFunc.apply(hidden_states, self.weight, self.eps, False)
            else:
                out = torch.empty_like(hidden_states)
                rms_norm(out, hidden_states, self.weight, self.eps, gemma=False)
                return out
        inp_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(inp_dtype) * self.weight

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class GemmaRMSNorm(RMSNorm):

    def forward(self, hidden_states):
        if _CPP_OPS_AVAILABLE and not torch.compiler.is_compiling():
            if self.training:
                return RMSNormFunc.apply(hidden_states, self.weight, self.eps, True)
            else:
                out = torch.empty_like(hidden_states)
                rms_norm(out, hidden_states, self.weight, self.eps, gemma=True)
                return out
        dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = x * (1.0 + self.weight.float())
        return x.to(dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"
