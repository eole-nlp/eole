"""Compile-compatible boundaries for optional flash-linear-attention kernels.

Keep each operator schema, native launch and fake implementation together.
Model-level dispatch and pure-PyTorch fallbacks live in gated_delta_net.py.
"""

import torch

try:
    from fla.modules.convolution import causal_conv1d_update as _fla_causal_conv1d_update

    @torch.library.custom_op("eole::_fla_causal_conv1d_update", mutates_args={"conv_state"})
    def _compiled_fla_causal_conv1d_update(
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        activation: str | None = None,
    ) -> torch.Tensor:
        # FLA 0.5.2 dispatch marks this entry point compiler.disable.
        # Keep it opaque to Dynamo while declaring the cache mutation,
        # so the surrounding decoder remains one compilable graph.
        out, _ = _fla_causal_conv1d_update(
            x.squeeze(-1),
            conv_state,
            residual=None,
            weight=weight,
            bias=bias,
            activation=activation,
        )
        return out.unsqueeze(-1)

    @_compiled_fla_causal_conv1d_update.register_fake
    def _fake_fla_causal_conv1d_update(x, conv_state, weight, bias=None, activation=None):
        return torch.empty_like(x, memory_format=torch.contiguous_format)

except ImportError:
    pass  # The model selects its pure-PyTorch fallback when this adapter is unavailable.

try:
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule as _fla_fused_recurrent_gated_delta_rule

    @torch.library.custom_op("eole::_fla_recurrent_gated_delta_rule", mutates_args=())
    def _compiled_fla_recurrent_gated_delta_rule(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        use_qk_l2norm_in_kernel: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Inductor lowers FLA's Python `scale` argument to FP64, promoting the
        # query/output reduction to double precision. Keep the native launch
        # opaque so it uses FP32 scalars, as in eager FLA. Normalization and
        # recurrent state accumulation remain FP32 in the original FLA kernel.
        return _fla_fused_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

    @_compiled_fla_recurrent_gated_delta_rule.register_fake
    def _fake_fla_recurrent_gated_delta_rule(q, k, v, g, beta, initial_state, use_qk_l2norm_in_kernel):
        return (
            torch.empty_like(v, memory_format=torch.contiguous_format),
            torch.empty_like(initial_state, dtype=torch.float32, memory_format=torch.contiguous_format),
        )

except ImportError:
    pass  # The model selects its pure-PyTorch fallback when this adapter is unavailable.

try:
    from fla.modules.fused_norm_gate import rms_norm_gated as _fla_rms_norm_gated

    @torch.library.custom_op("eole::_fla_rms_norm_gated", mutates_args=())
    def _compiled_fla_rms_norm_gated(
        x: torch.Tensor,
        gate: torch.Tensor,
        weight: torch.Tensor | None,
        bias: torch.Tensor | None,
        eps: float,
        activation: str,
    ) -> torch.Tensor:
        # FLA 0.5.2 dispatches to a compiler-disabled forward kernel inside
        # its autograd Function. The inference graph only needs its output.
        return _fla_rms_norm_gated(x, gate, weight, bias, activation, eps=eps)

    @_compiled_fla_rms_norm_gated.register_fake
    def _fake_fla_rms_norm_gated(x, gate, weight, bias, eps, activation):
        return torch.empty_like(x, memory_format=torch.contiguous_format)

except ImportError:
    pass  # The model selects its pure-PyTorch fallback when this adapter is unavailable.
