"""Gated Delta Network (GatedDeltaNet) – linear attention layer used in Qwen3.5.

Reference:
  "Delta Net: https://arxiv.org/abs/2406.06484"
  HuggingFace implementation:
    transformers/models/qwen3_5/modeling_qwen3_5.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import skip_init


# ---------------------------------------------------------------------------
# Optional fast-path libraries (causal_conv1d and flash-linear-attention).
# Both are optional; if not installed we fall back to pure-PyTorch kernels.
# ---------------------------------------------------------------------------
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    # Fall back to fla.modules.convolution when the causal_conv1d package is absent
    # but fla-core is installed.  The FLA API differs in two ways:
    #   1. causal_conv1d (FLA) expects x in [B, L, D] (sequence-first) and returns
    #      (output, final_state); causal_conv1d_fn expects [B, D, L] and returns
    #      just the output.
    #   2. causal_conv1d_update (FLA) has an extra `residual` positional arg before
    #      `weight`, and accepts x as [B, D] (2-D, no time dimension); the
    #      causal_conv1d package variant takes (x, conv_state, weight, bias, act)
    #      with x shaped [B, D, 1] and returns just the output.
    # The wrappers below absorb those differences so the rest of the code is unchanged.
    try:
        from fla.modules.convolution import causal_conv1d as _fla_causal_conv1d
        from fla.modules.convolution import causal_conv1d_update as _fla_causal_conv1d_update

        def causal_conv1d_fn(x, weight, bias=None, activation=None):
            # x: [B, D, L] channel-first → FLA expects [B, L, D] sequence-first
            out, _ = _fla_causal_conv1d(x.transpose(1, 2), weight=weight, bias=bias, activation=activation)
            return out.transpose(1, 2)  # back to [B, D, L]

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

        def causal_conv1d_update(x, conv_state, weight, bias=None, activation=None):
            if torch.compiler.is_compiling():
                return _compiled_fla_causal_conv1d_update(x, conv_state, weight, bias, activation)
            # x: [B, D, 1] channel-first → FLA expects [B, D] (2-D)
            # FLA signature: (x, cache, residual=None, weight=None, bias=None, activation=None)
            out, _ = _fla_causal_conv1d_update(
                x.squeeze(-1),
                conv_state,
                residual=None,
                weight=weight,
                bias=bias,
                activation=activation,
            )
            return out.unsqueeze(-1)  # back to [B, D, 1]

    except ImportError:
        causal_conv1d_fn = None
        causal_conv1d_update = None

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule as _fla_fused_recurrent_gated_delta_rule

    # FLA's fused_recurrent_gated_delta_rule has signature (q, k, v, g, gk, gv, beta, ...).
    # The extra gk/gv args sit between g and beta, so a positional call with (q,k,v,g,beta)
    # would silently pass beta to gk and leave the real beta=None — producing NaN during decode.
    # Wrap with keyword args to match eole's (q, k, v, g, beta, ...) convention.
    def fused_recurrent_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
    ):
        return _fla_fused_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

except ImportError:
    chunk_gated_delta_rule = None
    fused_recurrent_gated_delta_rule = None

# ---------------------------------------------------------------------------
# Pure-PyTorch fallback kernels
# ---------------------------------------------------------------------------


def _torch_causal_conv1d_update(hidden_states, conv_state, weight, bias=None, activation=None):
    """Single-step causal conv1d update (used in auto-regressive decoding)."""
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    if activation == "silu":
        out = F.silu(out[:, :, -seq_len:])
    else:
        out = out[:, :, -seq_len:]
    return out.to(hidden_states.dtype)


def _torch_speculative_conv1d(conv_extended, weight, bias=None):
    """Short causal stencil, suitable for fusion by the verifier compiler.

    The first window consists only of cached history, so skip it. Accumulate
    in float32 like the low-precision convolution, then round before SiLU.
    This avoids dispatching a general grouped convolution for a few tokens.
    """
    kernel = weight.size(-1)
    windows = conv_extended.unfold(-1, kernel, 1)[:, :, 1:, :]
    output = (windows.float() * weight[:, 0, :].float()[None, :, None, :]).sum(dim=-1)
    if bias is not None:
        output = output + bias.float()[None, :, None]
    return F.silu(output.to(conv_extended.dtype))


def _l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1)
        key = _l2norm(key, dim=-1)

    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1.0 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)
    for i in range(total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_i = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn_i @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def _torch_recurrent_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    initial_state,
    output_final_state,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1)
        key = _l2norm(key, dim=-1)

    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1.0 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(value)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


# ---------------------------------------------------------------------------
# RMSNorm variants used inside GatedDeltaNet
# ---------------------------------------------------------------------------

# Prefer the FLA fused kernel (faster on CUDA) when available; fall back to a
# pure-PyTorch implementation that is behaviourally identical.

try:
    from fla.modules.fused_norm_gate import FusedRMSNormGated as _FLARMSNormGated
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

    class RMSNormGated(_FLARMSNormGated):
        """FLA norm with a fullgraph-compatible inference boundary."""

        _fla_backend = True

        def forward(self, x, g, residual=None, prenorm=False, residual_in_fp32=False):
            if torch.compiler.is_compiling() and not torch.is_grad_enabled() and residual is None and not prenorm:
                return _compiled_fla_rms_norm_gated(x, g, self.weight, self.bias, self.eps, self.activation)
            return super().forward(x, g, residual=residual, prenorm=prenorm, residual_in_fp32=residual_in_fp32)

    _fla_rmsnorm_gated = True
except ImportError:
    _fla_rmsnorm_gated = False

    class RMSNormGated(nn.Module):
        """Pure-PyTorch RMSNorm with a silu gate.

        prenorm=False  → norm(x) * silu(z)   (default for GatedDeltaNet)
        prenorm=True → norm(x * silu(z))
        """

        def __init__(self, hidden_size, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x, z=None, prenorm=False):
            input_dtype = x.dtype
            if z is not None and prenorm:
                x = x * F.silu(z.to(torch.float32)).to(input_dtype)
                z = None  # already applied; skip post-norm multiply
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.variance_epsilon)
            x = self.weight * x.to(input_dtype)
            if z is not None and not prenorm:
                x = x * F.silu(z.to(torch.float32)).to(input_dtype)
            return x


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------


class GatedDeltaNet(nn.Module):
    """Gated Delta Network – linear recurrent attention layer.

    Used as the "linear_attention" layer type in hybrid models such as Qwen3.5.

    Weight naming follows the HuggingFace convention exactly so that the HF
    converter can map weights without renaming:

        in_proj_qkv  – joint QKV projection (key_dim*2 + value_dim outputs)
        in_proj_z    – gate projection for the output norm  (value_dim outputs)
        in_proj_b    – beta projection                       (num_v_heads outputs)
        in_proj_a    – alpha projection for A_log decay      (num_v_heads outputs)
        conv1d       – causal depthwise conv over QKV stream
        dt_bias      – bias for dt (time-step)               (num_v_heads,)
        A_log        – log of the decay factor               (num_v_heads,)
        norm         – RMSNormGated applied to value output
        out_proj     – final linear projection               (value_dim → hidden_size)
    """

    def __init__(self, decoder_config, layer_idx: int):
        super().__init__()
        hidden_size = decoder_config.hidden_size
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx

        self.num_v_heads = decoder_config.linear_num_value_heads
        self.num_k_heads = decoder_config.linear_num_key_heads
        self.head_k_dim = decoder_config.linear_key_head_dim
        self.head_v_dim = decoder_config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = decoder_config.linear_conv_kernel_dim

        # The conv1d operates on the concatenated QKV stream
        self.conv_dim = self.key_dim * 2 + self.value_dim

        self.in_proj_qkv = skip_init(
            nn.Linear,
            in_features=hidden_size,
            out_features=self.key_dim * 2 + self.value_dim,
            bias=False,
        )
        self.in_proj_z = skip_init(
            nn.Linear,
            in_features=hidden_size,
            out_features=self.value_dim,
            bias=False,
        )
        self.in_proj_b = skip_init(
            nn.Linear,
            in_features=hidden_size,
            out_features=self.num_v_heads,
            bias=False,
        )
        self.in_proj_a = skip_init(
            nn.Linear,
            in_features=hidden_size,
            out_features=self.num_v_heads,
            bias=False,
        )
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
            bias=False,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.norm = RMSNormGated(self.head_v_dim, eps=decoder_config.norm_eps)
        self.out_proj = skip_init(
            nn.Linear,
            in_features=self.value_dim,
            out_features=hidden_size,
            bias=False,
        )

        # Choose fast/fallback kernels
        self._causal_conv1d_fn = causal_conv1d_fn
        self._causal_conv1d_update = causal_conv1d_update or _torch_causal_conv1d_update
        self._chunk_gated_delta_rule = chunk_gated_delta_rule or _torch_chunk_gated_delta_rule
        self._recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule or _torch_recurrent_gated_delta_rule

        # Inference state (set by TransformerDecoder._init_cache / _disable_cache)
        self.conv_state = None
        self.recurrent_state = None
        # Transaction used by speculative verification. The persistent state is
        # left untouched until the accepted input prefix is known.
        self._speculating = False
        self._spec_buffers = None

    def begin_speculation(self, seq_len):
        if self.conv_state is None or self.recurrent_state is None:
            raise RuntimeError("GatedDeltaNet speculation requires initialized decode state")
        batch_size = self.recurrent_state.size(0)
        device = self.recurrent_state.device
        state_dtype = self.recurrent_state.dtype
        if (
            self._spec_buffers is None
            or self._spec_buffers["query"].shape[:2] != (batch_size, seq_len)
            or self._spec_buffers["query"].dtype != state_dtype
            or self._spec_buffers["query"].device != device
            or self._spec_buffers["conv_extended"].dtype != self.conv_state.dtype
            or self._spec_buffers["conv_extended"].device != self.conv_state.device
        ):
            self._spec_buffers = {
                "query": torch.empty(
                    batch_size, seq_len, self.num_v_heads, self.head_k_dim, dtype=state_dtype, device=device
                ),
                "key": torch.empty(
                    batch_size, seq_len, self.num_v_heads, self.head_k_dim, dtype=state_dtype, device=device
                ),
                "value": torch.empty(
                    batch_size, seq_len, self.num_v_heads, self.head_v_dim, dtype=state_dtype, device=device
                ),
                "g": torch.empty(batch_size, seq_len, self.num_v_heads, dtype=torch.float32, device=device),
                "beta": torch.empty(batch_size, seq_len, self.num_v_heads, dtype=state_dtype, device=device),
                "conv_extended": torch.empty(
                    batch_size,
                    self.conv_dim,
                    self.conv_kernel_size + seq_len,
                    dtype=self.conv_state.dtype,
                    device=self.conv_state.device,
                ),
                "final_recurrent_state": torch.empty_like(self.recurrent_state),
            }
        self._speculating = True

    def end_speculation(self):
        self._speculating = False

    def discard_speculation(self):
        self._speculating = False

    def commit_speculation(self, n_kept):
        """Commit the first n inputs consumed by the last speculative pass."""
        spec = self._spec_buffers
        if spec is None:
            raise RuntimeError("GatedDeltaNet has no speculative buffers to commit")
        seq_len = spec["query"].size(1)
        if not 1 <= n_kept <= seq_len:
            raise ValueError(f"n_kept must be in [1, {seq_len}], got {n_kept}")

        if n_kept == seq_len:
            recurrent_state = spec["final_recurrent_state"]
        else:
            # Re-run only this linear-attention layer's short accepted prefix
            # through the fused recurrent kernel. This avoids another full
            # decoder pass and avoids materializing a recurrent-state snapshot
            # for every speculative token.
            _, recurrent_state = self._recurrent_gated_delta_rule(
                spec["query"][:, :n_kept],
                spec["key"][:, :n_kept],
                spec["value"][:, :n_kept],
                spec["g"][:, :n_kept],
                spec["beta"][:, :n_kept],
                initial_state=self.recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )

        kernel = self.conv_kernel_size
        self.conv_state.copy_(spec["conv_extended"][:, :, n_kept : n_kept + kernel])
        self.recurrent_state.copy_(recurrent_state.to(self.recurrent_state.dtype))

    @staticmethod
    @torch.no_grad()
    def commit_speculation_group(layers, n_kept):
        """Commit a verified prefix for several GDN layers with one replay.

        Partial acceptance needs the recurrent state at an intermediate token
        boundary. The FLA recurrent operator returns only the final state, so
        replay that short accepted prefix. Running one operator per layer
        caused a long chain of tiny launches for hybrid models; concatenate
        same-shaped layer states along the batch dimension and replay them in
        one call instead.
        """
        if not layers:
            return

        specs = [layer._spec_buffers for layer in layers]
        seq_len = specs[0]["query"].size(1)
        if not 1 <= n_kept <= seq_len:
            raise ValueError(f"n_kept must be in [1, {seq_len}], got {n_kept}")

        # Fall back for uncommon heterogeneous GDN layouts. Qwen3.5 layers
        # share shapes, so their replay can be batched as independent rows.
        query_shape = specs[0]["query"].shape[1:]
        can_batch = all(
            all(
                spec[name].shape == specs[0][name].shape
                and spec[name].dtype == specs[0][name].dtype
                and spec[name].device == specs[0][name].device
                for name in ("query", "key", "value", "g", "beta")
            )
            and spec["query"].shape[1:] == query_shape
            and layer.recurrent_state.shape == layers[0].recurrent_state.shape
            and layer.recurrent_state.device == layers[0].recurrent_state.device
            and layer.recurrent_state.dtype == layers[0].recurrent_state.dtype
            for layer, spec in zip(layers, specs)
        )
        if not can_batch:
            for layer in layers:
                layer.commit_speculation(n_kept)
            return

        if n_kept == seq_len:
            # The verifier already computed these states in the persistent
            # dtype. Copy them directly instead of packing a large temporary
            # only to split it into the original per-layer shapes again.
            torch._foreach_copy_(
                [layer.conv_state for layer in layers],
                [
                    spec["conv_extended"][:, :, n_kept : n_kept + layer.conv_kernel_size]
                    for layer, spec in zip(layers, specs)
                ],
            )
            torch._foreach_copy_(
                [layer.recurrent_state for layer in layers],
                [spec["final_recurrent_state"] for spec in specs],
            )
            return
        else:
            # Reuse packing storage across cycles. Accepted prefix lengths
            # vary, but the allocated verifier span stays fixed; a compact
            # view provides the current replay shape without new
            # large allocations on every rejection.
            owner = layers[0]
            signature = tuple(
                (name, tuple(specs[0][name].shape), specs[0][name].dtype, specs[0][name].device)
                for name in ("query", "key", "value", "g", "beta")
            )
            signature += (
                (
                    len(layers),
                    tuple(owner.recurrent_state.shape),
                    owner.recurrent_state.dtype,
                    owner.recurrent_state.device,
                ),
            )
            if getattr(owner, "_spec_replay_signature", None) != signature:
                owner._spec_replay_buffers = {
                    name: torch.empty(
                        (len(layers) * tensor.size(0), *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device
                    )
                    for name, tensor in specs[0].items()
                    if name in ("query", "key", "value", "g", "beta")
                }
                owner._spec_replay_buffers["state"] = torch.empty(
                    (len(layers) * owner.recurrent_state.size(0), *owner.recurrent_state.shape[1:]),
                    dtype=owner.recurrent_state.dtype,
                    device=owner.recurrent_state.device,
                )
                owner._spec_replay_signature = signature
            packed = owner._spec_replay_buffers
            replay_inputs = {}
            for name in ("query", "key", "value", "g", "beta"):
                buffer = packed[name]
                rows = buffer.size(0)
                # Preserve contiguous inputs for fused kernels that flatten
                # batch and sequence without accepting arbitrary strides.
                compact = buffer.view(-1, *buffer.shape[2:])[: rows * n_kept].view(rows, n_kept, *buffer.shape[2:])
                torch.cat([spec[name][:, :n_kept] for spec in specs], dim=0, out=compact)
                replay_inputs[name] = compact
            torch.cat([layer.recurrent_state for layer in layers], dim=0, out=packed["state"])
            query, key, value, g, beta = (replay_inputs[name] for name in ("query", "key", "value", "g", "beta"))
            initial_state = packed["state"]
            _, recurrent_states = layers[0]._recurrent_gated_delta_rule(
                query,
                key,
                value,
                g,
                beta,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            if recurrent_states is None:
                raise RuntimeError("GatedDeltaNet speculative replay did not return its recurrent state")

        batch_size = layers[0].recurrent_state.size(0)
        conv_sources = []
        recurrent_sources = []
        for index, (layer, spec) in enumerate(zip(layers, specs)):
            kernel = layer.conv_kernel_size
            conv_sources.append(spec["conv_extended"][:, :, n_kept : n_kept + kernel])
            start = index * batch_size
            recurrent_sources.append(recurrent_states[start : start + batch_size])

        # One cast plus foreach copies avoids a separate conversion/copy launch
        # for every recurrent layer after the replay.
        recurrent_dtype = layers[0].recurrent_state.dtype
        if recurrent_states.dtype != recurrent_dtype:
            recurrent_states = recurrent_states.to(recurrent_dtype)
            recurrent_sources = [recurrent_states[i * batch_size : (i + 1) * batch_size] for i in range(len(layers))]
        torch._foreach_copy_([layer.conv_state for layer in layers], conv_sources)
        torch._foreach_copy_([layer.recurrent_state for layer in layers], recurrent_sources)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, hidden_states, attn_mask=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape

        # Mask padding to zero so padding tokens don't corrupt conv/recurrent state
        if attn_mask is not None and attn_mask.shape[1] > 1:
            hidden_states = (hidden_states * attn_mask[:, :, None]).to(hidden_states.dtype)

        use_precomputed = self.conv_state is not None and self.recurrent_state is not None and seq_len == 1
        use_speculative = self._speculating and seq_len > 1

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)  # (B, conv_dim, S)
        z = self.in_proj_z(hidden_states)  # (B, S, value_dim)
        z = z.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(hidden_states)  # (B, S, num_v_heads)
        a = self.in_proj_a(hidden_states)  # (B, S, num_v_heads)

        if use_precomputed:
            mixed_qkv = self._causal_conv1d_update(
                mixed_qkv,
                self.conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                "silu",
            )
        elif use_speculative:
            # Extend the cached raw QKV history and compute exactly the same
            # causal-convolution outputs as sequential decode. The extra first
            # valid convolution output belongs to the cache-only history.
            conv_extended = torch.cat([self.conv_state, mixed_qkv.to(self.conv_state.dtype)], dim=-1)
            if torch.compiler.is_compiling():
                mixed_qkv = _torch_speculative_conv1d(conv_extended, self.conv1d.weight, self.conv1d.bias).to(
                    hidden_states.dtype
                )
            else:
                conv_out = F.conv1d(
                    conv_extended,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    groups=self.conv_dim,
                )
                mixed_qkv = F.silu(conv_out[:, :, 1:]).to(hidden_states.dtype)
        else:
            if self.conv_state is not None:
                # save state for next step
                self.conv_state.copy_(
                    F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))[:, :, -self.conv_kernel_size :]
                )
            if self._causal_conv1d_fn is not None:
                mixed_qkv = self._causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation="silu",
                )
            else:
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)  # (B, S, conv_dim)

        # Split into Q, K, V streams
        query = mixed_qkv[:, :, : self.key_dim]  # (B, S, key_dim)
        key = mixed_qkv[:, :, self.key_dim : self.key_dim * 2]  # (B, S, key_dim)
        value = mixed_qkv[:, :, self.key_dim * 2 :]  # (B, S, value_dim)

        query = query.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        # Expand K/Q heads to match V heads (GQA-style for GDA layers, mirrors HF implementation)
        if self.num_v_heads // self.num_k_heads > 1:
            n_rep = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(n_rep, dim=2)
            key = key.repeat_interleave(n_rep, dim=2)

        # Discretise decay / gate
        # dt = F.softplus(a + self.dt_bias)  # (B, S, num_v_heads)
        # A = -torch.exp(self.A_log.float())  # (num_v_heads,)
        # g = dt * A.view(1, 1, -1)  # (B, S, num_v_heads)

        # If the model is loaded in fp16, without the .float() here, A might be -inf
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        beta = torch.sigmoid(b)  # (B, S, num_v_heads)

        # At padding positions, q/k/v are already zeroed (hidden_states was zeroed above).
        # But g and beta are still non-trivial: g = softplus(dt_bias)*A (decay < 1) and
        # beta = sigmoid(0) = 0.5.  This would decay the recurrent state at padding positions
        # even though no content is written, corrupting the state for shorter sequences when
        # batch_size > 1.  Fix: force g=0 (exp(0)=1, identity decay) and beta=0 (no write)
        # at padding positions so they are true no-ops for the recurrent state.
        if attn_mask is not None and attn_mask.shape[1] > 1:
            pad_mask = ~attn_mask  # (B, S), True = padding position
            g = g.masked_fill(pad_mask.unsqueeze(-1), 0.0)
            beta = beta.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        # Compute linear attention
        if use_precomputed:
            output, new_recurrent_state = self._recurrent_gated_delta_rule(
                query,
                key,
                value,
                g,
                beta,
                initial_state=self.recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            if new_recurrent_state is not None:
                self.recurrent_state.copy_(new_recurrent_state.to(self.recurrent_state.dtype))
        elif use_speculative:
            output, new_recurrent_state = self._recurrent_gated_delta_rule(
                query,
                key,
                value,
                g,
                beta,
                initial_state=self.recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            if new_recurrent_state is None:
                raise RuntimeError("GatedDeltaNet speculative forward did not return its recurrent state")
            buffers = self._spec_buffers
            buffers["query"].copy_(query)
            buffers["key"].copy_(key)
            buffers["value"].copy_(value)
            buffers["g"].copy_(g)
            buffers["beta"].copy_(beta)
            buffers["conv_extended"].copy_(conv_extended)
            buffers["final_recurrent_state"].copy_(new_recurrent_state)
        else:
            output, new_recurrent_state = self._chunk_gated_delta_rule(
                query,
                key,
                value,
                g,
                beta,
                initial_state=self.recurrent_state,
                output_final_state=(self.recurrent_state is not None),
                use_qk_l2norm_in_kernel=True,
            )
            if new_recurrent_state is not None and self.recurrent_state is not None:
                self.recurrent_state.copy_(new_recurrent_state.to(self.recurrent_state.dtype))

        # output: (B, S, num_v_heads, head_v_dim)
        output = self.norm(output, z)
        output = output.reshape(batch_size, seq_len, self.value_dim)
        output = self.out_proj(output)
        return output
