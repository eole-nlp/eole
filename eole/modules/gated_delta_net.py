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

        def causal_conv1d_update(x, conv_state, weight, bias=None, activation=None):
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
    from fla.modules.fused_norm_gate import FusedRMSNormGated as RMSNormGated

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

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, hidden_states, attn_mask=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape

        # Mask padding to zero so padding tokens don't corrupt conv/recurrent state
        if attn_mask is not None and attn_mask.shape[1] > 1:
            hidden_states = (hidden_states * attn_mask[:, :, None]).to(hidden_states.dtype)

        use_precomputed = self.conv_state is not None and self.recurrent_state is not None and seq_len == 1

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
        dt = F.softplus(a + self.dt_bias)  # (B, S, num_v_heads)
        A = -torch.exp(self.A_log.float())  # (num_v_heads,)
        g = dt * A.view(1, 1, -1)  # (B, S, num_v_heads)
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
