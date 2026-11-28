"""GDN kernels: optional fast paths (causal_conv1d / FLA), pure-PyTorch
reference kernels, dispatch helpers and the gated RMSNorm variants.

Split out of gated_delta_net.py; the layer itself lives in layer.py and the
public compatibility surface in gated_delta_net.py.

Reference:
  "Delta Net: https://arxiv.org/abs/2406.06484"
  HuggingFace implementation:
    transformers/models/qwen3_5/modeling_qwen3_5.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        from eole.ops.fla import _compiled_fla_causal_conv1d_update

        def causal_conv1d_fn(x, weight, bias=None, activation=None):
            # x: [B, D, L] channel-first → FLA expects [B, L, D] sequence-first
            out, _ = _fla_causal_conv1d(x.transpose(1, 2), weight=weight, bias=bias, activation=activation)
            return out.transpose(1, 2)  # back to [B, D, L]

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
    from eole.ops.fla import _compiled_fla_recurrent_gated_delta_rule

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
        if (
            torch.compiler.is_compiling()
            and not torch.is_grad_enabled()
            and initial_state is not None
            and g is not None
            and beta is not None
            and output_final_state
        ):
            # Materialize outside the opaque operator so surrounding compiled
            # operations can still fuse their stores into these input buffers.
            return _compiled_fla_recurrent_gated_delta_rule(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                g.contiguous(),
                beta.contiguous(),
                initial_state.contiguous(),
                use_qk_l2norm_in_kernel,
            )
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


# Select kernels from the input device so dispatch follows module device moves.
def _causal_conv1d_activation(activation):
    """Keep the PyTorch and optional GPU convolution activation contracts aligned."""
    if activation not in (None, "silu", "swish"):
        raise ValueError(f"Unsupported causal convolution activation: {activation!r}")
    return "silu" if activation == "swish" else activation


def _causal_conv1d_dispatch(x, weight, bias=None, activation=None):
    activation = _causal_conv1d_activation(activation)
    if x.is_cuda and causal_conv1d_fn is not None:
        return causal_conv1d_fn(x, weight, bias, activation=activation)
    output = F.conv1d(x, weight.unsqueeze(1), bias, padding=weight.shape[-1] - 1, groups=x.shape[1])
    output = output[..., : x.shape[-1]]
    return F.silu(output) if activation == "silu" else output


def _causal_conv1d_update_dispatch(x, conv_state, weight, bias=None, activation=None):
    activation = _causal_conv1d_activation(activation)
    kernel = causal_conv1d_update if x.is_cuda and causal_conv1d_update is not None else _torch_causal_conv1d_update
    return kernel(x, conv_state, weight, bias, activation)


def _chunk_gated_delta_rule_dispatch(query, key, value, g, beta, **kwargs):
    kernel = (
        chunk_gated_delta_rule
        if query.is_cuda and chunk_gated_delta_rule is not None
        else _torch_chunk_gated_delta_rule
    )
    return kernel(query, key, value, g, beta, **kwargs)


def _recurrent_gated_delta_rule_dispatch(query, key, value, g, beta, **kwargs):
    kernel = (
        fused_recurrent_gated_delta_rule
        if query.is_cuda and fused_recurrent_gated_delta_rule is not None
        else _torch_recurrent_gated_delta_rule
    )
    return kernel(query, key, value, g, beta, **kwargs)


def _torch_rms_norm_gated(x, g, weight, bias, eps, activation, residual=None, prenorm=False, residual_in_fp32=False):
    """Match FLA's residual/prenorm contract without launching a GPU kernel."""
    values = x.float()
    if residual is not None:
        values = values + residual.float()
    residual_dtype = residual.dtype if residual is not None else (torch.float32 if residual_in_fp32 else x.dtype)
    residual_out = values.to(residual_dtype)
    output = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
    if weight is not None:
        output = output * weight.float()
    if bias is not None:
        output = output + bias.float()
    gate = torch.sigmoid(g.float()) if activation == "sigmoid" else F.silu(g.float())
    output = (output * gate).to(x.dtype)
    return (output, residual_out) if prenorm else output


# ---------------------------------------------------------------------------
# RMSNorm variants used inside GatedDeltaNet
# ---------------------------------------------------------------------------

# Prefer the FLA fused kernel (faster on CUDA) when available; fall back to a
# pure-PyTorch implementation that is behaviourally identical.

try:
    from fla.modules.fused_norm_gate import FusedRMSNormGated as _FLARMSNormGated
    from eole.ops.fla import _compiled_fla_rms_norm_gated

    class RMSNormGated(_FLARMSNormGated):
        """FLA norm with a fullgraph-compatible inference boundary."""

        _fla_backend = True

        def forward(self, x, g, residual=None, prenorm=False, residual_in_fp32=False):
            if not x.is_cuda:
                return _torch_rms_norm_gated(
                    x, g, self.weight, self.bias, self.eps, self.activation, residual, prenorm, residual_in_fp32
                )
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
