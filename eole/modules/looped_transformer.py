"""Layer components and traversal for a decoder with cross-pass message gates."""

from dataclasses import dataclass, field
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import skip_init

from eole.modules.moe import MoE


@dataclass
class LayerVisitCache:
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None
    conv: dict[str, torch.Tensor] = field(default_factory=dict)
    verifying: bool = field(default=False, kw_only=True)


@dataclass
class FixedVisitCache(LayerVisitCache):
    slots: torch.Tensor | None = None

    @classmethod
    def from_prefill(cls, layer, capacity, length):
        slots = torch.arange(capacity, dtype=torch.long, device=layer.key.device)
        indices = (slots[: layer.key.size(-2)] + length - layer.key.size(-2)).remainder(capacity)

        def allocate(source):
            buffer = source.new_zeros((*source.shape[:-2], capacity, source.size(-1)))
            return buffer.index_copy_(2, indices, source)

        return cls(
            allocate(layer.key), allocate(layer.value), {name: x.clone() for name, x in layer.conv.items()}, slots
        )


def _linear(in_features, out_features):
    return skip_init(nn.Linear, in_features, out_features, bias=False)


def _linear_norm(width):
    norm = nn.RMSNorm(width)
    norm.preserve_checkpoint_dtype = True
    return norm


def _rms(x, eps, weight=None):
    return F.rms_norm(x.float(), (x.size(-1),), weight=weight, eps=eps)


def _mix(blocks, residual, queries, *, fast_decode=False):
    if fast_decode:
        from eole.triton.looped_decoder import mix_states

        return mix_states(torch.stack((*blocks, residual), dim=-2), queries)
    if not blocks:
        return _rms(residual, 1e-6).unsqueeze(-2).expand(*residual.shape[:-1], queries.size(0), -1)
    states = _rms(torch.stack((*blocks, residual), dim=-2), 1e-6)
    scores = (queries @ states.transpose(-2, -1)) * states.size(-1) ** -0.5
    return scores.softmax(dim=-1) @ states


class ShortConvolution(nn.Module):
    preserve_checkpoint_dtype = True

    def __init__(self, channels, width, *, heads=None, hidden_size=None):
        super().__init__()
        self.heads = heads
        self.width = width
        self.scale = math.sqrt((1 if heads is None else 2) * width + 1)
        self.register_buffer("dynamic_taps" if heads else "taps", torch.empty(channels, width))
        self.project_taps = _linear(hidden_size, heads * width) if heads else None

    def prepare(self):
        if self.project_taps is None:
            self.taps = self.taps.float()
        else:
            self.dynamic_taps = self.dynamic_taps.float()

    def forward(self, x, cache, name, inputs=None, *, fast_decode=False):
        taps = self.taps if self.project_taps is None else self.dynamic_taps
        if fast_decode and isinstance(cache, FixedVisitCache):
            from eole.triton.looped_decoder import conv_step

            dynamic = self.project_taps(inputs) if self.project_taps is not None else None
            return conv_step(x, cache.conv[name], taps, dynamic, self.heads, self.scale)
        channels = x.float().transpose(1, 2).contiguous()
        history = cache.conv.get(name)
        if history is None:
            history = channels.new_zeros(channels.size(0), channels.size(1), self.width - 1)
        context = torch.cat((history, channels), dim=-1)
        tail = context[..., context.size(-1) - self.width + 1 :]
        if isinstance(cache, FixedVisitCache):
            history.copy_(tail)
        else:
            cache.conv[name] = context[..., 1:] if cache.verifying else tail.clone()
        if self.project_taps is None:
            output = F.conv1d(context, taps.unsqueeze(1), groups=channels.size(1)).transpose(1, 2)
        else:
            windows = context.unfold(-1, self.width, 1).transpose(1, 2)
            windows = windows.unflatten(-2, (self.heads, -1))
            dynamic = self.project_taps(inputs).float().unflatten(-1, (self.heads, self.width))
            taps = taps.unflatten(0, (self.heads, -1)) + dynamic.flip(-1).unsqueeze(-2)
            output = (windows * taps).sum(-1).flatten(-2)
        return ((output + x.float()) / self.scale).to(x.dtype)


def _clip(x, threshold, weight=None, *, fast_decode=False):
    if fast_decode:
        from eole.triton.looped_decoder import clip_heads

        return clip_heads(x, threshold, weight)
    values = x.float()
    if weight is not None:
        values = values * weight
    norm = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
    return (values * (threshold / (norm + 1e-6)).clamp(max=1)).to(x.dtype)


class RelativeBiasAttention(nn.Module):
    preserve_checkpoint_dtype = True

    def __init__(self, cfg, sliding):
        super().__init__()
        loop = cfg.layer_loop
        self.heads = cfg.heads
        self.kv_heads = cfg.heads_kv if sliding else cfg.global_heads_kv or cfg.heads_kv
        self.head_dim = cfg.dim_per_head if sliding else cfg.global_head_dim or cfg.dim_per_head
        self.window = cfg.sliding_window if sliding else None
        self.qk_clip = loop.query_clip
        self.relative_clip = loop.relative_clip
        self.log_floor = loop.log_scale_floor
        self.log_alpha = loop.log_scale_alpha
        self.eps = cfg.norm_eps
        hidden = cfg.hidden_size
        query_size = self.heads * self.head_dim
        kv_size = self.kv_heads * self.head_dim
        self.projection_sizes = (query_size, self.heads * loop.relative_dim, query_size, self.heads, kv_size, kv_size)
        self.query_rows = sum(self.projection_sizes[:-2])
        self.input_projections = nn.ModuleList(_linear(hidden, size) for size in self.projection_sizes)
        self.register_buffer("input_weight", None, persistent=False)
        self.output = _linear(query_size, hidden)
        conv_heads = None if sliding else self.kv_heads
        self.k_conv = ShortConvolution(kv_size, loop.convolution_width, heads=conv_heads, hidden_size=hidden)
        self.v_conv = ShortConvolution(kv_size, loop.convolution_width, heads=conv_heads, hidden_size=hidden)
        self.register_buffer("query_scale", torch.empty(self.heads, self.head_dim))
        self.register_buffer("relative_scale", torch.empty(self.heads, loop.relative_dim))
        self.register_buffer("relative_profiles", torch.empty(loop.relative_dim, self.window or loop.relative_extent))

    def prepare(self, packed):
        self.query_scale = self.query_scale.float()
        self.relative_scale = self.relative_scale.float()
        self.relative_profiles = self.relative_profiles.to(self.output.weight.dtype)
        self.k_conv.prepare()
        self.v_conv.prepare()
        if packed:
            self.input_weight = torch.cat([linear.weight for linear in self.input_projections])
            for linear, weight in zip(self.input_projections, self.input_weight.split(self.projection_sizes)):
                linear.weight.data = weight

    def forward(self, queries, keys, values, positions, cache, *, fast_decode=False):
        q, relative, gate, sink, k, v = self._project_inputs(queries, keys, values, fast_decode)
        q = _clip(q.unflatten(-1, (self.heads, self.head_dim)), self.qk_clip, self.query_scale, fast_decode=fast_decode)
        k = self.k_conv(k, cache, "k", keys, fast_decode=fast_decode).unflatten(-1, (self.kv_heads, self.head_dim))
        v = self.v_conv(v, cache, "v", values, fast_decode=fast_decode).unflatten(-1, (self.kv_heads, self.head_dim))
        k = _clip(k, self.qk_clip, fast_decode=fast_decode).transpose(1, 2)
        v = _rms(v, self.eps).to(v.dtype).transpose(1, 2)
        # Keep the one-head softmax tile bounded for long global caches.
        fused_probs = fast_decode and cache.slots.numel() <= 16384
        k, v, distances = self._cache_with_current(k, v, positions, cache, fused_probs=fused_probs)

        relative = _clip(
            relative.unflatten(-1, (self.heads, -1)), self.relative_clip, self.relative_scale, fast_decode=fast_decode
        )
        if self.window is None:
            log_scale = 1 + self.log_alpha * ((positions + 1).float() / self.log_floor).clamp(min=1).log()
            q = (q.float() * log_scale[:, None, None]).to(q.dtype)
            relative = (relative.float() * log_scale[:, None, None]).to(relative.dtype)

        profiles = (relative @ self.relative_profiles).transpose(1, 2)
        repeats = self.heads // self.kv_heads
        if fast_decode:
            q = q.reshape(1, self.kv_heads, repeats, self.head_dim)
            dots = (q @ k.transpose(-1, -2)).reshape(1, self.heads, 1, k.size(-2))
        else:
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
            dots = q.transpose(1, 2) @ k.transpose(-1, -2)

        scale = self.head_dim**-0.5
        if fused_probs:
            from eole.triton.looped_decoder import attention_probabilities

            probabilities = attention_probabilities(dots, profiles, sink, positions, self.window, scale)
        else:
            extent = profiles.size(-1)
            relative_indices = distances.clamp(0, extent - 1).expand(*profiles.shape[:2], -1, -1)
            bias = profiles.gather(-1, relative_indices).masked_fill((distances < 0) | (distances >= extent), 0)
            masked = (distances < 0) | (distances > positions[:, None])
            if self.window is not None:
                masked = masked | (distances >= self.window)
            logits = (dots * scale + bias).masked_fill(masked, -torch.inf)
            sink = sink.transpose(1, 2).unsqueeze(-1)
            probabilities = torch.cat((logits, sink), dim=-1).float().softmax(dim=-1)[..., :-1].to(q.dtype)
        if fast_decode:
            output = (probabilities.reshape(1, self.kv_heads, repeats, -1) @ v).reshape(1, 1, -1)
        else:
            output = (probabilities @ v).transpose(1, 2).flatten(-2)
        return self.output((output.float() * gate.float().sigmoid()).to(output.dtype))

    def _project_inputs(self, queries, keys, values, fast_decode):
        if fast_decode and queries is keys and queries is values:
            return F.linear(queries, self.input_weight).split(self.projection_sizes, dim=-1)
        if fast_decode:
            projections = F.linear(queries, self.input_weight[: self.query_rows]).split(
                self.projection_sizes[:-2], dim=-1
            )
            key_weight, value_weight = self.input_weight[self.query_rows :].split(self.projection_sizes[-2:])
            return (*projections, F.linear(keys, key_weight), F.linear(values, value_weight))
        inputs = (queries,) * 4 + (keys, values)
        return tuple(projection(x) for x, projection in zip(inputs, self.input_projections))

    def _cache_with_current(self, k, v, positions, cache, *, fused_probs):
        if isinstance(cache, FixedVisitCache):
            index = positions if self.window is None else positions.remainder(cache.slots.numel())
            cache.key.index_copy_(2, index, k)
            cache.value.index_copy_(2, index, v)
            if fused_probs:
                return cache.key, cache.value, None
            distances = positions[:, None] - cache.slots
            if self.window is not None:
                distances = distances.remainder(cache.slots.numel())
            return cache.key, cache.value, distances

        if cache.key is not None:
            k = torch.cat((cache.key, k), dim=-2)
            v = torch.cat((cache.value, v), dim=-2)
        if not cache.verifying and self.window is not None and k.size(-2) >= self.window:
            # Keep window - 1 earlier tokens for the next query.
            start = k.size(-2) - self.window + 1
            cache.key, cache.value = k[..., start:, :].contiguous(), v[..., start:, :].contiguous()
        else:
            cache.key, cache.value = k, v
        key_positions = torch.arange(k.size(-2), device=k.device) + positions[-1] + 1 - k.size(-2)
        return k, v, positions[:, None] - key_positions


class CrossPassGates(nn.Module):
    """Three independent attention controllers and one dependent MoE controller."""

    def __init__(self, hidden, rank, eps):
        super().__init__()
        self.eps = eps
        self.down_projections = nn.ModuleList(_linear(hidden * 2, rank) for _ in range(4))
        self.forget_projections = nn.ModuleList(_linear(rank, hidden) for _ in range(4))
        self.gain_projections = nn.ModuleList(_linear(rank, hidden) for _ in range(4))
        self.down = self.up = None

    def prepare(self, packed):
        self.forget_projections.float()
        self.gain_projections.float()
        if packed:
            self.down = torch.stack([projection.weight for projection in self.down_projections])
            self.up = torch.stack(
                [
                    torch.cat((forget.weight, gain.weight))
                    for forget, gain in zip(self.forget_projections, self.gain_projections)
                ]
            )
            for i, (down, forget, gain) in enumerate(
                zip(self.down_projections, self.forget_projections, self.gain_projections)
            ):
                down.weight.data = self.down[i]
                forget.weight.data, gain.weight.data = self.up[i].chunk(2)

    def forward(self, x, message, index):
        features = torch.cat((_rms(x, self.eps), _rms(message, self.eps)), dim=-1)
        down = self.down_projections[index]
        latent = down(features.to(down.weight.dtype)).float()
        forget = self.forget_projections[index](latent).tanh()
        gain = self.gain_projections[index](latent).tanh()
        return (1 - forget) * x.float() + gain * message

    def decode(self, x, messages, norm, dtype, *, moe=False):
        from eole.triton.looped_decoder import gate_features, gate_output

        x = x.reshape(-1, x.size(-1)).contiguous()
        messages = messages.reshape_as(x).contiguous()
        features = gate_features(x, messages, self.down.dtype, self.eps)
        if moe:
            latent = F.linear(features, self.down[3]).float()
            gates = F.linear(latent, self.up[3])
        else:
            latent = torch.bmm(self.down[:3], features.unsqueeze(-1)).float()
            gates = torch.bmm(self.up[:3], latent).squeeze(-1)
        return gate_output(x, messages, gates, norm, dtype, self.eps)


class BlockResidualLayer(nn.Module):
    preserve_checkpoint_dtype = True

    def __init__(self, cfg, index, running_config):
        super().__init__()
        self.sliding = cfg.layer_types[index] == "sliding_attention"
        loop = cfg.layer_loop
        self.boundary = index % loop.residual_block_size == 0
        self.eps = cfg.norm_eps
        hidden = cfg.hidden_size
        self.local_attention_norm = _linear_norm(hidden) if self.sliding else None
        self.attention_norms = nn.ModuleList(_linear_norm(hidden) for _ in range(3)) if not self.sliding else None
        self.register_buffer("moe_norm", torch.empty(hidden))
        self.register_buffer("moe_query", torch.empty(hidden))
        query = torch.empty(hidden) if self.sliding else torch.empty(3, hidden)
        self.register_buffer("attention_query", query if index else None)
        self.attention = RelativeBiasAttention(cfg, self.sliding)
        self.attn_conv = ShortConvolution(hidden, loop.convolution_width)
        self.moe_conv = ShortConvolution(hidden, loop.convolution_width)
        self.moe = MoE(cfg, running_config, expert_format=loop.expert_formats[index])
        self.controllers = (
            CrossPassGates(hidden, loop.gate_rank, self.eps) if loop.core_start <= index < loop.core_end else None
        )

    def prepare(self, packed):
        self.attention.prepare(packed)
        self.attn_conv.prepare()
        self.moe_conv.prepare()
        self.moe.gate.float()
        self.moe.experts.prepare()
        self.moe_norm = self.moe_norm.float()
        norm_layers = (self.local_attention_norm,) if self.sliding else self.attention_norms
        norms = torch.stack([norm.weight.float() for norm in norm_layers])
        self._attention_norms = norms.expand(3, -1)
        if self.controllers is not None:
            self.controllers.prepare(packed)

    def forward(self, residual, blocks, positions, cache, messages=None, *, fast_decode=False):
        dtype = residual.dtype
        if self.attention_query is None:
            qkv = (residual,)
        else:
            qkv = _mix(
                blocks, residual, self.attention_query.view(-1, residual.size(-1)), fast_decode=fast_decode
            ).unbind(-2)
        norms = self._attention_norms
        if fast_decode and messages is not None:
            stacked = qkv[0].view(1, -1).expand(3, -1) if len(qkv) == 1 else torch.cat(qkv, dim=1)
            inputs = self.controllers.decode(stacked, messages[..., :3, :], norms, dtype)
            inputs = inputs.view(3, 1, 1, -1).unbind(0)
        elif fast_decode:
            from eole.triton.looped_decoder import normalize

            if len(qkv) == 1 and self.sliding:
                inputs = (normalize(qkv[0], norms[0], dtype, self.eps),) * 3
            else:
                stacked = torch.cat(qkv if len(qkv) == 3 else qkv * 3, dim=1)
                inputs = normalize(stacked, norms, dtype, self.eps).view(3, 1, 1, -1).unbind(0)
        else:
            if len(qkv) == 1:
                qkv = qkv * 3
            if messages is not None:
                qkv = [self.controllers(x, messages[..., i, :], i) for i, x in enumerate(qkv)]
            inputs = [_rms(x, self.eps, norm).to(dtype) for x, norm in zip(qkv, norms)]

        attended = self.attention(*inputs, positions, cache, fast_decode=fast_decode)
        output = self.attn_conv(attended, cache, "attn", fast_decode=fast_decode).float()
        if self.boundary:
            blocks.append(residual)
        else:
            output = residual.float() + output

        inputs = _mix(blocks, output, self.moe_query.unsqueeze(0), fast_decode=fast_decode).squeeze(-2)
        if fast_decode and messages is not None:
            normalized = self.controllers.decode(
                inputs, messages[..., 3, :], self.moe_norm.unsqueeze(0), dtype, moe=True
            ).view_as(inputs)
        elif fast_decode:
            from eole.triton.looped_decoder import normalize

            normalized = normalize(inputs, self.moe_norm, dtype, self.eps)
        else:
            if messages is not None:
                inputs = self.controllers(inputs, messages[..., 3, :], 3)
            normalized = _rms(inputs, self.eps, self.moe_norm).to(dtype)
        hidden = self.moe(normalized, fast_decode=fast_decode)
        return (output + self.moe_conv(hidden, cache, "moe", fast_decode=fast_decode)).to(dtype)


class SoftcappedLinear(nn.Linear):
    """Vocabulary head with a bounded logit range and an unpadded output vocabulary."""

    def __init__(self, hidden, padded_vocab, vocab_size, cap, device=None):
        super().__init__(hidden, padded_vocab, bias=False, device=device)
        self.vocab_size = vocab_size
        self.cap = cap

    def forward(self, hidden):
        logits = F.linear(hidden, self.weight[: self.vocab_size]).float()
        logits = self.cap * (logits / self.cap).tanh()
        return F.pad(logits, (0, self.out_features - self.vocab_size), value=-torch.inf)


@dataclass
class BlockResidualState:
    blocks: list[torch.Tensor] = field(default_factory=list)
    prelude: torch.Tensor | None = None
    prelude_blocks: tuple = ()
    messages_from: torch.Tensor | None = None
    message_blocks: tuple = ()


class LoopedLayerSchedule(nn.Module):
    """One set of weights per physical layer and one cache per execution visit."""

    preserve_checkpoint_dtype = True

    def __init__(self, cfg):
        super().__init__()
        loop = cfg.layer_loop
        start, end, passes = loop.core_start, loop.core_end, loop.passes
        if (
            not 0 < start < end < cfg.layers
            or start % loop.residual_block_size
            or end % loop.residual_block_size
            or len(loop.expert_formats) != cfg.layers
            or len(cfg.layer_types or ()) != cfg.layers
            or any(kind not in ("sliding_attention", "full_attention") for kind in cfg.layer_types)
            or cfg.sliding_window <= 0
            or cfg.with_cross_attn
        ):
            raise ValueError("Invalid block-residual loop configuration")
        self.layer_ids = tuple(range(start)) + tuple(range(start, end)) * passes + tuple(range(end, cfg.layers))
        self.pass_starts = tuple(range(start, start + (end - start) * passes, end - start))
        self.core_start = start
        self.loop_end_visit = start + (end - start) * passes
        self.eps = cfg.norm_eps
        self.register_buffer("embedding_norm", torch.empty(cfg.hidden_size))
        self.register_buffer("output_query", torch.empty(cfg.hidden_size))
        self.register_buffer("cross_pass_queries", torch.empty(2 * (end - start), cfg.hidden_size))
        self.register_buffer("cross_pass_kv_queries", torch.empty(2 * (end - start), cfg.hidden_size))

    def normalize_embedding(self, emb):
        return _rms(emb, self.eps, self.embedding_norm).to(emb.dtype)

    def new_cache(self):
        return [LayerVisitCache() for _ in self.layer_ids]

    def prepare(self):
        self.embedding_norm = self.embedding_norm.float()
        q, kv = self.cross_pass_queries.float(), self.cross_pass_kv_queries.float()
        self.message_queries = torch.stack((q[::2], kv[::2], kv[1::2], q[1::2]), dim=1)

    def before_visit(self, visit, index, hidden, state, fast_decode):
        if visit == self.core_start:
            state.prelude, state.prelude_blocks = hidden, tuple(state.blocks)
        elif visit in self.pass_starts:
            state.messages_from = hidden
            state.message_blocks = tuple(state.blocks[len(state.prelude_blocks) + 1 :])
            hidden, state.blocks = state.prelude, list(state.prelude_blocks)
        messages = None
        if self.core_start < visit < self.loop_end_visit and state.messages_from is not None:
            messages = _mix(
                state.message_blocks,
                state.messages_from,
                self.message_queries[index - self.core_start],
                fast_decode=fast_decode,
            )
        return hidden, messages

    def finish(self, hidden, state, fast_decode):
        return _mix(state.blocks, hidden, self.output_query.view(1, -1), fast_decode=fast_decode).squeeze(-2)
