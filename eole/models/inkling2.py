"""Text inference for the fixed-pass Inkling 2 Mini checkpoint."""

import math
from contextlib import ExitStack
from dataclasses import dataclass, field
import json
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F


@dataclass
class Inkling2LayerCache:
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None
    conv: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class Inkling2StaticLayerCache(Inkling2LayerCache):
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


@dataclass
class Inkling2Cache:
    """One KV and convolution state per layer visit, with a shared token position."""

    layers: list[Inkling2LayerCache]
    length: int = 0


def _rms(x, eps, weight=None):
    return F.rms_norm(x.float(), (x.size(-1),), weight=weight, eps=eps)


def _mix(blocks, residual, queries, *, fast_decode=False):
    if fast_decode:
        from eole.triton.inkling2_decode import mix_states

        return mix_states(torch.stack((*blocks, residual), dim=-2), queries)
    if not blocks:
        return _rms(residual, 1e-6).unsqueeze(-2).expand(*residual.shape[:-1], queries.size(0), -1)
    states = _rms(torch.stack((*blocks, residual), dim=-2), 1e-6)
    scores = (queries @ states.transpose(-2, -1)) * states.size(-1) ** -0.5
    return scores.softmax(dim=-1) @ states


def _linear(read, name):
    weight = read(name)
    layer = nn.Linear(weight.size(1), weight.size(0), bias=False, device="meta")
    layer.weight = nn.Parameter(weight, requires_grad=False)
    return layer


class Inkling2PackedExperts(nn.Module):
    """Keep released expert weights packed, dequantizing only selected experts to BF16."""

    def __init__(self, read, name, shape):
        super().__init__()
        weight, scales = read(name), read(name + "_scale")
        experts, rows, cols = shape
        self.nvfp4 = weight.dtype == torch.uint8
        self.block_size = 16 if self.nvfp4 else 32
        packed_cols = cols // 2 if self.nvfp4 else cols
        scale_dtype = torch.float8_e4m3fn if self.nvfp4 else torch.uint8
        weight_dtype = torch.uint8 if self.nvfp4 else torch.float8_e4m3fn
        if (
            weight.dtype != weight_dtype
            or weight.shape != (experts, rows, packed_cols)
            or scales.dtype != scale_dtype
            or cols % self.block_size
            or scales.shape != (experts, rows, cols // self.block_size)
        ):
            raise ValueError(f"Unsupported weights or scales for {name}: {weight.shape}, {scales.shape}")
        # Store bytes so routed gathers also work on devices without FP8 indexing kernels.
        self.register_buffer("weight", weight.view(torch.uint8))
        self.register_buffer("scales", scales.view(torch.uint8))
        if self.nvfp4:
            amax, denominator = read(name + "_amax"), read(name + "_outer_scale_denominator")
            if amax.shape != (experts,) or denominator.numel() != 1:
                raise ValueError(f"Unexpected global scales for {name}")
            self.register_buffer("outer_scale", amax.float() / denominator.float())
            self.register_buffer("fp4_values", weight.new_tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32))

    def forward(self, indices, dtype):
        packed = self.weight[indices]
        scales = self.scales[indices]
        if self.nvfp4:
            codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
            magnitude = self.fp4_values.index_select(0, (codes & 7).flatten().int()).view_as(codes)
            values = torch.where(codes < 8, magnitude, -magnitude)
            block_scales = scales.view(torch.float8_e4m3fn).float()
        else:
            values = packed.view(torch.float8_e4m3fn).float()
            block_scales = torch.ldexp(torch.ones_like(scales, dtype=torch.float32), scales.int() - 127)

        dequantized = (values.unflatten(-1, (-1, self.block_size)) * block_scales.unsqueeze(-1)).flatten(-2)
        if self.nvfp4:
            dequantized = dequantized * self.outer_scale[indices][..., None, None]
        return dequantized.to(dtype)


class Inkling2MoE(nn.Module):
    def __init__(self, read, cfg):
        super().__init__()
        experts, hidden, intermediate = cfg["n_routed_experts"], cfg["hidden_size"], cfg["intermediate_size"]
        self.top_k = cfg["num_experts_per_tok"]
        self.route_scale = cfg["route_scale"]
        self.gate_softcap = cfg["mlp_gate_softcap"]
        self.up_softcap = cfg["mlp_up_softcap"]
        self.register_buffer("router", read("gate.weight").float())
        self.register_buffer("correction", read("gate.bias").float())
        self.gate_up = Inkling2PackedExperts(read, "experts.w13_weight", (experts, 2 * intermediate, hidden))
        self.down = Inkling2PackedExperts(read, "experts.w2_weight", (experts, hidden, intermediate))

    def _activate(self, x):
        # The original checkpoint interleaves the gate and up rows.
        gate, up = x.float().unflatten(-1, (-1, 2)).unbind(-1)
        gate_factor = self.gate_softcap * (gate / self.gate_softcap).tanh()
        up_factor = self.up_softcap * (up / self.up_softcap).tanh()
        return (gate_factor * gate.sigmoid() * up_factor).to(x.dtype)

    def forward(self, x, *, packed_experts=False, fast_decode=False):
        flat = x.flatten(0, 1)
        scores = F.linear(flat.float(), self.router).sigmoid()
        indices = (scores + self.correction).topk(self.top_k, sorted=False).indices
        weights = scores.gather(-1, indices)
        weights = weights * (self.route_scale / (weights.sum(dim=-1, keepdim=True) + 1e-20))

        if len(flat) == 1:
            selected = indices[0]
            if packed_experts:
                from eole.triton.inkling2_moe import expert_gemv

                gate_up = expert_gemv(self.gate_up, flat, selected, shared_input=True)
                if fast_decode:
                    from eole.triton.inkling2_decode import situ, sum_routes

                    activated = situ(gate_up, self.gate_softcap, self.up_softcap)
                    routes = expert_gemv(self.down, activated, selected, shared_input=False)
                    return sum_routes(routes, weights[0], x.shape)
                routes = expert_gemv(self.down, self._activate(gate_up), selected, shared_input=False)
            else:
                gate_up = (self.gate_up(selected, x.dtype) @ flat.T).squeeze(-1)
                routes = (self.down(selected, x.dtype) @ self._activate(gate_up).unsqueeze(-1)).squeeze(-1)
            return (routes.float() * weights[0, :, None]).sum(dim=0).to(x.dtype).view_as(x)

        result = torch.zeros_like(flat, dtype=torch.float32)
        # Group prompt tokens by expert to reuse weights and bound dequantization scratch space.
        for expert in indices.unique().tolist():
            token, route = torch.where(indices == expert)
            gate_up = F.linear(flat[token], self.gate_up(expert, x.dtype))
            routes = F.linear(self._activate(gate_up), self.down(expert, x.dtype))
            result.index_add_(0, token, routes.float() * weights[token, route, None])
        return result.to(x.dtype).view_as(x)


class Inkling2Conv(nn.Module):
    def __init__(self, read, *, heads=None):
        super().__init__()
        self.heads = heads
        if heads is None:
            taps = read("weight").squeeze(1)
            self.project_taps = None
        else:
            taps = read("static_w").transpose(0, 1).flip(-1)
            self.project_taps = _linear(read, "weight_proj")
        self.register_buffer("taps", taps.float().contiguous())
        self.width = taps.size(-1)
        self.scale = math.sqrt((1 if heads is None else 2) * self.width + 1)

    def forward(self, x, cache, name, inputs=None, *, fast_decode=False):
        if fast_decode and isinstance(cache, Inkling2StaticLayerCache):
            from eole.triton.inkling2_decode import conv_step

            dynamic = self.project_taps(inputs) if self.project_taps is not None else None
            return conv_step(x, cache.conv[name], self.taps, dynamic, self.heads, self.scale)
        channels = x.float().transpose(1, 2).contiguous()
        history = cache.conv.get(name)
        if history is None:
            history = channels.new_zeros(channels.size(0), channels.size(1), self.width - 1)
        context = torch.cat((history, channels), dim=-1)
        tail = context[..., context.size(-1) - self.width + 1 :]
        if isinstance(cache, Inkling2StaticLayerCache):
            history.copy_(tail)
        else:
            cache.conv[name] = tail.clone()
        if self.project_taps is None:
            output = F.conv1d(context, self.taps.unsqueeze(1), groups=channels.size(1)).transpose(1, 2)
        else:
            windows = context.unfold(-1, self.width, 1).transpose(1, 2)
            windows = windows.unflatten(-2, (self.heads, -1))
            dynamic = self.project_taps(inputs).float().unflatten(-1, (self.heads, self.width))
            taps = self.taps.unflatten(0, (self.heads, -1)) + dynamic.flip(-1).unsqueeze(-2)
            output = (windows * taps).sum(-1).flatten(-2)
        return ((output + x.float()) / self.scale).to(x.dtype)


def _clip(x, threshold, weight=None, *, fast_decode=False):
    if fast_decode:
        from eole.triton.inkling2_decode import clip_heads

        return clip_heads(x, threshold, weight)
    values = x.float()
    if weight is not None:
        values = values * weight
    norm = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
    return (values * (threshold / (norm + 1e-6)).clamp(max=1)).to(x.dtype)


class Inkling2Attention(nn.Module):
    def __init__(self, read, cfg, sliding):
        super().__init__()
        prefix = "swa_" if sliding else ""
        self.heads = cfg[prefix + "num_attention_heads"]
        self.kv_heads = cfg[prefix + "num_key_value_heads"]
        self.head_dim = cfg[prefix + "head_dim"]
        self.window = cfg["sliding_window_size"] if sliding else None
        self.qk_clip = cfg["qk_l2_clip_threshold"]
        self.relative_clip = cfg["rel_l2_clip_threshold"]
        self.log_floor = cfg["log_scaling_n_floor"]
        self.log_alpha = cfg["log_scaling_alpha"]
        self.eps = cfg["rms_norm_eps"]
        # The query, relative bias, output gate and sink always share an input.
        projection_names = ("wq_du", "wr_du", "gate_attn", "attn_sink_proj", "wk_dv", "wv_dv")
        weights = [read(f"{name}.weight") for name in projection_names]
        self.projection_sizes = tuple(weight.size(0) for weight in weights)
        self.query_rows = sum(self.projection_sizes[:-2])
        self.register_buffer("input_weight", torch.cat(weights))
        self.output = _linear(read, "wo_ud.weight")
        conv_heads = None if sliding else self.kv_heads
        self.k_conv = Inkling2Conv(lambda name: read("k_sconv." + name), heads=conv_heads)
        self.v_conv = Inkling2Conv(lambda name: read("v_sconv." + name), heads=conv_heads)
        self.register_buffer("query_scale", read("q_l2_clip.scale").float())
        self.register_buffer("relative_scale", read("rel_l2_clip.scale").float())
        self.register_buffer("relative_profiles", read("rel_logits_proj.proj"))

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
            from eole.triton.inkling2_decode import attention_probabilities

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
        return tuple(F.linear(x, w) for x, w in zip(inputs, self.input_weight.split(self.projection_sizes)))

    def _cache_with_current(self, k, v, positions, cache, *, fused_probs):
        if isinstance(cache, Inkling2StaticLayerCache):
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
        if self.window is not None and k.size(-2) >= self.window:
            # Keep window - 1 earlier tokens for the next query.
            start = k.size(-2) - self.window + 1
            cache.key, cache.value = k[..., start:, :].contiguous(), v[..., start:, :].contiguous()
        else:
            cache.key, cache.value = k, v
        key_positions = torch.arange(k.size(-2), device=k.device) + positions[-1] + 1 - k.size(-2)
        return k, v, positions[:, None] - key_positions


class Inkling2MessageGates(nn.Module):
    """Three independent attention controllers and one dependent MoE controller."""

    def __init__(self, read, eps):
        super().__init__()
        self.eps = eps
        names = ("attn", "k", "v", "mlp")
        self.register_buffer("down", torch.stack([read(f"{name}.down_weight") for name in names]))
        self.register_buffer(
            "up",
            torch.stack(
                [torch.cat([read(f"{name}.{gate}_weight") for gate in ("forget", "gain")]).float() for name in names]
            ),
        )

    def forward(self, x, message, index):
        features = torch.cat((_rms(x, self.eps), _rms(message, self.eps)), dim=-1)
        latent = F.linear(features.to(self.down.dtype), self.down[index]).float()
        forget_weight, gain_weight = self.up[index].chunk(2)
        return (1 - F.linear(latent, forget_weight).tanh()) * x.float() + F.linear(latent, gain_weight).tanh() * message

    def decode(self, x, messages, norm, dtype, *, moe=False):
        from eole.triton.inkling2_decode import gate_features, gate_output

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


class Inkling2Layer(nn.Module):
    def __init__(self, read, cfg, index):
        super().__init__()
        self.sliding = index in cfg["local_layer_ids"]
        self.boundary = index % cfg["n_layers_per_attn_res_block"] == 0
        self.eps = cfg["rms_norm_eps"]
        loop = cfg["loop_config"]

        def layer_weight(name):
            return read(f"layers.{index}.{name}")

        norm_names = ("attn_norm",) if self.sliding else ("q_input_norm", "k_input_norm", "v_input_norm")
        norms = torch.stack([layer_weight(name + ".weight").float() for name in norm_names])
        self.register_buffer("attention_norms", norms)
        self.register_buffer("moe_norm", layer_weight("mlp_norm.weight").float())
        self.register_buffer("moe_query", read(f"attn_res.mlp_queries.{index}").reshape(1, -1).float())
        query = None if index == 0 else read(f"attn_res.attn_queries.{index - 1}").reshape(len(norm_names), -1)
        self.register_buffer("attention_query", query.float() if query is not None else None)
        self.attention = Inkling2Attention(lambda name: layer_weight("attn." + name), cfg, self.sliding)
        self.attn_conv = Inkling2Conv(lambda name: layer_weight("attn_sconv." + name))
        self.moe_conv = Inkling2Conv(lambda name: layer_weight("mlp_sconv." + name))
        self.moe = Inkling2MoE(lambda name: layer_weight("mlp." + name), cfg)
        if loop["core_start"] <= index < loop["core_end"]:
            self.controllers = Inkling2MessageGates(lambda name: layer_weight("controllers." + name), self.eps)

    def forward(self, residual, blocks, positions, cache, messages=None, *, packed_experts=False):
        dtype = residual.dtype
        fast_decode = packed_experts and isinstance(cache, Inkling2StaticLayerCache)
        if self.attention_query is None:
            qkv = (residual,)
        else:
            qkv = _mix(blocks, residual, self.attention_query, fast_decode=fast_decode).unbind(-2)
        norms = self.attention_norms.expand(3, -1)
        if fast_decode and messages is not None:
            stacked = qkv[0].view(1, -1).expand(3, -1) if len(qkv) == 1 else torch.cat(qkv, dim=1)
            inputs = self.controllers.decode(stacked, messages[..., :3, :], norms, dtype)
            inputs = inputs.view(3, 1, 1, -1).unbind(0)
        elif fast_decode:
            from eole.triton.inkling2_decode import normalize

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

        inputs = _mix(blocks, output, self.moe_query, fast_decode=fast_decode).squeeze(-2)
        if fast_decode and messages is not None:
            normalized = self.controllers.decode(
                inputs, messages[..., 3, :], self.moe_norm.unsqueeze(0), dtype, moe=True
            ).view_as(inputs)
        elif fast_decode:
            from eole.triton.inkling2_decode import normalize

            normalized = normalize(inputs, self.moe_norm, dtype, self.eps)
        else:
            if messages is not None:
                inputs = self.controllers(inputs, messages[..., 3, :], 3)
            normalized = _rms(inputs, self.eps, self.moe_norm).to(dtype)
        hidden = self.moe(normalized, packed_experts=packed_experts, fast_decode=fast_decode)
        return (output + self.moe_conv(hidden, cache, "moe", fast_decode=fast_decode)).to(dtype)


class Inkling2Mini(nn.Module):
    def __init__(self, read, cfg):
        super().__init__()
        loop = cfg["loop_config"]
        self.core_start, self.core_end, self.passes = loop["core_start"], loop["core_end"], loop["num_passes"]
        block_size = cfg["n_layers_per_attn_res_block"]
        layer_count = cfg["num_hidden_layers"]
        if (
            loop["is_varloop"]
            or not 0 < self.core_start < self.core_end < layer_count
            or self.core_start % block_size
            or self.core_end % block_size
            or self.passes < 2
            or cfg["n_shared_experts"]
            or cfg["q_bias"]
            or cfg["o_bias"]
            or not cfg["use_gate_bias"]
            or not cfg["use_value_norm_no_gamma"]
            or not cfg["attn_res_normalize_values"]
        ):
            raise ValueError("Unsupported Inkling 2 Mini text configuration")

        self.visit_order = (
            tuple(range(self.core_start))
            + tuple(range(self.core_start, self.core_end)) * self.passes
            + tuple(range(self.core_end, layer_count))
        )
        self.num_visits = len(self.visit_order)
        self.eps = cfg["rms_norm_eps"]
        self.softcap = cfg["final_logit_softcapping"]
        self.vocab_size = cfg["unpadded_vocab_size"]
        self.register_buffer("embedding", read("embed.weight"))
        self.register_buffer("embedding_norm", read("embed_norm.weight").float())
        self.register_buffer("output_weight", read("unembed.weight")[: self.vocab_size])
        self.register_buffer("output_norm", read("norm.weight").float())
        self.register_buffer("output_query", read("attn_res.output_query").reshape(1, -1).float())
        q, kv = (
            read("middle_cycle_cross_pass" + suffix).unflatten(0, (-1, 2)) for suffix in ("_queries", "_kv_queries")
        )
        self.register_buffer("message_queries", torch.stack((q[:, 0], kv[:, 0], kv[:, 1], q[:, 1]), dim=1).float())
        self.layers = nn.ModuleList(Inkling2Layer(read, cfg, i) for i in range(layer_count))

    @classmethod
    def from_checkpoint(cls, path, device):
        path = Path(path)
        config = json.loads((path / "config.json").read_text())
        if config["model_type"] != "inkling2_loop":
            raise ValueError("Expected the official thinkingmachines/Inkling-2-Mini checkpoint")
        prefix = "model.llm."
        index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
        with ExitStack() as stack:
            shards = {
                name: stack.enter_context(safe_open(path / name, framework="pt", device="cpu"))
                for name in {file for key, file in index.items() if key.startswith(prefix)}
            }

            def read(name):
                key = prefix + name
                return shards[index[key]].get_tensor(key).to(device)

            return cls(read, config["text_config"]).eval()

    def new_cache(self):
        return Inkling2Cache([Inkling2LayerCache() for _ in range(self.num_visits)])

    def forward(self, input_ids, cache=None, *, all_logits=False, packed_experts=False):
        if cache is None:
            cache = self.new_cache()
        positions = torch.arange(cache.length, cache.length + input_ids.size(1), device=input_ids.device)
        logits = self._forward(input_ids, cache.layers, positions, all_logits, packed_experts)
        cache.length += input_ids.size(1)
        return logits

    def _forward(self, input_ids, layers, positions, all_logits, packed_experts):
        fast_decode = packed_experts and isinstance(layers[0], Inkling2StaticLayerCache)
        hidden = _rms(F.embedding(input_ids, self.embedding), self.eps, self.embedding_norm).to(self.embedding.dtype)
        blocks = []
        visit = 0

        for layer in self.layers[: self.core_start]:
            hidden = layer(hidden, blocks, positions, layers[visit], packed_experts=packed_experts)
            visit += 1

        prelude, prelude_blocks = hidden, tuple(blocks)
        previous_blocks = ()
        previous_hidden = None
        for pass_idx in range(self.passes):
            hidden, blocks = prelude, list(prelude_blocks)
            for offset, layer in enumerate(self.layers[self.core_start : self.core_end]):
                messages = None
                if pass_idx:
                    messages = _mix(
                        previous_blocks, previous_hidden, self.message_queries[offset], fast_decode=fast_decode
                    )
                hidden = layer(hidden, blocks, positions, layers[visit], messages, packed_experts=packed_experts)
                visit += 1
            previous_blocks = tuple(blocks[len(prelude_blocks) + 1 :])
            previous_hidden = hidden

        for layer in self.layers[self.core_end :]:
            hidden = layer(hidden, blocks, positions, layers[visit], packed_experts=packed_experts)
            visit += 1

        if not all_logits:
            hidden = hidden[:, -1:]
            blocks = [block[:, -1:] for block in blocks]
        hidden = _mix(blocks, hidden, self.output_query, fast_decode=fast_decode).squeeze(-2)
        hidden = _rms(hidden, self.eps, self.output_norm).to(self.embedding.dtype)
        logits = F.linear(hidden, self.output_weight).float()
        return self.softcap * (logits / self.softcap).tanh()
