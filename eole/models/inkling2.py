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
class Inkling2Cache:
    """One KV and convolution state per layer visit, with a shared token position."""

    layers: list[Inkling2LayerCache]
    length: int = 0


def _rms(x, eps, weight=None):
    return F.rms_norm(x.float(), (x.size(-1),), weight=weight, eps=eps)


def _mix(blocks, residual, queries):
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

    def forward(self, x):
        flat = x.flatten(0, 1)
        scores = F.linear(flat.float(), self.router).sigmoid()
        indices = (scores + self.correction).topk(self.top_k, sorted=False).indices
        weights = scores.gather(-1, indices)
        weights = weights * (self.route_scale / (weights.sum(dim=-1, keepdim=True) + 1e-20))

        if len(flat) == 1:
            selected = indices[0]
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

    def forward(self, x, cache, name, inputs=None):
        channels = x.float().transpose(1, 2).contiguous()
        history = cache.conv.get(name)
        if history is None:
            history = channels.new_zeros(channels.size(0), channels.size(1), self.width - 1)
        context = torch.cat((history, channels), dim=-1)
        cache.conv[name] = context[..., context.size(-1) - self.width + 1 :].clone()
        if self.project_taps is None:
            output = F.conv1d(context, self.taps.unsqueeze(1), groups=channels.size(1)).transpose(1, 2)
        else:
            windows = context.unfold(-1, self.width, 1).transpose(1, 2)
            windows = windows.unflatten(-2, (self.heads, -1))
            dynamic = self.project_taps(inputs).float().unflatten(-1, (self.heads, self.width))
            taps = self.taps.unflatten(0, (self.heads, -1)) + dynamic.flip(-1).unsqueeze(-2)
            output = (windows * taps).sum(-1).flatten(-2)
        return ((output + x.float()) / self.scale).to(x.dtype)


def _clip(x, threshold, weight=None):
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
        self.q = _linear(read, "wq_du.weight")
        self.k = _linear(read, "wk_dv.weight")
        self.v = _linear(read, "wv_dv.weight")
        self.relative = _linear(read, "wr_du.weight")
        self.output = _linear(read, "wo_ud.weight")
        self.output_gate = _linear(read, "gate_attn.weight")
        self.sink = _linear(read, "attn_sink_proj.weight")
        conv_heads = None if sliding else self.kv_heads
        self.k_conv = Inkling2Conv(lambda name: read("k_sconv." + name), heads=conv_heads)
        self.v_conv = Inkling2Conv(lambda name: read("v_sconv." + name), heads=conv_heads)
        self.register_buffer("query_scale", read("q_l2_clip.scale").float())
        self.register_buffer("relative_scale", read("rel_l2_clip.scale").float())
        self.register_buffer("relative_profiles", read("rel_logits_proj.proj"))

    def forward(self, queries, keys, values, positions, cache):
        q = _clip(self.q(queries).unflatten(-1, (self.heads, self.head_dim)), self.qk_clip, self.query_scale)
        k = self.k_conv(self.k(keys), cache, "k", keys).unflatten(-1, (self.kv_heads, self.head_dim))
        v = self.v_conv(self.v(values), cache, "v", values).unflatten(-1, (self.kv_heads, self.head_dim))
        k, v = _clip(k, self.qk_clip).transpose(1, 2), _rms(v, self.eps).to(v.dtype).transpose(1, 2)
        if cache.key is not None:
            k = torch.cat((cache.key, k), dim=-2)
            v = torch.cat((cache.value, v), dim=-2)
        if self.window is not None and k.size(-2) >= self.window:
            # Cache at most window - 1 earlier tokens for the next query.
            start = k.size(-2) - self.window + 1
            cache.key, cache.value = k[..., start:, :].contiguous(), v[..., start:, :].contiguous()
        else:
            cache.key, cache.value = k, v

        relative = _clip(
            self.relative(queries).unflatten(-1, (self.heads, -1)), self.relative_clip, self.relative_scale
        )
        if self.window is None:
            log_scale = 1 + self.log_alpha * ((positions + 1).float() / self.log_floor).clamp(min=1).log()
            q = (q.float() * log_scale[:, None, None]).to(q.dtype)
            relative = (relative.float() * log_scale[:, None, None]).to(relative.dtype)

        end = positions[-1] + 1
        distances = positions[:, None] - (torch.arange(k.size(-2), device=k.device) + end - k.size(-2))
        profiles = (relative @ self.relative_profiles).transpose(1, 2)
        extent = profiles.size(-1)
        relative_indices = distances.clamp(0, extent - 1).expand(*profiles.shape[:2], -1, -1)
        bias = profiles.gather(-1, relative_indices)
        bias = bias.masked_fill((distances < 0) | (distances >= extent), 0)
        masked = distances < 0
        if self.window is not None:
            masked = masked | (distances >= self.window)

        repeats = self.heads // self.kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
        logits = (q.transpose(1, 2) @ k.transpose(-1, -2)) * self.head_dim**-0.5
        logits = (logits + bias).masked_fill(masked, -torch.inf)
        sink = self.sink(queries).transpose(1, 2).unsqueeze(-1)
        probabilities = torch.cat((logits, sink), dim=-1).float().softmax(dim=-1)[..., :-1].to(q.dtype)
        output = (probabilities @ v).transpose(1, 2).flatten(-2)
        return self.output((output.float() * self.output_gate(queries).float().sigmoid()).to(output.dtype))


class Inkling2MessageGate(nn.Module):
    def __init__(self, read, eps):
        super().__init__()
        self.eps = eps
        self.down = _linear(read, "down_weight")
        self.gain = _linear(read, "gain_weight").float()
        self.forget = _linear(read, "forget_weight").float()

    def forward(self, x, message):
        features = torch.cat((_rms(x, self.eps), _rms(message, self.eps)), dim=-1)
        latent = self.down(features.to(self.down.weight.dtype)).float()
        return (1 - self.forget(latent).tanh()) * x.float() + self.gain(latent).tanh() * message


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
            self.controllers = nn.ModuleList(
                Inkling2MessageGate(lambda key: layer_weight(f"controllers.{name}.{key}"), self.eps)
                for name in ("attn", "k", "v", "mlp")
            )

    def forward(self, residual, blocks, positions, cache, messages=None):
        dtype = residual.dtype
        qkv = (residual,) if self.attention_query is None else _mix(blocks, residual, self.attention_query).unbind(-2)
        if len(qkv) == 1:
            qkv = qkv * 3
        if messages is not None:
            qkv = [self.controllers[i](x, messages[..., i, :]) for i, x in enumerate(qkv)]
        norms = self.attention_norms.expand(3, -1)
        inputs = [_rms(x, self.eps, norm).to(dtype) for x, norm in zip(qkv, norms)]
        output = self.attn_conv(self.attention(*inputs, positions, cache), cache, "attn").float()
        if self.boundary:
            blocks.append(residual)
        else:
            output = residual.float() + output

        inputs = _mix(blocks, output, self.moe_query).squeeze(-2)
        if messages is not None:
            inputs = self.controllers[3](inputs, messages[..., 3, :])
        hidden = self.moe(_rms(inputs, self.eps, self.moe_norm).to(dtype))
        return (output + self.moe_conv(hidden, cache, "moe")).to(dtype)


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

        self.num_visits = layer_count + (self.passes - 1) * (self.core_end - self.core_start)
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

    def forward(self, input_ids, cache=None, *, all_logits=False):
        if cache is None:
            cache = self.new_cache()
        positions = torch.arange(cache.length, cache.length + input_ids.size(1), device=input_ids.device)
        hidden = _rms(F.embedding(input_ids, self.embedding), self.eps, self.embedding_norm).to(self.embedding.dtype)
        blocks = []
        visit = 0

        for layer in self.layers[: self.core_start]:
            hidden = layer(hidden, blocks, positions, cache.layers[visit])
            visit += 1

        prelude, prelude_blocks = hidden, tuple(blocks)
        previous_blocks = ()
        previous_hidden = None
        for pass_idx in range(self.passes):
            hidden, blocks = prelude, list(prelude_blocks)
            for offset, layer in enumerate(self.layers[self.core_start : self.core_end]):
                messages = None
                if pass_idx:
                    messages = _mix(previous_blocks, previous_hidden, self.message_queries[offset])
                hidden = layer(hidden, blocks, positions, cache.layers[visit], messages)
                visit += 1
            previous_blocks = tuple(blocks[len(prelude_blocks) + 1 :])
            previous_hidden = hidden

        for layer in self.layers[self.core_end :]:
            hidden = layer(hidden, blocks, positions, cache.layers[visit])
            visit += 1

        if not all_logits:
            hidden = hidden[:, -1:]
            blocks = [block[:, -1:] for block in blocks]
        hidden = _mix(blocks, hidden, self.output_query).squeeze(-2)
        hidden = _rms(hidden, self.eps, self.output_norm).to(self.embedding.dtype)
        logits = F.linear(hidden, self.output_weight).float()
        cache.length += input_ids.size(1)
        return self.softcap * (logits / self.softcap).tanh()
