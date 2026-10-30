"""Experimental BF16 DFlash2 drafter; target verification remains in Eole.

Adapted from https://github.com/z-lab/dflash/blob/main/dflash/model.py.
Copyright (c) 2026 Z Lab. MIT license; see licenses/dflash-MIT.txt.
The prototype deliberately uses eager PyTorch SDPA and stores only target
context K/V in the drafter: speculative noise K/V never needs rollback.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F


def rms_norm(x, weight, eps):
    y = x.float()
    return (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)).to(x.dtype) * weight


class Norm(nn.Module):
    def __init__(self, size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)


def rotate(x, positions, theta):
    dim = x.shape[-1]
    inv = 1.0 / theta ** (torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
    phase = positions.float()[:, None] * inv[None, :]
    phase = torch.cat((phase, phase), -1)[None, None]
    cos, sin = phase.cos().to(x.dtype), phase.sin().to(x.dtype)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def convolve(hidden, dynamic, base, group_size):
    batch, length, size = hidden.shape
    groups = size // group_size
    blocks = hidden.reshape(batch, length, groups, group_size)
    dynamic = dynamic.reshape(batch, length, base.shape[0], groups, 1)
    output = torch.zeros_like(blocks)
    for offset in range(base.shape[0]):
        values = blocks if offset == 0 else F.pad(blocks[:, :-offset], (0, 0, 0, 0, offset, 0))
        output = output + base[offset].reshape(1, 1, groups, group_size).to(hidden.dtype) * values
        output = torch.addcmul(output, dynamic[:, :, offset], values)
    return output.reshape_as(hidden)


class DynamicConv(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.group_size = cfg.dflash_config["conv_group_size"]
        self.kernel_size = cfg.dflash_config["conv_kernel_size"]
        self.base_kernel = nn.Parameter(torch.empty(2, self.kernel_size, cfg.hidden_size))
        self.kernel_projection = nn.Linear(
            cfg.hidden_size, 2 * self.kernel_size * cfg.hidden_size // self.group_size, bias=False
        )

    def prepare(self, hidden):
        dynamic = self.kernel_projection(hidden).reshape(
            *hidden.shape[:-1], 2, self.kernel_size, hidden.shape[-1] // self.group_size
        )
        return convolve(hidden, dynamic[..., 0, :, :], self.base_kernel[0], self.group_size), dynamic[..., 1, :, :]

    def finish(self, hidden, dynamic):
        return convolve(hidden, dynamic, self.base_kernel[1], self.group_size)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        for name, size in [
            ("q_proj", cfg.num_attention_heads * cfg.head_dim),
            ("k_proj", cfg.num_key_value_heads * cfg.head_dim),
            ("v_proj", cfg.num_key_value_heads * cfg.head_dim),
        ]:
            setattr(self, name, nn.Linear(cfg.hidden_size, size, bias=cfg.attention_bias))
        self.o_proj = nn.Linear(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size, bias=cfg.attention_bias)
        self.q_norm = Norm(cfg.head_dim, cfg.rms_norm_eps)
        self.k_norm = Norm(cfg.head_dim, cfg.rms_norm_eps)
        self.cache = None

    def forward(self, hidden, context, context_positions, positions):
        cfg = self.cfg
        theta = cfg.rope_parameters["rope_theta"]

        def heads(x):
            return x.reshape(x.shape[0], x.shape[1], -1, cfg.head_dim).transpose(1, 2)

        q = rotate(self.q_norm(heads(self.q_proj(hidden))), positions, theta)
        # Context is identical across draft layers and contains only committed
        # target states. No hidden noise tokens enter this persistent cache.
        kc = rotate(self.k_norm(heads(self.k_proj(context))), context_positions, theta)
        vc = heads(self.v_proj(context))
        if self.cache is not None:
            kc = torch.cat((self.cache[0], kc), 2)
            vc = torch.cat((self.cache[1], vc), 2)
            context_positions = torch.cat((self.cache[2], context_positions))
        # Committed context is contiguous and ends immediately before the
        # anchor. Slice by known shape, avoiding a GPU boolean-index sync.
        keep = cfg.sliding_window - 1
        kc, vc, context_positions = kc[:, :, -keep:], vc[:, :, -keep:], context_positions[-keep:]
        self.cache = kc, vc, context_positions
        kn = rotate(self.k_norm(heads(self.k_proj(hidden))), positions, theta)
        vn = heads(self.v_proj(hidden))
        k, v = torch.cat((kc, kn), 2), torch.cat((vc, vn), 2)
        key_positions = torch.cat((context_positions, positions))
        mask = (positions[:, None] - key_positions[None, :]).abs() < cfg.sliding_window
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[None, None], enable_gqa=True)
        return self.o_proj(attended.transpose(1, 2).reshape(hidden.shape[0], hidden.shape[1], -1))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = Norm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = Norm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attention_conv, self.mlp_conv = DynamicConv(cfg), DynamicConv(cfg)

    def forward(self, hidden, context, context_positions, positions):
        x, dynamic = self.attention_conv.prepare(self.input_layernorm(hidden))
        hidden = hidden + self.attention_conv.finish(self.self_attn(x, context, context_positions, positions), dynamic)
        x, dynamic = self.mlp_conv.prepare(self.post_attention_layernorm(hidden))
        return hidden + self.mlp_conv.finish(self.mlp(x), dynamic)


class Selector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        rank = cfg.dflash_config["selector_rank"]
        self.top_k = cfg.dflash_config["selector_top_k"]
        self.predecessor_codebook = nn.Embedding(cfg.vocab_size, rank)
        self.successor_codebook = nn.Embedding(cfg.vocab_size, rank)
        self.hidden_projection = nn.Linear(cfg.hidden_size, rank, bias=False)

    def forward(self, hidden, logits, anchor):
        unary, candidates = logits.topk(self.top_k, dim=-1, sorted=False)
        hidden = self.hidden_projection(hidden)
        path = []
        predecessor = anchor[:, 0]
        for position in range(hidden.size(1)):
            scores = unary[:, position] + torch.einsum(
                "br,bkr->bk",
                self.predecessor_codebook(predecessor) * hidden[:, position],
                self.successor_codebook(candidates[:, position]),
            )
            predecessor = candidates[:, position].gather(-1, scores.argmax(-1, keepdim=True))[:, 0]
            path.append(predecessor[:, None])
        return path


class DFlash2Adapter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.target_layer_ids = cfg.dflash_config["target_layer_ids"]
        self.block_size = cfg.dflash_config["block_size"]
        self.layers = nn.ModuleList([Layer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.fc = nn.Linear(len(self.target_layer_ids) * cfg.hidden_size, cfg.hidden_size, bias=False)
        self.hidden_norm = Norm(cfg.hidden_size, cfg.rms_norm_eps)
        self.norm = Norm(cfg.hidden_size, cfg.rms_norm_eps)
        self.candidate_selector = Selector(cfg)
        self.pending = None

    @classmethod
    def load(cls, directory, target):
        from safetensors.torch import load_file

        cfg = SimpleNamespace(**json.loads((Path(directory) / "config.json").read_text()))
        if cfg.architectures != ["DFlash2DraftModel"] or cfg.is_causal or cfg.hidden_act != "silu":
            raise ValueError("Unsupported DFlash2 checkpoint architecture")
        if (
            cfg.num_target_layers != len(target.decoder.transformer_layers)
            or cfg.hidden_size != target.decoder.hidden_size
        ):
            raise ValueError("DFlash2 checkpoint does not match target decoder dimensions")
        if cfg.sliding_window < 2 or any(t != "sliding_attention" for t in cfg.layer_types):
            raise ValueError("Prototype requires sliding-attention drafter layers")
        ids = cfg.dflash_config["target_layer_ids"]
        if ids != sorted(set(ids)) or not ids or ids[0] < 0 or ids[-1] >= cfg.num_target_layers - 1:
            raise ValueError("DFlash2 feature taps must be unique, ordered, and precede final target normalization")
        if cfg.vocab_size != target.tgt_emb.embeddings.num_embeddings:
            raise ValueError("DFlash2 and target vocabularies must match")
        if target.tgt_emb.embeddings.weight.dtype != torch.bfloat16:
            raise ValueError("DFlash2 prototype requires BF16 target activations and embeddings")
        device = next(target.parameters()).device
        if device.type != "cuda":
            raise ValueError("DFlash2 prototype requires a CUDA target")
        with torch.device("meta"):
            model = cls(cfg)
        weights = load_file(str(Path(directory) / "model.safetensors"), device=str(device))
        for name in ("predecessor_codebook", "successor_codebook"):
            key = f"candidate_selector.{name}"
            if key in weights:
                weights[key + ".weight"] = weights.pop(key)
        model.load_state_dict(weights, strict=True, assign=True)
        return model.eval()

    def reset(self):
        self.pending = None
        for layer in self.layers:
            layer.self_attn.cache = None

    def set_context(self, features, start):
        # Old prompt positions cannot be attended by the sliding drafter.
        count = min(features.size(1), self.cfg.sliding_window)
        self.pending = features[:, -count:].clone(), start + features.size(1) - count

    def draft(self, target, anchor, start, count):
        if self.pending is None:
            raise RuntimeError("DFlash2 target context was not initialized")
        features, context_start = self.pending
        context = self.hidden_norm(self.fc(features))
        positions = torch.arange(start, start + count + 1, device=anchor.device)
        context_positions = torch.arange(context_start, context_start + features.size(1), device=anchor.device)
        ids = anchor.new_full((1, count + 1), self.cfg.dflash_config["mask_token_id"])
        ids[:, :1] = anchor
        hidden = target.tgt_emb(ids, step=start)
        for layer in self.layers:
            hidden = layer(hidden, context, context_positions, positions)
        hidden = self.norm(hidden)[:, 1:]
        logits = target.generator(hidden)
        return self.candidate_selector(hidden, logits, anchor)
