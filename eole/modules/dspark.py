"""DSpark block drafter for a target with fixed layer visits."""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import skip_init


def _norm(x, norm):
    return F.rms_norm(x.float(), (x.size(-1),), norm.weight.float(), norm.eps).to(x.dtype)


def _rotate(x, cos, sin):
    a, b = x.float().chunk(2, dim=-1)
    return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1).to(x.dtype)


class DSparkAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = config.hidden_size
        self.heads, self.kv_heads, self.head_dim = (
            config.heads,
            config.kv_heads,
            config.head_dim,
        )
        self.q_proj = skip_init(nn.Linear, hidden, self.heads * self.head_dim, bias=False)
        self.k_proj = skip_init(nn.Linear, hidden, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = skip_init(nn.Linear, hidden, self.kv_heads * self.head_dim, bias=False)
        self.k_proj_ctx = skip_init(nn.Linear, hidden, self.kv_heads * self.head_dim, bias=False)
        self.v_proj_ctx = skip_init(nn.Linear, hidden, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = skip_init(nn.Linear, self.heads * self.head_dim, hidden, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_norm_ctx = nn.RMSNorm(self.head_dim, eps=config.norm_eps)

    def context_kv(self, hidden, rope):
        k = _norm(
            self.k_proj_ctx(hidden).unflatten(-1, (self.kv_heads, self.head_dim)),
            self.k_norm_ctx,
        )
        v = self.v_proj_ctx(hidden).unflatten(-1, (self.kv_heads, self.head_dim))
        return _rotate(k, *rope), v

    def forward(self, hidden, rope, cache):
        q = _norm(self.q_proj(hidden).unflatten(-1, (self.heads, self.head_dim)), self.q_norm)
        k = _norm(
            self.k_proj(hidden).unflatten(-1, (self.kv_heads, self.head_dim)),
            self.k_norm,
        )
        v = self.v_proj(hidden).unflatten(-1, (self.kv_heads, self.head_dim))
        q, k = _rotate(q, *rope), _rotate(k, *rope)
        if cache is not None:
            k, v = (torch.cat((context, value), dim=1) for context, value in zip(cache, (k, v)))
        repeat = self.heads // self.kv_heads
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2).repeat_interleave(repeat, dim=1),
            v.transpose(1, 2).repeat_interleave(repeat, dim=1),
        )
        return self.o_proj(out.transpose(1, 2).flatten(-2))


class DSparkMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden, mid = config.hidden_size, config.intermediate_size
        self.gate_proj = skip_init(nn.Linear, hidden, mid, bias=False)
        self.up_proj = skip_init(nn.Linear, hidden, mid, bias=False)
        self.down_proj = skip_init(nn.Linear, mid, hidden, bias=False)

    def forward(self, hidden):
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class DSparkLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = DSparkAttention(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.dflare_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = DSparkMLP(config)

    def forward(self, hidden, rope, cache):
        hidden = hidden + self.self_attn(_norm(hidden, self.input_layernorm), rope, cache)
        return hidden + self.mlp(_norm(hidden, self.post_attention_layernorm))


class RecurrentDraftHead(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        hidden, rank = config.hidden_size, config.markov_rank
        self.markov_w1 = skip_init(nn.Embedding, vocab_size, rank)
        self.hidden_out = skip_init(nn.Linear, rank, hidden)
        self.rnn_proj = skip_init(nn.Linear, 2 * (rank + hidden), 3 * rank)
        self.dflare_norm = nn.RMSNorm(hidden, eps=config.norm_eps)

    def propose(self, hidden, feature, anchor, vocabulary):
        rank = self.markov_w1.embedding_dim
        vocab = vocabulary.weight[: vocabulary.vocab_size]
        base = F.linear(hidden, vocab).float()
        f_rnn = feature.expand_as(hidden)
        context = F.linear(
            torch.cat((hidden, f_rnn), dim=-1).float(),
            self.rnn_proj.weight[:, 2 * rank :].float(),
            self.rnn_proj.bias.float(),
        )
        recurrent = self.rnn_proj.weight[:, : 2 * rank].float()
        state = hidden.new_zeros((1, rank))
        prev, tokens = anchor.flatten(), []
        for step in range(hidden.size(1)):
            inputs = torch.cat((state, self.markov_w1(prev).float()), dim=-1)
            raw = (context[:, step] + F.linear(inputs, recurrent)).to(hidden.dtype)
            forget, candidate, latent = raw.chunk(3, dim=-1)
            gate = forget.sigmoid()
            state = gate * state + (1 - gate) * candidate.tanh()
            bias = F.linear(self.hidden_out(latent.tanh()), vocab).float()
            prev = (base[:, step] + bias).to(hidden.dtype).argmax(dim=-1)
            tokens.append(prev[:, None])
        return tokens


class DSparkDrafter(nn.Module):
    """Use target visit inputs and final normalized hidden states as draft context."""

    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config
        self.block_size = config.block_size
        self.layers = nn.ModuleList(DSparkLayer(config) for _ in range(config.layers))
        self.fuse_weight = nn.Parameter(torch.empty(config.layers + 1, len(config.target_visits)))
        self.mask_embed = nn.Parameter(torch.empty(config.hidden_size))
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.markov_head = RecurrentDraftHead(config, vocab_size)
        self.reset()

    @classmethod
    def load(cls, model_path, target, config, token=None):
        from safetensors.torch import load_file
        from eole.models.hf_resolver import is_hf_model_id

        if is_hf_model_id(model_path):
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(model_path, "mtp.safetensors", token=token)
        else:
            path = Path(model_path) / "mtp.safetensors"
            if not path.is_file():
                raise ValueError("DSpark requires the official mtp.safetensors beside the model config")
        head = target.generator
        draft = cls(config, head.out_features)
        draft.load_state_dict(load_file(path), strict=True, assign=True)
        draft = draft.to(device=head.weight.device, dtype=head.weight.dtype).eval()
        draft.register_buffer(
            "mix_weights",
            draft.fuse_weight.detach().float().softmax(-1),
            persistent=False,
        )
        return draft

    def reset(self):
        self.cache = [None] * len(self.layers)
        self.last_feature = None

    def _rope(self, positions, device):
        cfg = self.config
        freqs = cfg.rope_theta ** (-torch.arange(0, cfg.head_dim, 2, device=device).float() / cfg.head_dim)
        phase = (positions[:, None] * freqs).unsqueeze(1)
        return phase.cos(), phase.sin()

    def set_context(self, features, start):
        targets = features.unflatten(-1, (len(self.config.target_visits), -1))
        weights = self.mix_weights
        rope = self._rope(
            torch.arange(start, start + features.size(1), device=features.device),
            features.device,
        )
        for i, layer in enumerate(self.layers):
            context = (targets.float() * weights[i, :, None]).sum(dim=-2).to(features.dtype)
            k, v = layer.self_attn.context_kv(_norm(context, layer.dflare_norm), rope)
            if self.cache[i] is not None:
                k, v = (torch.cat((old, new), dim=1) for old, new in zip(self.cache[i], (k, v)))
            if i in self.config.local_layers:
                window = self.config.sliding_window
                k, v = k[:, -window:].contiguous(), v[:, -window:].contiguous()
            self.cache[i] = k, v
        last = (targets[:, -1:].float() * weights[-1, :, None]).sum(dim=-2).to(features.dtype)
        self.last_feature = _norm(last, self.markov_head.dflare_norm)

    def draft(self, target, anchor, start, count):
        normalized_anchor = target.decoder.layer_schedule.normalize_embedding(target.tgt_emb(anchor))
        masks = self.mask_embed.expand(1, self.block_size - 1, -1)
        hidden = torch.cat((normalized_anchor, masks), dim=1)
        rope = self._rope(
            torch.arange(start, start + self.block_size, device=anchor.device),
            anchor.device,
        )
        for layer, cache in zip(self.layers, self.cache):
            hidden = layer(hidden, rope, cache)
        return self.markov_head.propose(
            _norm(hidden[:, 1 : count + 1], self.norm),
            self.last_feature,
            anchor,
            target.generator,
        )
