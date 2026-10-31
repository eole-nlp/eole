"""Fixed-buffer CUDA graph execution for the experimental DFlash2 drafter."""

import weakref

import torch

from eole.modules.dflash2 import DFlash2Adapter, normalize_rotary, rotary_tables
from eole.triton.dflash2 import put_kv, put_positions
from eole.utils.decoder_graph import DecoderGraph


class DFlash2GraphAdapter(DFlash2Adapter):
    """Batch-one fixed-ring drafter. Graph buffers survive request resets.

    The prompt populates context K/V once, outside replay. Each subsequent
    replay appends only the committed target features, then proposes seven
    tokens. Noise K/V is temporary, so rejection needs no drafter rollback.
    """

    @classmethod
    def load(cls, directory, target):
        model = super().load(directory, target)
        object.__setattr__(model, "_target_ref", weakref.ref(target))
        return model

    @torch.no_grad()
    def prepare_draft_graph(self):
        if hasattr(self, "_graph"):
            return self._graph.graph is not None
        if self.hidden_norm.weight.device.type != "cuda":
            raise ValueError("DFlash2 draft graphs require CUDA")
        if self.block_size & (self.block_size - 1) or self.cfg.sliding_window < self.block_size:
            raise ValueError("DFlash2 graph blocks must be powers of two and fit the sliding window")
        device = self.hidden_norm.weight.device
        dtype = self.hidden_norm.weight.dtype
        self._context_positions = torch.full((self.cfg.sliding_window,), -1, device=device, dtype=torch.int64)
        self._context_kv = []
        for layer in self.layers:
            shape = (1, self.cfg.num_key_value_heads, self.cfg.sliding_window, self.cfg.head_dim)
            self._context_kv.append(
                (torch.zeros(shape, device=device, dtype=dtype), torch.zeros(shape, device=device, dtype=dtype))
            )
        self._feature_buffer = torch.zeros(
            (1, self.block_size, len(self.target_layer_ids) * self.cfg.hidden_size), device=device, dtype=dtype
        )
        # Replay inputs: [anchor token, block start, context start, valid count].
        # Keep these as device values so positions/counts can change without capture.
        self._metadata = torch.zeros(4, device=device, dtype=torch.int64)
        self._host_metadata = torch.zeros(4, dtype=torch.int64, pin_memory=True)
        self._graph = DecoderGraph(self._forward_block, self._forward_block)
        self._has_prefix = False
        # Warm twice, then capture/replay. A zero valid count leaves context empty.
        for _ in range(3):
            self._graph(self._feature_buffer, tgt_pad_mask=self._metadata)
        self.reset()
        torch.cuda.synchronize()
        return self._graph.graph is not None

    def reset(self):
        super().reset()
        if hasattr(self, "_context_positions"):
            self._context_positions.fill_(-1)
            for k, v in self._context_kv:
                k.zero_()
                v.zero_()
            self._has_prefix = False

    def _append_prefix(self, features, start):
        context = self.hidden_norm(self.fc(features))
        cfg = self.cfg
        cp = torch.arange(start, start + features.size(1), device=features.device)
        tables = rotary_tables(cp, cfg.head_dim, cfg.rope_parameters["rope_theta"], features.dtype)
        indices = cp.remainder(cfg.sliding_window)
        for layer, (kc, vc) in zip(self.layers, self._context_kv):
            a = layer.self_attn
            k, v = a.project_context(context)
            k = k.reshape(1, -1, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2)
            k = normalize_rotary(k, a.k_norm, tables)
            v = v.reshape(1, -1, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2)
            kc.index_copy_(2, indices, k)
            vc.index_copy_(2, indices, v)
        self._context_positions.index_copy_(0, indices, cp)

    def _forward_block(self, features, *, tgt_pad_mask):
        # DecoderGraph stages two tensors; its second input carries integer metadata
        # here rather than a padding mask: anchor, block start, context start, count.
        control = tgt_pad_mask
        cfg = self.cfg
        target = self._target_ref()
        count = self.block_size
        cp = control[2] + torch.arange(count, device=features.device)
        qp = control[1] + torch.arange(count, device=features.device)
        context = self.hidden_norm(self.fc(features))
        tables = rotary_tables(torch.cat((cp, qp)), cfg.head_dim, cfg.rope_parameters["rope_theta"], features.dtype)
        ctx_tables = tuple(t[..., :count, :] for t in tables)
        noise_tables = tuple(t[..., count:, :] for t in tables)
        put_positions[(1,)](self._context_positions, control, cfg.sliding_window, count)
        visible = (self._context_positions >= 0) & (self._context_positions < control[1])
        mask_ctx = visible[None, :] & ((qp[:, None] - self._context_positions[None, :]) < cfg.sliding_window)
        mask_noise = (qp[:, None] - qp[None, :]).abs() < cfg.sliding_window
        mask = torch.cat((mask_ctx, mask_noise), dim=-1)[None, None]
        ids = torch.cat(
            (
                control[:1],
                torch.full((count - 1,), cfg.dflash_config["mask_token_id"], device=features.device, dtype=torch.int64),
            )
        )[None]
        hidden = target.tgt_emb(ids)

        def heads(x):
            return x.reshape(1, count, -1, cfg.head_dim).transpose(1, 2)

        for layer, (kc, vc) in zip(self.layers, self._context_kv):
            a = layer.self_attn
            residual = hidden
            x, dynamic = layer.attention_conv.prepare(layer.input_layernorm(hidden))
            q, kn, vn = a.project_noise(x)
            ck, cv = a.project_context(context)
            q = normalize_rotary(heads(q), a.q_norm, noise_tables)
            kn = normalize_rotary(heads(kn), a.k_norm, noise_tables)
            vn = heads(vn)
            ck = normalize_rotary(heads(ck), a.k_norm, ctx_tables)
            cv = heads(cv)
            put_kv[(cfg.num_key_value_heads * count,)](
                ck, cv, kc, vc, control, *ck.stride()[:3], *cv.stride()[:3], cfg.sliding_window, count, cfg.head_dim
            )
            k = torch.cat((kc, kn), 2)
            v = torch.cat((vc, vn), 2)
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
            y = a.o_proj(y.transpose(1, 2).reshape(1, count, -1))
            hidden = residual + layer.attention_conv.finish(y, dynamic)
            hidden = layer.forward_mlp(hidden)
        hidden = self.norm(hidden)[:, 1:]
        logits = target.generator(hidden)
        proposals = self.candidate_selector(hidden, logits, control[:1].reshape(1, 1))
        return torch.cat(proposals, dim=1)

    @torch.no_grad()
    def draft(self, target, anchor, start, count):
        if self.pending is None:
            raise RuntimeError("DFlash2 target context was not initialized")
        if not 1 <= count < self.block_size:
            raise ValueError("Invalid DFlash2 proposal count")
        features, context_start = self.pending
        if not self._has_prefix:
            self._append_prefix(features, context_start)
            self._has_prefix = True
            features = features[:, :0]
            context_start = start
        n = features.size(1)
        if n > self.block_size:
            raise ValueError("DFlash2 committed context exceeds the replay block")
        self._feature_buffer.zero_()
        self._feature_buffer[:, :n].copy_(features)
        self._host_metadata[1:] = torch.tensor([start, context_start, n], device="cpu")
        self._metadata[1:].copy_(self._host_metadata[1:], non_blocking=True)
        self._metadata[:1].copy_(anchor.reshape(1))
        # Returned proposals alias replay storage and must be consumed before replay.
        proposed = self._graph(self._feature_buffer, tgt_pad_mask=self._metadata)
        self.pending = None
        return list(proposed[:, :count].split(1, dim=1))
