"""Fixed-buffer CUDA graph execution for the DFlash2 drafter."""

import weakref

import torch

from eole.modules.speculation.dflash2 import DFlash2Adapter, normalize_rotary, rotary_tables
from eole.triton.dflash2 import put_kv, put_positions
from eole.utils.decoder_graph import DecoderGraph


class DFlash2GraphAdapter(DFlash2Adapter):
    """Fixed-ring drafter. Graph buffers survive request resets.

    The prompt populates context K/V once, outside replay. Each subsequent
    replay appends only the committed target features, then proposes block_size - 1
    tokens. Noise K/V is temporary, so rejection needs no drafter rollback.
    """

    @classmethod
    def load(cls, directory, target):
        model = super().load(directory, target)
        object.__setattr__(model, "_target_ref", weakref.ref(target))
        return model

    def _require_target(self):
        target = self._target_ref()
        if target is None:
            raise RuntimeError("DFlash2 target is no longer available; reload the drafter with its target model")
        return target

    @torch.no_grad()
    def prepare_draft_graph(self, batch=1, ring=None):
        self._require_target()
        if hasattr(self, "_graph"):
            return self._graph.graph is not None
        if self.hidden_norm.weight.device.type != "cuda":
            raise ValueError("DFlash2 draft graphs require CUDA")
        if not 2 <= self.block_size <= self.cfg.sliding_window:
            raise ValueError("DFlash2 graph blocks must contain at least two positions and fit the sliding window")
        device = self.hidden_norm.weight.device
        dtype = self.hidden_norm.weight.dtype
        if ring is None:
            self._context_positions = torch.full((batch, self.cfg.sliding_window), -1, device=device, dtype=torch.int64)
            shape = (batch, self.cfg.num_key_value_heads, self.cfg.sliding_window, self.cfg.head_dim)
            self._context_kv = [
                (torch.zeros(shape, device=device, dtype=dtype), torch.zeros(shape, device=device, dtype=dtype))
                for _ in self.layers
            ]
        else:
            self._context_positions, self._context_kv = ring
        self._feature_buffer = torch.zeros(
            (batch, self.block_size, len(self.target_layer_ids) * self.cfg.hidden_size), device=device, dtype=dtype
        )
        # Replay inputs: [anchor token, block start, context start, valid count].
        # Keep these as device values so positions/counts can change without capture.
        self._metadata = torch.zeros((batch, 4), device=device, dtype=torch.int64)
        self._host_metadata = torch.zeros((batch, 4), dtype=torch.int64, pin_memory=True)
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
        self._context_positions.index_copy_(1, indices, cp[None])

    def _forward_block(self, features, *, tgt_pad_mask):
        # DecoderGraph stages two tensors; its second input carries integer metadata
        # here rather than a padding mask: anchor, block start, context start, count.
        control = tgt_pad_mask
        cfg = self.cfg
        target = self._require_target()
        count = self.block_size
        batch = features.size(0)
        steps = torch.arange(count, device=features.device)
        cp = control[:, 2:3] + steps
        qp = control[:, 1:2] + steps
        context = self.hidden_norm(self.fc(features))
        tables = rotary_tables(
            torch.cat((cp, qp), dim=1), cfg.head_dim, cfg.rope_parameters["rope_theta"], features.dtype
        )
        ctx_tables = tuple(t[..., :count, :] for t in tables)
        noise_tables = tuple(t[..., count:, :] for t in tables)
        put_positions[(1, batch)](self._context_positions, control, cfg.sliding_window, count)
        positions = self._context_positions
        visible = (positions >= 0) & (positions < control[:, 1:2])
        mask_ctx = visible[:, None, :] & ((qp[:, :, None] - positions[:, None, :]) < cfg.sliding_window)
        mask_noise = torch.ones((batch, count, count), device=features.device, dtype=torch.bool)
        mask = torch.cat((mask_ctx, mask_noise), dim=-1)[:, None]
        ids = torch.cat(
            (
                control[:, :1],
                torch.full(
                    (batch, count - 1), cfg.dflash_config["mask_token_id"], device=features.device, dtype=torch.int64
                ),
            ),
            dim=1,
        )
        hidden = target.tgt_emb(ids)

        def heads(x):
            return x.reshape(batch, count, -1, cfg.head_dim).transpose(1, 2)

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
            put_kv[(cfg.num_key_value_heads * count, batch)](
                ck,
                cv,
                kc,
                vc,
                control,
                *ck.stride()[:3],
                *cv.stride()[:3],
                cfg.sliding_window,
                count,
                cfg.head_dim,
                cfg.num_key_value_heads
            )
            k = torch.cat((kc, kn), 2)
            v = torch.cat((vc, vn), 2)
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
            y = a.o_proj(y.transpose(1, 2).reshape(batch, count, -1))
            hidden = residual + layer.attention_conv.finish(y, dynamic)
            hidden = layer.forward_mlp(hidden)
        hidden = self.norm(hidden)[:, 1:]
        logits = target.generator(hidden)
        proposals = self.candidate_selector(hidden, logits, control[:, :1])
        return torch.cat(proposals, dim=1)

    @torch.no_grad()
    def draft(self, target, anchor, start, count):
        if self.pending is None:
            raise RuntimeError("DFlash2 target context was not initialized")
        if not 1 <= count < self.block_size:
            raise ValueError("Invalid DFlash2 proposal count")
        if self._require_target() is not target:
            raise ValueError("DFlash2 was prepared for a different target; reload the drafter")
        features, context_start = self.pending
        self.stage_context(self, 0, features, start, context_start)
        proposed = self.replay(anchor)
        self.pending = None
        return list(proposed[:, :count].split(1, dim=1))

    def row_ring(self, slot):
        """Return one slot's storage for singleton graph capture."""
        row = slice(slot, slot + 1)
        return self._context_positions[row], [(k[row], v[row]) for k, v in self._context_kv]

    def stage_context(self, program, row, features, start, context_start):
        # The singleton adapter owns prefix admission. Its ring is shared
        # with the bank, so either graph can draft the next block.
        if not self._has_prefix:
            self._append_prefix(features, context_start)
            self._has_prefix = True
            features = None
        program.stage_row(row, features, start, context_start)

    def stage_row(self, row, features, start, context_start):
        n = features.size(1) if features is not None else 0
        if n > self.block_size:
            raise ValueError("DFlash2 committed context exceeds the replay block")
        if n:
            self._feature_buffer[row : row + 1, :n].copy_(features)
        self._host_metadata[row, 1] = start
        self._host_metadata[row, 2] = context_start
        self._host_metadata[row, 3] = n

    def replay(self, anchors):
        # The scheduler reads verification results before staging another block.
        self._metadata.copy_(self._host_metadata, non_blocking=True)
        self._metadata[:, :1].copy_(anchors)
        return self._graph(self._feature_buffer, tgt_pad_mask=self._metadata)
