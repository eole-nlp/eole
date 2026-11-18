"""Request-owned DFlash2 drafting for the fixed-slot greedy scheduler.

The shared target verifies one constant span (anchor plus block drafts) per
cycle across all active rows; each row drafts through its own drafter ring
buffers while the drafter weights stay shared. Rejected speculative target KV
writes are discarded by rewinding the row's position counter, like the
continuous MTP runner, and committed target features draft the next block.
"""

from copy import copy
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from eole.modules.dflash2_graph import DFlash2GraphAdapter
from eole.modules.gated_delta_net import GatedDeltaNet
from eole.predict.continuous import ContinuousGreedy, GreedyRequest
from eole.predict.continuous_mtp import verified_prefix
from eole.utils.decoder_graph import DecoderGraph
from eole.utils.logging import logger
from eole.utils.request_cache import RequestCache
from eole.utils.verifier_cache import CachedVerifier


@dataclass
class DFlash2Request(GreedyRequest):
    drafter_context: torch.Tensor | None = None
    context_start: int = 0
    drafted: int = 0
    accepted: int = 0
    cycles: int = 0
    accepted_by_position: list = field(default_factory=list)


@dataclass
class DFlash2Program:
    cache: RequestCache
    specs: list
    verifier: DecoderGraph | None = None


class ContinuousDFlash2(ContinuousGreedy):
    request_type = DFlash2Request
    speculation = "dflash2"

    @torch.inference_mode()
    def __init__(
        self, predictor, slots=4, max_pending=32, prefill_chunk_size=256, compiled=False, prefill_idle_chunk_size=None
    ):
        adapter = getattr(predictor, "dflash2", None)
        if adapter is None:
            raise ValueError("Continuous DFlash2 requires a DFlash2 drafter (dflash2_model)")
        self.num_drafts = predictor.self_speculative_num_tokens
        if self.num_drafts != adapter.block_size - 1:
            raise ValueError("Continuous DFlash2 drafts the full drafter block")
        self.span = self.num_drafts + 1
        self.window = adapter.cfg.sliding_window
        self.linear_layers = [m for m in predictor.model.decoder.modules() if isinstance(m, GatedDeltaNet)]
        # The base compiled single-token step graph is unused; verification is
        # captured separately below.
        super().__init__(predictor, slots, max_pending, prefill_chunk_size, False, prefill_idle_chunk_size)
        self.decoder._dflash2_layer_ids = tuple(adapter.target_layer_ids)
        self.decoder._dflash2_window = self.window
        self._pad_cache(self.bank)
        self.row_caches = [self.bank.row(i) for i in range(slots)]
        self._extend_rope(self.capacity + self.span)
        self.draft_bank = None
        if slots > 1 and isinstance(adapter, DFlash2GraphAdapter):
            self.draft_bank = self._prepare_adapter(copy(adapter), batch=slots)
        self.adapters = []
        for slot in range(slots):
            ring = self.draft_bank.row_ring(slot) if self.draft_bank is not None else None
            self.adapters.append(self._prepare_adapter(copy(adapter), ring=ring))
        self.programs = []
        self.host_seqlens = [0] * slots
        for cache in [self.bank] + self.row_caches:
            cache.bind()
            for layer in self.linear_layers:
                layer.begin_speculation(self.span)
            self.programs.append(DFlash2Program(cache, [layer._spec_buffers for layer in self.linear_layers]))
        if compiled:
            self._prepare_verifier_graphs()
        else:
            for layer in self.linear_layers:
                layer.end_speculation()
        for _, state in self.bank.layers:
            for value in state.values():
                value.zero_()
        self.bank.metadata["cache_seqlens"].zero_()

    def _prepare_adapter(self, adapter, *, batch=1, ring=None):
        """Give this per-row copy fresh drafter state under shared weights."""
        for name in (
            "_graph",
            "_context_kv",
            "_context_positions",
            "_feature_buffer",
            "_metadata",
            "_host_metadata",
            "_has_prefix",
        ):
            if hasattr(adapter, name):
                object.__delattr__(adapter, name)
        adapter.pending = None
        if isinstance(adapter, DFlash2GraphAdapter):
            import weakref

            object.__setattr__(adapter, "_target_ref", weakref.ref(self.model))
            if not adapter.prepare_draft_graph(batch=batch, ring=ring):
                raise RuntimeError("Continuous DFlash2 drafter capture failed")
        return adapter

    def _prepare_verifier_graphs(self):
        from eole import EOLE_COMPILE_MODE, EOLE_TORCH_COMPILE

        if not EOLE_TORCH_COMPILE or EOLE_COMPILE_MODE != "0":
            raise ValueError("Compiled continuous DFlash2 requires compile mode 0")
        directory = self.decoder._verifier_cache_directory
        cached = CachedVerifier(self.decoder, directory, continuous=True) if directory else None
        logger.info("Preparing continuous DFlash2 verifiers for %d programs…", len(self.programs))
        for program in self.programs:
            program.verifier = DecoderGraph(cached, cached) if cached else self._compile(self._verify_step)
            self._bind(program)
            for layer in self.linear_layers:
                layer.begin_speculation(self.span)
            self.decoder._speculative_forward = True
            rows = program.cache.metadata["cache_seqlens"].numel()
            dtype = next(self.model.parameters()).dtype
            emb = torch.zeros(rows, self.span, self.decoder.hidden_size, device=self.next_tokens.device, dtype=dtype)
            mask = torch.zeros(rows, 1, self.span, device=emb.device, dtype=torch.bool)
            try:
                for _ in range(3):
                    program.verifier(emb, tgt_pad_mask=mask)
                    program.cache.metadata["cache_seqlens"].zero_()
                if program.verifier.graph is None:
                    raise RuntimeError("Continuous DFlash2 verifier capture failed")
            finally:
                self.decoder._speculative_forward = False
                for layer in self.linear_layers:
                    layer.end_speculation()

    @staticmethod
    def _compile(function):
        compiled = torch.compile(function, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
        return DecoderGraph(compiled, compiled)

    def _bind(self, program):
        program.cache.bind()
        for layer, spec in zip(self.linear_layers, program.specs):
            layer._spec_buffers = spec

    def _verify_step(self, emb, *, tgt_pad_mask):
        return self.decoder._forward_eager(emb, tgt_pad_mask=tgt_pad_mask, _decode_positions=self.decoder.cache_seqlens)

    def _pad_cache(self, cache):
        if cache.metadata["cache_len_tgt"] != self.capacity:
            return
        for _, state in cache.layers:
            for name in ("kcache", "vcache"):
                if name in state:
                    state[name] = F.pad(state[name], (0, 0, 0, 0, 0, self.span))
        cache.metadata["cache_len_tgt"] += self.span
        cache.metadata["left_pad_attn_mask"] = F.pad(cache.metadata["left_pad_attn_mask"], (0, self.span), value=True)
        cache.metadata["position_indices"] = torch.arange(self.capacity + self.span, device=self.next_tokens.device)[
            None
        ]

    def _extend_rope(self, size):
        rope = self.decoder.rope
        if rope.cos_sin.size(0) < size:
            old = rope.cos_sin
            cos, sin = rope.forward_1d(size)
            tail = torch.cat((cos, sin), -1)[old.size(0) :].to(old.dtype)
            rope.cos_sin = torch.cat((old, tail), 0)

    def _prefill_chunk(self, request, ids):
        emb = self.model.tgt_emb(ids, step=request.prefilled)
        mask = torch.zeros(1, 1, ids.size(1), device=emb.device, dtype=torch.bool)
        if request.cache is None:
            self.decoder._disable_cache()
            self.decoder._init_cache(emb, mask)
        else:
            request.cache.bind()
        hidden, attns = self.decoder._forward_eager(emb, tgt_pad_mask=mask)
        request.cache = RequestCache.take(self.decoder)
        self._pad_cache(request.cache)
        features = attns.get("dflash2_features")
        if features is not None:
            prior = request.drafter_context
            if prior is not None and features.size(1) < self.window:
                features = torch.cat((prior[:, -(self.window - features.size(1)) :], features), dim=1)
            request.drafter_context = features

        return hidden

    def _admit(self, request, token, events):
        slot = request.slot
        features = request.drafter_context
        request.context_start = request.prefilled - features.size(1)
        request.accepted_by_position = [0] * self.num_drafts
        self.adapters[slot].reset()
        super()._admit(request, token, events)
        self.host_seqlens[slot] = request.prefilled

    def _decode_active(self, events):
        active = dict(self.active)
        if len(active) == 1 and self.slots > 1:
            slot = next(iter(active))
            program, slots = self.programs[slot + 1], [slot]
        else:
            program, slots = self.programs[0], list(range(self.slots))
            for slot in slots:
                if slot not in active:
                    program.cache.metadata["cache_seqlens"][slot] = 0
        self._bind(program)
        mask = torch.zeros(len(slots), 1, self.span, device=self.next_tokens.device, dtype=torch.bool)
        starts = [self.host_seqlens[slot] if slot in active else 0 for slot in slots]
        if self.draft_bank is not None and len(active) > 1:
            for row, slot in enumerate(slots):
                request = active.get(slot)
                if request is None:
                    self.draft_bank.stage_row(row, None, 0, 0)
                    continue
                self.adapters[slot].stage_context(
                    self.draft_bank, row, request.drafter_context, starts[row], request.context_start
                )
            draft_tensor = self.draft_bank.replay(self.next_tokens)
        else:
            drafts = []
            for row, slot in enumerate(slots):
                request = active.get(slot)
                if request is None:
                    drafts.append(self.next_tokens.new_zeros(1, self.num_drafts))
                    continue
                adapter = self.adapters[slot]
                adapter.set_context(request.drafter_context, request.context_start)
                proposed = adapter.draft(self.model, self.next_tokens[slot : slot + 1], starts[row], self.num_drafts)
                drafts.append(torch.cat(proposed, 1))
            draft_tensor = torch.cat(drafts, 0)
        inputs = torch.cat((self.next_tokens[slots], draft_tensor), 1)
        emb = self.model.tgt_emb(inputs)
        for layer in self.linear_layers:
            layer.begin_speculation(self.span)
        self.decoder._speculative_forward = True
        try:
            hidden, attns = (program.verifier or self._verify_step)(emb, tgt_pad_mask=mask)
        finally:
            self.decoder._speculative_forward = False
            for layer in self.linear_layers:
                layer.end_speculation()
        features = attns["dflash2_features"]
        predicted = self.model.generator(hidden).log_softmax(-1).argmax(-1)
        decisions = torch.cat((predicted, draft_tensor), 1).tolist()  # one synchronization for all rows
        counts = []
        for row, slot in enumerate(slots):
            request = active.get(slot)
            if request is None:
                counts.append(0)
                continue
            targets, proposed = decisions[row][: self.span], decisions[row][self.span :]
            accepted, count = verified_prefix(targets, proposed, request.max_tokens - len(request.tokens), self.eos)
            request.cycles += 1
            request.drafted += self.num_drafts
            request.accepted += min(accepted, count)
            for index in range(min(accepted, count)):
                request.accepted_by_position[index] += 1
            counts.append(count)
            self.next_tokens[slot, 0] = targets[count - 1]
            for token in targets[:count]:
                self._emit(request, token, events)
            # Committed drafter context: this cycle's anchor row plus its
            # accepted drafts, aligned at this cycle's starting position.
            request.drafter_context = features[row : row + 1, :count]
            request.context_start = starts[row]
        GatedDeltaNet.commit_speculation_group(self.linear_layers, counts)
        committed = torch.tensor(counts, device=self.next_tokens.device, dtype=torch.int32)
        program.cache.metadata["cache_seqlens"].sub_(self.span - committed)
        for row, slot in enumerate(slots):
            if slot in active:
                self.host_seqlens[slot] += counts[row]

    def cancel(self, request_id):
        super().cancel(request_id)
        request = self.requests[request_id]
        request.drafter_context = None

    def statistics(self, request_id):
        request = self.requests.get(request_id)
        if request is None:
            return {}
        return dict(
            speculation_mode="dflash2",
            mtp_active=request.cycles > 0,
            mtp_drafted_tokens=request.drafted,
            mtp_accepted_tokens=request.accepted,
            mtp_acceptance=request.accepted / request.drafted if request.drafted else None,
            mtp_cycles=request.cycles,
            mtp_accepted_by_position=request.accepted_by_position[:],
        )

    @torch.inference_mode()
    def close(self):
        for adapter in self.adapters:
            adapter.reset()
            graph = getattr(adapter, "_graph", None)
            if graph is not None:
                graph.reset()
        self.adapters = []
        if self.draft_bank is not None:
            self.draft_bank._graph.reset()
            self.draft_bank = None
        for program in self.programs:
            if program.verifier is not None:
                program.verifier.reset()
        self.programs.clear()
        for layer in self.linear_layers:
            layer._spec_buffers = None
            for name in (
                "_row_commit_buffers",
                "_row_commit_signature",
                "_spec_replay_buffers",
                "_spec_replay_signature",
            ):
                if hasattr(layer, name):
                    delattr(layer, name)
        super().close()
