"""Request-owned recurrent Qwen MTP for the fixed-slot greedy scheduler.

The target verifies fixed-width chunks. Each row commits only its own accepted
prefix; logical lengths hide speculative KV writes beyond that prefix.
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from eole.predict.continuous import ContinuousGreedy, GreedyRequest
from eole.modules.gated_delta_net import GatedDeltaNet
from eole.utils.decoder_graph import DecoderGraph
from eole.utils.logging import logger
from eole.utils.request_cache import RequestCache
from eole.utils.verifier_cache import CachedVerifier


def verified_prefix(targets, proposals, remaining, eos):
    """Return matching drafts and the emitted target prefix, including correction."""
    accepted = 0
    while accepted < len(proposals) and targets[accepted] == proposals[accepted]:
        accepted += 1
    count = min(accepted + 1, remaining)
    for index, token in enumerate(targets[:count]):
        if token in eos:
            return accepted, index + 1
    return accepted, count


@dataclass
class MTPState:
    keys: torch.Tensor
    values: torch.Tensor
    leftpad: torch.Tensor
    lengths: torch.Tensor
    hidden: torch.Tensor
    tokens: torch.Tensor
    counts: torch.Tensor

    def row(self, slot):
        return type(self)(*(value[slot : slot + 1] for value in self.__dict__.values()))

    def bind(self, attention):
        attention.kcache, attention.vcache, attention.cache_leftpad = self.keys, self.values, self.leftpad


@dataclass
class MTPRequest(GreedyRequest):
    mtp: MTPState | None = None
    last_hidden: torch.Tensor | None = None
    drafted: int = 0
    accepted: int = 0
    cycles: int = 0
    accepted_by_position: list = field(default_factory=list)


@dataclass
class MTPProgram:
    cache: RequestCache
    mtp: MTPState
    specs: list
    verifier: DecoderGraph | None = None
    drafter: DecoderGraph | None = None


class ContinuousMTP(ContinuousGreedy):
    request_type = MTPRequest
    supports_mtp = True
    speculation = "mtp"

    @torch.inference_mode()
    def __init__(
        self,
        predictor,
        slots=4,
        max_pending=32,
        prefill_chunk_size=256,
        compiled=False,
        prefill_idle_chunk_size=None,
        *,
        prefix_cache_entries=0
    ):
        heads = getattr(predictor.model, "mtp_heads", [])
        if len(heads) != 1 or getattr(heads[0], "emb_norm", None) is None:
            raise ValueError("Continuous MTP requires one recurrent Qwen MTP head")
        self.num_drafts = getattr(predictor, "self_speculative_num_tokens", 3)
        if not isinstance(self.num_drafts, int) or self.num_drafts < 1:
            raise ValueError("MTP draft count must be a positive integer")
        self.span = self.num_drafts + 1
        self.head = heads[0]
        self.attention = self.head.layer.self_attn
        super().__init__(
            predictor,
            slots,
            max_pending,
            prefill_chunk_size,
            False,
            prefill_idle_chunk_size,
            prefix_cache_entries=prefix_cache_entries,
        )
        if (
            self.decoder.rope_local is not None
            or getattr(self.decoder.rope, "mode", "1d") != "1d"
            or self.span > self.capacity
        ):
            raise ValueError("Continuous MTP requires global 1-D RoPE and a cache larger than its verifier span")
        # Keep the advertised context budget intact. Extra physical KV slots are
        # only speculative padding and are never committed past a request's budget.
        self._pad_cache(self.bank)
        self.row_caches = [self.bank.row(i) for i in range(slots)]
        self._extend_rope(self.capacity + self.span)
        self.mtp = self._new_state(slots)
        self.linear_layers = [m for m in self.decoder.modules() if isinstance(m, GatedDeltaNet)]
        self.programs = []
        directory = self.decoder._verifier_cache_directory
        cached = CachedVerifier(self.decoder, directory, continuous=True) if compiled and directory else None
        for cache, mtp in [(self.bank, self.mtp)] + [
            (self.row_caches[i], self.mtp.row(i)) for i in range(slots) if slots > 1
        ]:
            cache.bind()
            for layer in self.linear_layers:
                layer.begin_speculation(self.span)
            specs = [layer._spec_buffers for layer in self.linear_layers]
            program = MTPProgram(cache, mtp, specs)
            if compiled:
                from eole import EOLE_TORCH_COMPILE, EOLE_COMPILE_MODE

                if not EOLE_TORCH_COMPILE or EOLE_COMPILE_MODE != "0":
                    raise ValueError("Compiled continuous MTP requires compile mode 0")
                logger.info("Preparing continuous MTP graphs for %d decode rows…", mtp.lengths.numel())
                program.verifier = DecoderGraph(cached, cached) if cached else self._compile(self._verify_step)
                program.drafter = self._compile(self._draft_step)
                self._bind(program)
                self.decoder._speculative_forward = True
                emb = torch.zeros(
                    mtp.lengths.numel(),
                    self.span,
                    self.decoder.hidden_size,
                    device=self.next_tokens.device,
                    dtype=mtp.hidden.dtype,
                )
                mask = torch.zeros(emb.size(0), 1, self.span, device=emb.device, dtype=torch.bool)
                try:
                    for _ in range(3):
                        program.verifier(emb, tgt_pad_mask=mask)
                        cache.metadata["cache_seqlens"].zero_()
                    for _ in range(3):
                        program.drafter(mtp.hidden, tgt_pad_mask=mask)
                    if program.verifier.graph is None or program.drafter.graph is None:
                        raise RuntimeError("Continuous MTP graph capture failed")
                    logger.info("Continuous MTP verifier and drafter captured for %d rows", mtp.lengths.numel())
                finally:
                    self.decoder._speculative_forward = False
                    for layer in self.linear_layers:
                        layer.end_speculation()
            else:
                for layer in self.linear_layers:
                    layer.end_speculation()
            self.programs.append(program)
        for _, state in self.bank.layers:
            for value in state.values():
                value.zero_()
        self.bank.metadata["cache_seqlens"].zero_()
        self.mtp.keys.zero_()
        self.mtp.values.zero_()

    @staticmethod
    def _compile(function):
        compiled = torch.compile(function, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
        return DecoderGraph(compiled, compiled)

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
            rope.cos_sin = torch.cat((old, tail), 0)  # preserve all existing position values exactly

    def _new_state(self, batch):
        template = self.next_tokens
        dtype = next(self.model.parameters()).dtype
        shape = (batch, self.capacity + self.span, self.attention.heads_kv, self.attention.dim_per_head)
        keys = torch.zeros(shape, device=template.device, dtype=dtype)
        return MTPState(
            keys,
            torch.zeros_like(keys),
            torch.zeros(batch, device=template.device, dtype=torch.int32),
            torch.zeros(batch, device=template.device, dtype=torch.int32),
            torch.zeros(batch, self.span, self.decoder.hidden_size, device=template.device, dtype=dtype),
            torch.zeros(batch, self.span, device=template.device, dtype=torch.long),
            torch.zeros(batch, device=template.device, dtype=torch.int32),
        )

    def _bind(self, program):
        program.cache.bind()
        program.mtp.bind(self.attention)
        self._draft_tokens = program.mtp.tokens
        self._draft_counts = program.mtp.counts
        self._draft_lengths = program.mtp.lengths
        for layer, spec in zip(self.linear_layers, program.specs):
            layer._spec_buffers = spec

    def _draft_step(self, hidden, *, tgt_pad_mask):
        lengths, counts = self._draft_lengths, self._draft_counts
        positions = lengths[:, None] + torch.arange(self.span, device=hidden.device)
        output = self.head(
            hidden,
            self.model.tgt_emb.embeddings(self._draft_tokens),
            position_embeddings=self.decoder.rope.cos_sin[positions],
            cache_seqlens=lengths,
        )
        rows = torch.arange(hidden.size(0), device=hidden.device)
        output = output[rows, (counts - 1).clamp_min(0)][:, None]
        keep = lengths + counts
        proposals = [self.model.generator(output[:, 0]).argmax(-1)]
        for index in range(self.num_drafts - 1):
            position = keep + index
            output = self.head(
                output,
                self.model.tgt_emb.embeddings(proposals[-1][:, None]),
                position_embeddings=self.decoder.rope.cos_sin[position[:, None]],
                cache_seqlens=position,
            )
            proposals.append(self.model.generator(output[:, 0]).argmax(-1))
        return torch.stack(proposals, 1), keep

    def _verify_step(self, emb, *, tgt_pad_mask):
        return self.decoder._forward_eager(emb, tgt_pad_mask=tgt_pad_mask, _decode_positions=self.decoder.cache_seqlens)

    def _prefill_chunk(self, request, ids):
        hidden = super()._prefill_chunk(request, ids)
        self._pad_cache(request.cache)
        if request.mtp is None:
            request.mtp = self._new_state(1)
        paired = hidden[:, :-1] if request.last_hidden is None else torch.cat((request.last_hidden, hidden[:, :-1]), 1)
        tokens = ids[:, 1:] if request.last_hidden is None else ids
        if tokens.size(1):
            request.mtp.bind(self.attention)
            positions = request.mtp.lengths[:, None] + torch.arange(tokens.size(1), device=ids.device)
            self.head(
                paired,
                self.model.tgt_emb.embeddings(tokens),
                position_embeddings=self.decoder.rope.cos_sin[positions],
                cache_seqlens=request.mtp.lengths,
            )
            request.mtp.lengths.add_(tokens.size(1))
        request.last_hidden = hidden[:, -1:].clone()
        return hidden

    def _save_prefill_state(self, request):
        return request.mtp, request.last_hidden

    def _restore_prefill_state(self, request, state):
        request.mtp, request.last_hidden = state

    def _admit(self, request, token, events):
        slot = request.slot
        self.mtp.keys[slot : slot + 1].copy_(request.mtp.keys)
        self.mtp.values[slot : slot + 1].copy_(request.mtp.values)
        self.mtp.lengths[slot : slot + 1].copy_(request.mtp.lengths)
        self.mtp.hidden[slot, 0].copy_(request.last_hidden[0, 0])
        self.mtp.tokens[slot, 0] = token
        self.mtp.counts[slot] = 1
        request.accepted_by_position = [0] * self.num_drafts
        super()._admit(request, token, events)
        request.mtp, request.last_hidden = None, None

    def _decode_active(self, events):
        active = dict(self.active)
        if len(active) == 1 and self.slots > 1:
            slot = next(iter(active))
            program, slots = self.programs[slot + 1], [slot]
        else:
            program, slots = self.programs[0], list(range(self.slots))
            for slot in slots:
                if slot not in active:
                    self.bank.metadata["cache_seqlens"][slot] = 0
                    self.mtp.lengths[slot] = 0
                    self.mtp.counts[slot] = 0
        self._bind(program)
        state = program.mtp
        mask = torch.zeros(state.lengths.numel(), 1, self.span, device=state.keys.device, dtype=torch.bool)
        drafts, keep = (program.drafter or self._draft_step)(state.hidden, tgt_pad_mask=mask)
        state.lengths.copy_(keep)
        inputs = torch.cat((self.next_tokens[slots], drafts), 1)
        emb = self.model.tgt_emb(inputs)
        for layer in self.linear_layers:
            layer.begin_speculation(self.span)
        self.decoder._speculative_forward = True
        try:
            hidden, _ = (program.verifier or self._verify_step)(emb, tgt_pad_mask=mask)
        finally:
            self.decoder._speculative_forward = False
            for layer in self.linear_layers:
                layer.end_speculation()
        predicted = self.model.generator(hidden).log_softmax(-1).argmax(-1)
        decisions = torch.cat((predicted, drafts), 1).tolist()  # one synchronization for all rows
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
        GatedDeltaNet.commit_speculation_group(self.linear_layers, counts)
        committed = torch.tensor(counts, device=state.keys.device, dtype=torch.int32)
        program.cache.metadata["cache_seqlens"].sub_(self.span - committed)
        state.hidden.copy_(hidden)
        state.tokens.copy_(predicted)
        state.counts.copy_(committed)
        for row, slot in enumerate(slots):
            if slot not in self.active:
                state.counts[row] = 0

    def cancel(self, request_id):
        super().cancel(request_id)
        self.requests[request_id].mtp = None
        self.requests[request_id].last_hidden = None

    def statistics(self, request_id):
        request = self.requests.get(request_id)
        if request is None:
            return {}
        return dict(
            **super().statistics(request_id),
            speculation_mode="mtp",
            mtp_active=request.cycles > 0,
            mtp_drafted_tokens=request.drafted,
            mtp_accepted_tokens=request.accepted,
            mtp_acceptance=request.accepted / request.drafted if request.drafted else None,
            mtp_cycles=request.cycles,
            mtp_accepted_by_position=request.accepted_by_position[:],
        )

    def close(self):
        self.decoder._speculative_forward = False
        for layer in self.linear_layers:
            layer.end_speculation()
            layer._spec_buffers = None
            for name in (
                "_row_commit_buffers",
                "_row_commit_signature",
                "_spec_replay_buffers",
                "_spec_replay_signature",
            ):
                if hasattr(layer, name):
                    delattr(layer, name)
        for program in self.programs:
            for graph in (program.drafter, program.verifier):
                if graph is not None:
                    graph.reset()
        self.programs.clear()
        self.attention.kcache, self.attention.vcache, self.attention.cache_leftpad = None, None, None
        self.mtp = None
        super().close()
