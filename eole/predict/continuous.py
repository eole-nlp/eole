"""Bounded, fixed-slot ordinary-greedy decoding.

One inference worker owns this runner and all model forwards. The opt-in Responses service uses this core.
Prefill yields between bounded chunks; compiled step graphs are optional.
Requests may join/leave between ticks without copying the bank on every step.
"""

from collections import deque
from dataclasses import dataclass, field
from threading import get_ident

import torch

from eole.constants import PositionEncodingType
from eole.utils.request_cache import RequestCache


@dataclass
class GreedyRequest:
    request_id: str
    prompt: torch.Tensor
    max_tokens: int
    tokens: list = field(default_factory=list)
    status: str = "queued"
    slot: int | None = None
    cache: RequestCache | None = None
    prefilled: int = 0
    prefill_chunks: list = field(default_factory=list)


class ContinuousGreedy:
    request_type = GreedyRequest
    supports_mtp = False

    @torch.inference_mode()
    def __init__(
        self, predictor, slots=4, max_pending=32, prefill_chunk_size=256, compiled=False, prefill_idle_chunk_size=None
    ):
        self.predictor = predictor
        self.model = predictor.model
        self.decoder = self.model.decoder
        self.worker = get_ident()
        self.closed = False
        self.vocab_size = self.model.tgt_emb.embeddings.num_embeddings
        if slots < 1 or max_pending < 1 or prefill_chunk_size < 1:
            raise ValueError("Slot and queue limits must be positive")
        if (predictor.self_speculative_decoding and not self.supports_mtp) or getattr(
            predictor, "dflash2", None
        ) is not None:
            raise ValueError("Continuous prototype requires speculation disabled")
        if getattr(predictor, "parallel_gpu", 1) != 1 or self.decoder.with_cross_attn:
            raise ValueError("Continuous prototype requires single-GPU decoder-only inference")
        if self.decoder.position_encoding_type != PositionEncodingType.Rotary or self.decoder.dynamic_shapes:
            raise ValueError("Continuous prototype requires rotary positions and fixed cache capacity")
        if any(
            getattr(getattr(layer, "self_attn", None), "multidimensional_rope", False)
            for layer in self.decoder.transformer_layers
        ):
            raise ValueError("Multidimensional rotary positions are not supported")
        self.prefilling = None
        self.prefill_chunk_size = prefill_chunk_size
        self.prefill_idle_chunk_size = (
            prefill_chunk_size if prefill_idle_chunk_size is None else prefill_idle_chunk_size
        )
        if not isinstance(self.prefill_idle_chunk_size, int) or self.prefill_idle_chunk_size < prefill_chunk_size:
            raise ValueError("Idle prefill chunk must be an integer at least as large as the active chunk")
        if (
            any(
                getattr(predictor, name, 0)
                for name in (
                    "min_length",
                    "presence_penalty",
                    "frequency_penalty",
                    "block_ngram_repeat",
                    "ban_unk_token",
                )
            )
            or predictor.beam_size != 1
            or predictor.n_best != 1
            or predictor.top_k != 1
            or getattr(predictor, "top_p", 0) not in (0, 1)
        ):
            raise ValueError("Continuous serving requires unconstrained batch-one greedy defaults")
        if getattr(self.decoder, "_prefill_cache", None) is not None:
            raise ValueError("Continuous prefix reuse is not implemented; set prefill_cache_size=0")
        self.pending = deque()
        self.active = {}
        self.requests = {}
        self.max_pending = max_pending
        self.slots = slots
        parameter = next(self.model.parameters())
        if parameter.device.type != "cuda":
            raise ValueError("Continuous prototype requires CUDA FlashAttention")
        emb = torch.zeros(slots, 1, self.decoder.hidden_size, device=parameter.device, dtype=parameter.dtype)
        self.decoder._disable_cache()
        self.decoder._init_cache(emb, torch.zeros(slots, 1, 1, device=emb.device, dtype=torch.bool))
        self.bank = RequestCache.take(self.decoder)
        if not self.bank.metadata["flash"]:
            raise ValueError("Continuous prototype requires FlashAttention KV-cache support")
        self.next_tokens = torch.zeros(slots, 1, device=emb.device, dtype=torch.long)
        self.eos = set(predictor._tgt_eos_idx)
        self.capacity = self.bank.metadata["cache_len_tgt"]
        self.step_graph = None
        self.row_caches = [self.bank.row(slot) for slot in range(slots)]
        self.single_graphs = [None] * slots
        if compiled:
            from eole import EOLE_TORCH_COMPILE, EOLE_COMPILE_MODE
            from eole.utils.decoder_graph import DecoderGraph

            if not EOLE_TORCH_COMPILE or EOLE_COMPILE_MODE != "0":
                raise ValueError("Compiled continuous steps require EOLE_TORCH_COMPILE=1 and mode 0")
            function = torch.compile(
                self._step,
                fullgraph=True,
                dynamic=False,
                options={
                    "triton.cudagraphs": False,
                },
            )
            self.step_graph = DecoderGraph(function, function)
            programs = [(self.bank, self.step_graph, emb)]
            if slots > 1:
                for slot, cache in enumerate(self.row_caches):
                    graph = DecoderGraph(function, function)
                    self.single_graphs[slot] = graph
                    programs.append((cache, graph, emb[:1]))
            with torch.inference_mode():
                for cache, graph, inputs in programs:
                    cache.bind()
                    mask = torch.zeros(inputs.size(0), 1, 1, device=emb.device, dtype=torch.bool)
                    for _ in range(3):
                        graph(inputs, tgt_pad_mask=mask)
                    if graph.graph is None:
                        raise RuntimeError("Continuous decode graph capture did not succeed")
                for _, state in self.bank.layers:
                    for value in state.values():
                        value.zero_()
                self.bank.metadata["cache_seqlens"].zero_()
            torch.cuda.synchronize()

    def _step(self, emb, *, tgt_pad_mask):
        return self.decoder.forward_step(emb)

    @property
    def has_work(self):
        return bool(self.active or self.pending or self.prefilling)

    def _check_worker(self):
        if get_ident() != self.worker:
            raise RuntimeError("Continuous runner must stay on its owning inference worker")

    def submit(self, request_id, prompt_ids, max_tokens):
        self._check_worker()
        if self.closed:
            raise RuntimeError("Continuous runner is closed")
        if request_id in self.requests:
            raise ValueError("Duplicate request ID")
        if len(self.pending) >= self.max_pending:
            raise ValueError("Continuous pending queue is full")
        if max_tokens < 1 or not prompt_ids or len(prompt_ids) + max_tokens > self.capacity:
            raise ValueError("Prompt/output budget exceeds cache capacity or is empty")
        if any(not isinstance(token, int) or not 0 <= token < self.vocab_size for token in prompt_ids):
            raise ValueError("Prompt contains an invalid token ID")
        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=self.next_tokens.device)
        request = self.request_type(request_id, prompt, max_tokens)
        self.requests[request_id] = request
        self.pending.append(request)
        return request

    def cancel(self, request_id):
        self._check_worker()
        request = self.requests[request_id]
        if request.status in ("queued", "prefilling", "running"):
            request.status = "cancelled"
            if request is self.prefilling:
                self.prefilling = None
                request.cache = None
                request.slot = None
            elif request.slot is not None:
                self.active.pop(request.slot)
                request.slot = None
            else:
                self.pending.remove(request)
        # Caller may now release its completed/cancelled handle with forget().

    def forget(self, request_id):
        self._check_worker()
        request = self.requests[request_id]
        if request.status in ("queued", "prefilling", "running"):
            raise ValueError("Cannot release an active request")
        del self.requests[request_id]

    def _emit(self, request, token, events):
        request.tokens.append(token)
        if token in self.eos:
            request.status = "stop"
        elif len(request.tokens) >= request.max_tokens:
            request.status = "length"
        if request.status != "running":
            self.active.pop(request.slot)
            request.slot = None
        events.append((request.request_id, token, request.status))

    @torch.inference_mode()
    def tick(self):
        self._check_worker()
        if self.closed:
            raise RuntimeError("Continuous runner is closed")
        events = []
        if self.active:
            self._decode_active(events)
        self._prefill_one(events)
        return events

    def _decode_active(self, events):
        if len(self.active) == 1 and self.slots > 1:
            slot = next(iter(self.active))
            cache, graph = self.row_caches[slot], self.single_graphs[slot]
            inputs = self.next_tokens[slot : slot + 1]
            slots = [slot]
        else:
            cache, graph = self.bank, self.step_graph
            inputs = self.next_tokens
            slots = range(self.slots)
            for slot in range(self.slots):
                if slot not in self.active:
                    self.bank.metadata["cache_seqlens"][slot] = 0
        cache.bind()
        emb = self.model.tgt_emb(inputs, step=1)
        if graph is None:
            hidden, _ = self.decoder.forward_step(emb)
        else:
            mask = torch.zeros(inputs.size(0), 1, 1, device=emb.device, dtype=torch.bool)
            hidden, _ = graph(emb, tgt_pad_mask=mask)
        predicted = self.model.generator(hidden[:, -1]).log_softmax(-1).argmax(-1)
        inputs.copy_(predicted[:, None])
        for slot, token in zip(slots, predicted.tolist()):
            if slot in self.active:
                self._emit(self.active[slot], token, events)

    def _prefill_one(self, events):
        # At most one bounded prefill chunk follows each decode round.
        if self.prefilling is None and self.pending:
            free = next((slot for slot in range(self.slots) if slot not in self.active), None)
            if free is not None:
                self.prefilling = self.pending.popleft()
                self.prefilling.slot = free
                self.prefilling.status = "prefilling"
        request = self.prefilling
        if request is not None:
            chunk = self.prefill_chunk_size if self.active else self.prefill_idle_chunk_size
            end = min(request.prefilled + chunk, request.prompt.size(1))
            ids = request.prompt[:, request.prefilled : end]
            hidden = self._prefill_chunk(request, ids)
            request.prefill_chunks.append(end - request.prefilled)
            request.prefilled = end
            if end == request.prompt.size(1):
                token = self.model.generator(hidden[:, -1]).log_softmax(-1).argmax(-1).item()
                self._admit(request, token, events)

    def _prefill_chunk(self, request, ids):
        emb = self.model.tgt_emb(ids, step=request.prefilled)
        mask = torch.zeros(1, 1, ids.size(1), device=emb.device, dtype=torch.bool)
        if request.cache is None:
            self.decoder._disable_cache()
            self.decoder._init_cache(emb, mask)
        else:
            request.cache.bind()
        hidden, _ = self.decoder._forward_eager(emb, tgt_pad_mask=mask)
        request.cache = RequestCache.take(self.decoder)
        return hidden

    def _admit(self, request, token, events):
        slot = request.slot
        self.bank.copy_row(slot, request.cache)
        request.cache = None
        self.next_tokens[slot, 0] = token
        self.active[slot] = request
        request.status = "running"
        self.prefilling = None
        self._emit(request, token, events)

    def close(self):
        self._check_worker()
        for request_id in list(self.requests):
            self.cancel(request_id)
        self.decoder._disable_cache()
        self.requests.clear()
        self.pending.clear()
        self.active.clear()
        self.bank = None
        self.row_caches = []
        for graph in self.single_graphs:
            if graph is not None:
                graph.reset()
        self.single_graphs = []
        self.prefilling = None
        if self.step_graph is not None:
            self.step_graph.reset()
        self.closed = True
