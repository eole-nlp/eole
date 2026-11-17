"""Bounded, fixed-slot ordinary-greedy decoding prototype.

One inference worker owns this runner and all model forwards. This is not wired
into HTTP endpoints yet: prefill is still synchronous, and decode is eager.
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


class ContinuousGreedy:
    def __init__(self, predictor, slots=4, max_pending=32):
        self.predictor = predictor
        self.model = predictor.model
        self.decoder = self.model.decoder
        self.worker = get_ident()
        self.closed = False
        self.vocab_size = self.model.tgt_emb.embeddings.num_embeddings
        if slots < 1 or max_pending < 1:
            raise ValueError("Slot and queue limits must be positive")
        if predictor.self_speculative_decoding or getattr(predictor, "dflash2", None) is not None:
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
        request = GreedyRequest(request_id, prompt, max_tokens)
        self.requests[request_id] = request
        self.pending.append(request)
        return request

    def cancel(self, request_id):
        self._check_worker()
        request = self.requests[request_id]
        if request.status in ("queued", "running"):
            request.status = "cancelled"
            if request.slot is not None:
                self.active.pop(request.slot)
                request.slot = None
            else:
                self.pending.remove(request)
        # Caller may now release its completed/cancelled handle with forget().

    def forget(self, request_id):
        self._check_worker()
        request = self.requests[request_id]
        if request.status in ("queued", "running"):
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
        for slot in range(self.slots):
            if slot in self.active or not self.pending:
                continue
            request = self.pending.popleft()
            # Detach previous references before _init_cache can reuse/zero them.
            self.decoder._disable_cache()
            emb = self.model.tgt_emb(request.prompt, step=0)
            mask = torch.zeros(1, 1, request.prompt.size(1), device=emb.device, dtype=torch.bool)
            self.decoder._init_cache(emb, mask)
            hidden, _ = self.decoder(emb, step=0, tgt_pad_mask=mask)
            scores = self.model.generator(hidden[:, -1])
            token = scores.log_softmax(-1).argmax(-1).item()
            source = RequestCache.take(self.decoder)
            self.bank.copy_row(slot, source)
            self.next_tokens[slot, 0] = token
            request.status, request.slot = "running", slot
            self.active[slot] = request
            self._emit(request, token, events)
        if self.active:
            self.bank.bind()
            for slot in range(self.slots):
                if slot not in self.active:
                    self.decoder.cache_seqlens[slot] = 0
            emb = self.model.tgt_emb(self.next_tokens, step=1)
            hidden, _ = self.decoder.forward_step(emb)
            scores = self.model.generator(hidden[:, -1])
            predicted = scores.log_softmax(-1).argmax(-1)
            self.next_tokens.copy_(predicted[:, None])
            for slot, token in enumerate(predicted.tolist()):
                if slot in self.active:
                    self._emit(self.active[slot], token, events)
        return events

    def close(self):
        self._check_worker()
        for request_id in list(self.requests):
            self.cancel(request_id)
        self.decoder._disable_cache()
        self.requests.clear()
        self.pending.clear()
        self.active.clear()
        self.bank = None
        self.closed = True
