"""Prefix KV cache for chunked prefill.

Stores per-chunk state tensors so that identical prompt prefixes can be
served from cache instead of recomputed on subsequent requests.  The cache
is keyed by a rolling SHA-256 hash of the token IDs, which ensures that two
different prefixes never share a cache entry.

Each cache entry stores per-layer state without a batch dimension (one entry
per sequence):

* **Standard attention layers**: position-indexed KV slices
  ``(k_slice, v_slice)`` each of shape ``(chunk_len, heads_kv, dim_per_head)``.
* **Linear attention layers** (GatedDeltaNet): end-of-chunk accumulated states
  ``(conv_state, recurrent_state)`` whose shapes match the layer's own
  ``conv_state`` and ``recurrent_state`` tensors (without the batch dimension).
  These states are a deterministic function of the entire token prefix, so
  restoring them on a cache hit is equivalent to running the forward pass.

Usage example::

    cache = PrefillCache(max_entries=512)

    prev_key = None
    for chunk_ids in chunks:
        key = PrefillCache.compute_key(chunk_ids, prev_key)
        cached = cache.get(key)
        if cached is None:
            # compute and store
            emb_out, layer_states = run_forward(chunk_ids)
            cache.put(key, emb_out, layer_states)
        else:
            emb_out, layer_states = cached
            # restore layer_states to the model
        prev_key = key
"""

import hashlib
import threading
from collections import OrderedDict
from typing import Optional

import torch


class PrefillCache:
    """Thread-safe LRU cache for chunked prefill layer states.

    Maps a rolling hash of the token-ID prefix to the hidden-state output
    and per-layer state tensors produced when that prefix was last processed.
    This avoids recomputing expensive attention (and linear-attention recurrent
    state updates) for repeated prompt prefixes (e.g. the same system prompt
    shared across many requests).

    One cache entry is stored **per sequence** (without a batch dimension);
    for batch size > 1, the caller creates B separate entries per chunk.

    All tensors are stored on CPU to keep GPU memory free for active
    inference; the caller must move them back to the correct device/dtype
    before use.

    Args:
        max_entries (int): Maximum number of chunk-prefix entries to keep.
            Older entries are evicted in LRU order when the cache is full.
    """

    def __init__(self, max_entries: int = 512):
        self._max_entries = max_entries
        self._store: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_key(token_ids: torch.Tensor, prev_key: Optional[bytes] = None) -> bytes:
        """Compute a rolling SHA-256 key for a chunk of token IDs.

        Chaining via ``prev_key`` ensures that two chunks with identical
        token content but different preceding prefixes get different keys.

        Args:
            token_ids: 1-D integer tensor ``(chunk_len,)`` of token IDs.
                The tensor is moved to CPU internally if necessary.
            prev_key: Raw-bytes key returned by the previous chunk call,
                or ``None`` for the very first chunk.

        Returns:
            bytes: 32-byte SHA-256 digest usable as a dict key.
        """
        h = hashlib.sha256()
        if prev_key is not None:
            h.update(prev_key)
        h.update(token_ids.cpu().numpy().tobytes())
        return h.digest()

    # ------------------------------------------------------------------
    # Cache operations
    # ------------------------------------------------------------------

    def get(self, key: bytes):
        """Return ``(emb_out, layer_states)`` for *key*, or ``None`` if absent.

        ``emb_out`` is a CPU tensor of shape ``(chunk_len, hidden_size)``.

        ``layer_states`` is a list (one entry per decoder layer).  Each element
        is a pair of CPU tensors whose meaning depends on the layer type:

        * Standard attention: ``(k_slice, v_slice)`` each of shape
          ``(chunk_len, heads_kv, dim_per_head)``.
        * Linear attention: ``(conv_state, recurrent_state)`` with shapes
          matching the layer's own state tensors (without the batch dimension).

        The caller is responsible for moving tensors back to the correct
        device and dtype before use.
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is not None:
                self._store.move_to_end(key)  # mark as recently used
            return entry

    def put(self, key: bytes, emb_out: torch.Tensor, layer_states: list) -> None:
        """Insert or update a cache entry.

        Args:
            key: Key produced by :meth:`compute_key`.
            emb_out: Hidden-state output for the chunk of shape
                ``(chunk_len, hidden_size)``.  Stored on CPU.
            layer_states: Per-layer list.  Each element is a pair of tensors:

                * Standard attention: ``(k_slice, v_slice)`` with shape
                  ``(chunk_len, heads_kv, dim_per_head)``.
                * Linear attention: ``(conv_state, recurrent_state)`` with
                  shapes matching the layer's own state tensors (no batch dim).

                Both tensors in each pair are detached and stored on CPU.
        """
        cpu_emb = emb_out.detach().cpu()
        cpu_states = []
        for item in layer_states:
            if item is not None:
                a, b = item
                cpu_states.append((a.detach().cpu(), b.detach().cpu()))
            else:
                cpu_states.append(None)
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key] = (cpu_emb, cpu_states)
                return
            if len(self._store) >= self._max_entries:
                self._store.popitem(last=False)  # evict LRU entry
            self._store[key] = (cpu_emb, cpu_states)

    def clear(self) -> None:
        """Remove all cached entries."""
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)
