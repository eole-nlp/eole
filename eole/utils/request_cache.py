"""Explicit ownership of ordinary decoder KV and recurrent request state.

These handles borrow no weight copies. Binding is restricted to the scheduler's
single inference worker; it is not a lock or permission for parallel forwards.
"""

from dataclasses import dataclass

import torch


@dataclass
class RequestCache:
    decoder: object
    metadata: dict
    layers: list

    @classmethod
    def take(cls, decoder):
        """Transfer initialized cache tensors out of the decoder without copying."""
        if decoder.training or decoder.cache_seqlens is None:
            raise ValueError("Request caches require initialized inference state")
        if decoder.with_cross_attn or getattr(decoder, "_hybrid_cache_storage", None) is not None:
            raise ValueError("Request caches support ordinary decoder-only storage")
        if getattr(decoder, "_speculative_forward", False):
            raise ValueError("Cannot capture speculative state as an ordinary request")
        metadata = {
            name: getattr(decoder, name)
            for name in (
                "cache_len_tgt",
                "flash",
                "position_indices",
                "left_pad_attn_mask",
                "cache_seqlens",
            )
        }
        layers = []
        for layer in decoder.transformer_layers:
            if layer.layer_type == "linear_attention":
                attention = layer.linear_attn
                names = ("conv_state", "recurrent_state")
            else:
                attention = layer.self_attn
                names = ("kcache", "vcache", "cache_leftpad")
            state = {name: getattr(attention, name) for name in names}
            if not all(isinstance(value, torch.Tensor) for value in state.values()):
                raise ValueError("Request layer caches are not initialized")
            layers.append((attention, state))
        cache = cls(decoder, metadata, layers)
        decoder._disable_cache()
        return cache

    def bind(self):
        """Bind owned tensors, never copying caches at each decode step."""
        self.decoder._reset_decode_graph()
        for name, value in self.metadata.items():
            setattr(self.decoder, name, value)
        for attention, state in self.layers:
            for name, value in state.items():
                setattr(attention, name, value)

    def row(self, slot):
        """A storage-sharing single-row view for low-concurrency decode graphs."""
        if not 0 <= slot < self.metadata["cache_seqlens"].numel():
            raise ValueError("Invalid request cache row")
        metadata = dict(self.metadata)
        for name in ("cache_seqlens", "left_pad_attn_mask"):
            metadata[name] = metadata[name][slot : slot + 1]
        layers = [
            (attention, {name: value[slot : slot + 1] for name, value in state.items()})
            for attention, state in self.layers
        ]
        return type(self)(self.decoder, metadata, layers)

    @staticmethod
    def pad_kv(tensor, extra):
        # Byte padding preserves FP8 bits and works on builds without FP8 pad.
        dtype = tensor.dtype
        if dtype == torch.float8_e4m3fn:
            tensor = tensor.view(torch.uint8)
        return torch.nn.functional.pad(tensor, (0, 0, 0, 0, 0, extra)).view(dtype)

    @torch.no_grad()
    def copy_row(self, slot, source):
        """Admit one prefilled request into a fixed-capacity decode bank.

        This one-time admission copy also overwrites stale KV/state from a
        previous slot owner. Positions and GDN state belong to the same row.
        Validate every layout before mutating any destination tensor.
        """
        if source.decoder is not self.decoder or len(source.layers) != len(self.layers):
            raise ValueError("Cannot transfer request state between different decoders")
        count = self.metadata["cache_seqlens"].size(0)
        if not 0 <= slot < count or source.metadata["cache_seqlens"].numel() != 1:
            raise ValueError("Admission requires one source request and a valid destination slot")
        if self.metadata["flash"] != source.metadata["flash"]:
            raise ValueError("Request cache backends differ")
        pairs = [
            (self.metadata[name][slot : slot + 1], source.metadata[name])
            for name in (
                "cache_seqlens",
                "left_pad_attn_mask",
            )
        ]
        for (_, target), (_, original) in zip(self.layers, source.layers):
            if target.keys() != original.keys():
                raise ValueError("Request layer cache kinds differ")
            pairs.extend((value[slot : slot + 1], original[name]) for name, value in target.items())
        for target, original in pairs:
            if target.shape != original.shape or target.dtype != original.dtype or target.device != original.device:
                raise ValueError("Request cache layouts differ; use the same capacity and dtype")
        for target, original in pairs:
            target.copy_(original)
