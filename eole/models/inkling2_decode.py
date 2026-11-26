"""A request-local, fixed-capacity decode cache for Inkling 2 Mini."""

from functools import partial

import torch

from eole.models.inkling2 import Inkling2StaticLayerCache
from eole.utils.decoder_graph import DecoderGraph


def check_packed_decode(model):
    device = model.embedding.device
    if (
        device.type != "cuda"
        or torch.version.cuda is None
        or torch.cuda.get_device_capability(device) < (8, 9)
        or model.embedding.dtype != torch.bfloat16
    ):
        raise ValueError("Inkling packed decode requires BF16 weights on NVIDIA SM89 or newer")


def _step(model, layers, position, packed_experts, token, *, tgt_pad_mask=None):
    logits = model._forward(token, layers, position, False, packed_experts)
    position.add_(1)
    return logits


class Inkling2Decode:
    """Continue a single prefilled sequence with stable KV and convolution buffers.

    Local layers use circular buffers. Full-attention layers append into the
    requested capacity. The position advances on the device inside each replay.
    """

    def __init__(self, model, prefill, capacity, *, capture=True, packed_experts=True):
        self.length = prefill.length
        self.capacity = capacity
        device = model.embedding.device
        if (
            self.length < 1
            or capacity <= self.length
            or len(prefill.layers) != model.num_visits
            or any(layer.key is None or layer.key.size(0) != 1 for layer in prefill.layers)
        ):
            raise ValueError("Inkling decode needs one non-empty prefix and room for new tokens")
        if packed_experts:
            check_packed_decode(model)
        if capture and device.type != "cuda":
            raise ValueError("Inkling CUDA graph decode requires a CUDA device")

        self.layers = [
            Inkling2StaticLayerCache.from_prefill(
                state, min(model.layers[index].attention.window or capacity, capacity), self.length
            )
            for index, state in zip(model.visit_order, prefill.layers)
        ]
        self.position = torch.full((1,), self.length, dtype=torch.long, device=device)
        self._step = partial(_step, model, self.layers, self.position, packed_experts)
        self.graph = DecoderGraph(self._step, self._step) if capture else None

    def __call__(self, token):
        if self.length >= self.capacity:
            raise ValueError(f"Inkling decode exceeded its {self.capacity}-token cache")
        if token.shape != (1, 1) or token.dtype != torch.long or token.device != self.position.device:
            raise ValueError("Inkling decode expects a single int64 token on the model device")
        logits = self.graph(token) if self.graph is not None else self._step(token)
        self.length += 1
        return logits
