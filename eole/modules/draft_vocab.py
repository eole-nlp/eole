"""Inference-only vocabulary projection for greedy MTP proposals."""

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class RestrictedDraftHead(nn.Module):
    """Copy selected dense generator rows without changing target weights.

    Buffers are request-independent but omitted from checkpoints. This head
    proposes tokens only; the original generator must still verify proposals.
    """

    def __init__(self, generator, token_ids):
        super().__init__()
        if type(generator) is not nn.Linear:
            raise ValueError("Restricted MTP vocabulary requires a dense torch.nn.Linear generator")
        if not isinstance(token_ids, list) or not token_ids:
            raise ValueError("Draft vocabulary must be a nonempty JSON list of token IDs")
        if any(type(token) is not int or not 0 <= token < generator.out_features for token in token_ids):
            raise ValueError("Draft vocabulary IDs must be integers within the generator vocabulary")
        if len(set(token_ids)) != len(token_ids):
            raise ValueError("Draft vocabulary must not contain duplicate token IDs")
        # Preserve full-vocabulary argmax tie ordering, regardless of file order.
        ids = torch.tensor(sorted(token_ids), device=generator.weight.device, dtype=torch.long)
        self.register_buffer("token_ids", ids, persistent=False)
        self.register_buffer("weight", generator.weight.detach().index_select(0, ids), persistent=False)
        self.register_buffer(
            "bias", None if generator.bias is None else generator.bias.detach().index_select(0, ids), persistent=False
        )

    @classmethod
    def from_file(cls, generator, path):
        return cls(generator, json.loads(Path(path).expanduser().read_text()))

    def forward(self, hidden):
        logits = F.linear(hidden, self.weight, self.bias)
        return self.token_ids[logits.argmax(dim=-1, keepdim=True)]
