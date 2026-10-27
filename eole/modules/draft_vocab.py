"""Inference-only vocabulary projection for greedy MTP proposals."""

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def _omit_draft_checkpoint(module, state_dict, prefix, local_metadata):
    # Packed projections have parameters of their own; omit the whole subtree.
    for key in list(state_dict):
        if key.startswith(prefix):
            del state_dict[key]


class RestrictedDraftHead(nn.Module):
    """Copy selected dense generator rows without changing target weights.

    Dense buffers and optional packed parameters are omitted from checkpoints.
    This head proposes tokens only; the full generator must verify proposals.
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
        self.projection = None
        self.register_state_dict_post_hook(_omit_draft_checkpoint)
        self.register_buffer("token_ids", ids, persistent=False)
        self.register_buffer("weight", generator.weight.detach().index_select(0, ids), persistent=False)
        self.register_buffer(
            "bias", None if generator.bias is None else generator.bias.detach().index_select(0, ids), persistent=False
        )

    @classmethod
    def from_file(cls, generator, path):
        return cls(generator, json.loads(Path(path).expanduser().read_text()))

    @torch.no_grad()
    def quantize(self, bits, group_size=128):
        """Pack selected rows before releasing their temporary dense storage."""
        from eole.modules.vocabulary_quantization import quantize_vocabulary_head

        count, hidden_size = self.weight.shape
        # Marlin requires multiples of 64 output rows. Padded logits are never
        # eligible proposals, even when every real logit is negative.
        padding = (-count) % 64
        dense = nn.Linear(hidden_size, count + padding, bias=self.bias is not None, device="meta")
        weight = F.pad(self.weight, (0, 0, 0, padding)) if padding else self.weight
        dense.weight = nn.Parameter(weight, requires_grad=False)
        if self.bias is not None:
            dense.bias = nn.Parameter(F.pad(self.bias, (0, padding)), requires_grad=False)
        self.projection = quantize_vocabulary_head(dense, bits, group_size)
        self.weight = None
        self.bias = None

    def forward(self, hidden):
        logits = (
            self.projection(hidden)[..., : self.token_ids.numel()]
            if self.projection is not None
            else F.linear(hidden, self.weight, self.bias)
        )
        return self.token_ids[logits.argmax(dim=-1, keepdim=True)]
