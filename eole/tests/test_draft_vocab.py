"""Restricted proposals must never restrict the target generator."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from eole.models.model import MTPDraftingMixin
from eole.modules.draft_vocab import RestrictedDraftHead
from eole.predict.inference import Inference


class DraftHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = SimpleNamespace(self_attn=SimpleNamespace(parallel_gpu=1))

    def forward(self, hidden, embedding, **kwargs):
        return hidden


class DraftModel(MTPDraftingMixin, nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = nn.Linear(2, 5, bias=False)
        self.generator.weight.data.copy_(torch.tensor([[0.0, 0.0], [2.0, 0.0], [1.0, 0.0], [0.0, 0.0], [10.0, 0.0]]))
        self.mtp_heads = nn.ModuleList([DraftHead(), DraftHead()])
        self.decoder = SimpleNamespace(rope=None)
        self.tgt_emb = SimpleNamespace(embeddings=nn.Embedding(5, 2))


class RestrictedDraftHeadTest(unittest.TestCase):
    def test_matches_dense_selected_rows_with_bias_and_original_ids(self):
        torch.manual_seed(42)
        generator = nn.Linear(8, 17)
        head = RestrictedDraftHead(generator, [13, 3, 8])
        hidden = torch.randn(2, 4, 8)
        ids = torch.tensor([3, 8, 13])
        expected = ids[generator(hidden).index_select(-1, ids).argmax(-1, keepdim=True)]
        self.assertTrue(torch.equal(head(hidden), expected))
        self.assertEqual(head.state_dict(), {})

    def test_ties_follow_original_vocabulary_order(self):
        generator = nn.Linear(2, 5, bias=False)
        generator.weight.data.zero_()
        head = RestrictedDraftHead(generator, [4, 2])
        self.assertEqual(head(torch.ones(1, 2)).item(), 2)

    def test_qwen_first_draft_uses_shortlist_without_a_prefix_cache(self):
        model = DraftModel()
        model.mtp_heads[0].emb_norm = nn.Identity()
        model._mtp_draft_head = RestrictedDraftHead(model.generator, [2, 1])
        drafts = model.draft_mtp_tokens(torch.tensor([[[1.0, 0.0]]]), torch.tensor([[0]]), 0)
        self.assertEqual([token.item() for token in drafts], [1])

    def test_invalid_vocabularies_are_rejected(self):
        generator = nn.Linear(2, 5)
        for ids in ([], {}, [True], [1.5], [-1], [5], [1, 1]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                RestrictedDraftHead(generator, ids)
        with self.assertRaisesRegex(ValueError, "dense"):
            RestrictedDraftHead(nn.Identity(), [0])

    def test_quantized_head_combination_is_rejected_before_conversion(self):
        for quantization in ("int8", "int4"):
            with self.subTest(quantization=quantization):
                model = DraftModel()
                original = model.generator
                config = SimpleNamespace(vocabulary_quantization=quantization, self_speculative_draft_vocab="ids.json")
                with self.assertRaisesRegex(ValueError, "vocabulary_quantization='none'"):
                    Inference._prepare_inference_projections(SimpleNamespace(model=model), config)
                self.assertIs(model.generator, original)

    def test_inference_wiring_changes_drafts_but_preserves_target(self):
        model = DraftModel()
        hidden, seed = torch.tensor([[[1.0, 0.0]]]), torch.tensor([[0]])
        weight = model.generator.weight.clone()
        self.assertEqual([x.item() for x in model.draft_mtp_tokens(hidden, seed, 0)], [4, 4])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ids.json"
            path.write_text(json.dumps([2, 1]))
            config = SimpleNamespace(fuse_gate=False, self_speculative_draft_vocab=str(path))
            Inference._prepare_inference_projections(SimpleNamespace(model=model), config)
        self.assertEqual([x.item() for x in model.draft_mtp_tokens(hidden, seed, 0)], [1, 1])
        self.assertEqual(model.generator(hidden).argmax(-1).item(), 4)
        self.assertTrue(torch.equal(model.generator.weight, weight))
        self.assertFalse(any("_mtp_draft_head" in key for key in model.state_dict()))


if __name__ == "__main__":
    unittest.main()
