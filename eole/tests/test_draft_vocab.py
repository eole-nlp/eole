"""Restricted proposals must never restrict the target generator."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
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

    def test_quantized_heads_are_built_from_the_original_dense_rows(self):
        for quantization in ("int8", "int4"):
            with self.subTest(quantization=quantization), tempfile.TemporaryDirectory() as directory:
                model = DraftModel()
                original = model.generator.weight.detach().clone()
                path = Path(directory) / "ids.json"
                path.write_text(json.dumps([2, 1]))
                config = SimpleNamespace(
                    vocabulary_quantization=quantization,
                    vocabulary_quantization_group_size=64,
                    self_speculative_draft_vocab=str(path),
                )
                with patch(
                    "eole.modules.vocabulary_quantization.quantize_vocabulary_head",
                    side_effect=lambda head, bits, group: head,
                ) as quantize:
                    Inference._prepare_inference_projections(SimpleNamespace(model=model), config)
                self.assertEqual(quantize.call_count, 2)
                self.assertEqual([call.args[1:] for call in quantize.call_args_list], [(int(quantization[3:]), 64)] * 2)
                draft = model._mtp_draft_head
                torch.testing.assert_close(draft.projection.weight[:2], original[[1, 2]])
                torch.testing.assert_close(model.generator.weight, original)
                self.assertIsNone(draft.weight)
                self.assertEqual(draft(torch.tensor([[1.0, 0.0]])).item(), 1)
                self.assertIn("generator.weight", model.state_dict())
                self.assertFalse(any("_mtp_draft_head" in key for key in model.state_dict()))

    def test_padded_rows_cannot_win_over_negative_real_logits(self):
        generator = nn.Linear(128, 10, bias=False)
        generator.weight.data.fill_(-1)
        head = RestrictedDraftHead(generator, [8, 2, 5])
        with patch(
            "eole.modules.vocabulary_quantization.quantize_vocabulary_head",
            side_effect=lambda dense, bits, group: dense,
        ):
            head.quantize(4)
        self.assertEqual(head.projection.out_features, 64)
        self.assertEqual(head(torch.ones(1, 128)).item(), 2)
        self.assertEqual(head.state_dict(), {})

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_quantized_shortlist_matches_selected_dequantized_rows(self):
        from eole.modules.vocabulary_quantization import quantize_rows
        from eole.tests.test_vocabulary_quantization import dequantize

        for bits in (4, 8):
            generator = nn.Linear(256, 128, bias=False, device="cuda", dtype=torch.bfloat16)
            generator.weight.data.fill_(-0.25)
            generator.weight.data[17].fill_(0.5)
            original = generator.weight.detach().clone()
            head = RestrictedDraftHead(generator, list(reversed(range(67))))
            selected = head.weight.detach().cpu()
            packed, scales = quantize_rows(selected, bits, 128)
            reference = dequantize(packed, scales, bits, 128).cuda().to(torch.bfloat16)
            head.quantize(bits)
            x = torch.ones(2, 256, device="cuda", dtype=torch.bfloat16)
            torch.testing.assert_close(
                head.projection(x)[..., :67], torch.nn.functional.linear(x, reference), atol=0.02, rtol=0.02
            )
            self.assertTrue(head(x).eq(17).all())
            torch.testing.assert_close(generator.weight, original)
            self.assertIsNone(head.weight)
            self.assertEqual(head.state_dict(), {})

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
