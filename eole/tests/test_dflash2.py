"""Small CPU regressions for the experimental external drafter path."""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from eole.modules.dflash2 import Attention, DFlash2Adapter, Selector, convolve, prepare_attention, rotate, rotary_tables
from eole.tests.test_speculative_decoding import _build_model, _generator


def config():
    return SimpleNamespace(
        hidden_size=16,
        head_dim=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_bias=False,
        rms_norm_eps=1e-6,
        sliding_window=8,
        rope_parameters={"rope_theta": 10000000.0},
        vocab_size=12,
        dflash_config={"selector_rank": 4, "selector_top_k": 3},
    )


class TestDFlash2(unittest.TestCase):
    def test_split_half_rope(self):
        x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
        # At position zero RoPE must be the identity.
        torch.testing.assert_close(rotate(x, torch.tensor([0]), 10000.0), x)
        angle = torch.tensor([1.0, 0.01, 1.0, 0.01])
        expected = x * angle.cos() + torch.tensor([[[[-3.0, -4.0, 1.0, 2.0]]]]) * angle.sin()
        torch.testing.assert_close(rotate(x, torch.tensor([1]), 10000.0), expected)

    def test_shared_mask_uses_retained_absolute_positions(self):
        cfg = config()
        previous = torch.arange(6)
        context = torch.arange(6, 15)
        positions = torch.arange(15, 19)
        _, _, retained, mask = prepare_attention(cfg, context, positions, previous, torch.float32)
        torch.testing.assert_close(retained, torch.arange(8, 15))
        expected = torch.tensor([[abs(q - k) < 8 for k in range(8, 19)] for q in range(15, 19)])[None, None]
        torch.testing.assert_close(mask, expected)

    def test_shared_attention_matches_independent_cache_updates(self):
        torch.manual_seed(19)
        independent = Attention(config()).eval()
        shared = copy.deepcopy(independent)
        with torch.no_grad():
            for start, length in [(5, 5), (8, 3), (10, 2), (14, 4)]:
                context = torch.randn(1, length, 16)
                hidden = torch.randn(1, 4, 16)
                cp, qp = torch.arange(start - length, start), torch.arange(start, start + 4)
                previous = shared.cache[2] if shared.cache is not None else None
                state = prepare_attention(shared.cfg, cp, qp, previous, hidden.dtype)
                actual = shared(hidden, context, cp, qp, state)
                expected = independent(hidden, context, cp, qp)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for a, b in zip(shared.cache, independent.cache):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_drafter_builds_rotary_tables_once_per_block(self):
        cfg = config()
        cfg.num_hidden_layers, cfg.intermediate_size = 3, 32
        cfg.dflash_config.update(
            target_layer_ids=[0, 1], block_size=8, conv_group_size=4, conv_kernel_size=2, mask_token_id=11
        )
        model = DFlash2Adapter(cfg).eval()
        for name, p in model.named_parameters():
            if "base_kernel" in name:
                torch.nn.init.normal_(p, std=0.05)
        embedding = torch.nn.Embedding(12, 16)
        target = SimpleNamespace(
            tgt_emb=lambda ids, step: embedding(ids), generator=torch.nn.Linear(16, 12, bias=False)
        )
        with torch.no_grad(), patch("eole.modules.dflash2.rotary_tables", wraps=rotary_tables) as tables:
            model.set_context(torch.randn(1, 5, 32), 0)
            self.assertEqual(len(model.draft(target, torch.tensor([[4]]), 5, 7)), 7)
            self.assertEqual(tables.call_count, 1)
            model.set_context(torch.randn(1, 3, 32), 5)
            model.draft(target, torch.tensor([[4]]), 8, 7)
            self.assertEqual(tables.call_count, 2)

    def test_dynamic_convolution_does_not_read_future(self):
        torch.manual_seed(8)
        x = torch.randn(1, 5, 8)
        dynamic = torch.randn(1, 5, 2, 2)
        base = torch.randn(2, 8)
        expected = torch.zeros_like(x)
        for t in range(5):
            for d in range(8):
                for offset in range(min(t + 1, 2)):
                    expected[0, t, d] += (base[offset, d] + dynamic[0, t, offset, d // 4]) * x[0, t - offset, d]
        torch.testing.assert_close(convolve(x, dynamic, base, 4), expected)

    def test_context_cache_equals_fresh_context_and_stays_bounded(self):
        torch.manual_seed(9)
        cached = Attention(config()).eval()
        fresh = copy.deepcopy(cached)
        context = torch.randn(1, 15, 16)
        with torch.no_grad():
            cached(
                torch.randn(1, 4, 16),
                context[:, :6],
                torch.arange(6),
                torch.arange(6, 10),
            )
            noise = torch.randn(1, 4, 16)
            actual = cached(noise, context[:, 6:], torch.arange(6, 15), torch.arange(15, 19))
            expected = fresh(noise, context, torch.arange(15), torch.arange(15, 19))
        torch.testing.assert_close(actual, expected)
        self.assertLessEqual(cached.cache[0].size(2), 8)
        self.assertLess(int(cached.cache[2][-1]), 15)  # no noise tokens persisted

    def test_selector_transition_can_override_unary_argmax(self):
        selector = Selector(config())
        with torch.no_grad():
            for p in selector.parameters():
                p.zero_()
            selector.hidden_projection.weight[0, 0] = 1
            selector.predecessor_codebook.weight[2, 0] = 10
            selector.successor_codebook.weight[4, 0] = 1
        hidden = torch.zeros(1, 1, 16)
        hidden[..., 0] = 1
        logits = torch.full((1, 1, 12), -100.0)
        logits[..., 3], logits[..., 4] = 2.0, 1.0
        self.assertEqual(selector(hidden, logits, torch.tensor([[2]]))[0].item(), 4)

    def test_chunked_prefill_preserves_bounded_feature_taps(self):
        target, _ = _build_model(126, gated_delta=False)
        target.decoder._dflash2_layer_ids = (0, 1)
        target.decoder._dflash2_window = 3
        chunked = copy.deepcopy(target)
        chunked.decoder.prefill_chunk_size = 2
        prompt = torch.tensor([[4, 5, 6, 7, 8]])
        mask = torch.zeros(1, 1, 5, dtype=torch.bool)
        with torch.no_grad():
            emb = target.tgt_emb(prompt)
            target.decoder._init_cache(emb, mask)
            full, full_attn = target.decoder(emb, step=0, tgt_pad_mask=mask)
            chunked.decoder._init_cache(emb, mask)
            parts, part_attn = chunked.decoder(emb, step=0, tgt_pad_mask=mask)
        torch.testing.assert_close(parts, full)
        self.assertEqual(part_attn["dflash2_features"].shape, (1, 3, 32))
        torch.testing.assert_close(part_attn["dflash2_features"], full_attn["dflash2_features"])

    def test_external_drafter_uses_feature_taps_and_target_verification(self):
        self._check_external_drafter(False)

    def test_external_drafter_with_gated_delta_target(self):
        self._check_external_drafter(True)

    def _check_external_drafter(self, gated_delta):
        target, vocab = _build_model(123, gated_delta=gated_delta)
        plain_target = copy.deepcopy(target)
        target.decoder._dflash2_layer_ids = (0, 1)
        gen = _generator(target, vocab, True)
        contexts = []

        class WrongDrafter:
            block_size = 5

            def reset(self):
                pass

            def set_context(self, features, start):
                contexts.append((features.clone(), start))

            def draft(self, model, anchor, start, count):
                return [anchor.new_full((1, 1), 4) for _ in range(count)]

        gen.dflash2 = WrongDrafter()

        def native_mtp_must_not_run(*args, **kwargs):
            raise AssertionError("Native MTP called for external drafter")

        target.draft_mtp_tokens = native_mtp_must_not_run
        target.set_mtp_context = native_mtp_must_not_run
        target.commit_mtp_draft = native_mtp_must_not_run
        prompt = torch.tensor([[4, 5, 6, 7]])
        batch = {"srclen": torch.tensor([4]), "src": prompt, "left_pad": True}
        with torch.no_grad():
            plain = _generator(plain_target, vocab, False).predict_batch(batch, False)
            actual = gen.predict_batch(batch, False)
        torch.testing.assert_close(actual["predictions"][0][0], plain["predictions"][0][0])
        self.assertGreater(gen._speculative_drafted_tokens, 0)
        self.assertEqual(contexts[0][0].shape, (1, 4, 32))
        self.assertEqual(contexts[0][1], 0)
        self.assertIsNone(gen._dflash2_features)
