"""Prefill preparation must not affect the request's state or RNG stream."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from eole.predict.generator import GeneratorLM


class TestPrefillWarmup(unittest.TestCase):
    def make_generator(self):
        gen = GeneratorLM.__new__(GeneratorLM)
        gen.max_length = 32
        gen.beam_size = 1
        gen.model = SimpleNamespace(
            decoder=SimpleNamespace(_prefill_cache=object(), _disable_cache=Mock()),
            mtp_heads=[SimpleNamespace(emb_norm=object())],
            init_mtp_cache=Mock(),
            clear_mtp_cache=Mock(),
        )
        gen._decode_and_generate = Mock(return_value=(None, None, None, torch.ones(1, 3, 4)))
        return gen

    def test_prefill_and_mtp_prefix_are_warmed_once_per_batch_layout(self):
        gen = self.make_generator()
        src = torch.ones(1, 3, dtype=torch.long)
        lengths = torch.tensor([3])
        prefix_cache = gen.model.decoder._prefill_cache

        def decode(*args, **kwargs):
            self.assertIsNone(gen.model.decoder._prefill_cache)
            torch.rand(3)
            return None, None, None, torch.ones(1, 3, 4)

        gen._decode_and_generate.side_effect = decode
        rng = torch.get_rng_state().clone()
        gen._warmup_prefill(src, lengths, speculative=True)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        gen.model.init_mtp_cache.assert_called_once()
        gen.model.clear_mtp_cache.assert_called_once()
        gen.model.decoder._disable_cache.assert_called_once()
        self.assertIs(gen.model.decoder._prefill_cache, prefix_cache)
        gen._warmup_prefill(src, lengths, speculative=True)
        self.assertEqual(gen._decode_and_generate.call_count, 1)
        gen._warmup_prefill(src[:, :2], lengths - 1, speculative=True)
        self.assertEqual(gen._decode_and_generate.call_count, 1)
        gen._warmup_prefill(src.expand(2, -1), lengths.expand(2), speculative=True)
        self.assertEqual(gen._decode_and_generate.call_count, 2)

    def test_normal_decoding_does_not_prime_mtp(self):
        gen = self.make_generator()
        gen._warmup_prefill(torch.ones(1, 3), torch.tensor([3]), speculative=False)
        gen.model.init_mtp_cache.assert_not_called()
        gen.model.clear_mtp_cache.assert_not_called()
        gen.model.decoder._disable_cache.assert_called_once()

    def test_selection_warmup_does_not_advance_the_real_strategy(self):
        class Strategy:
            calls = 0
            return_attention = True

            def __init__(self):
                self.tokens = torch.zeros(1, 1, dtype=torch.long)

            def advance(self, scores, attn):
                Strategy.calls += 1
                self.tokens.add_(1)
                torch.rand(1)

        gen = self.make_generator()
        strategy = Strategy()
        rng = torch.get_rng_state().clone()
        gen._warmup_prefill(torch.ones(1, 3), torch.tensor([3]), speculative=False, decode_strategy=strategy)
        self.assertEqual(Strategy.calls, 1)
        self.assertTrue(gen._decode_and_generate.call_args.kwargs["return_attn"])
        self.assertEqual(strategy.tokens.item(), 0)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_failed_preparation_cleans_up_and_can_be_retried(self):
        gen = self.make_generator()
        prefix_cache = gen.model.decoder._prefill_cache
        gen._decode_and_generate.side_effect = RuntimeError("prefill failed")
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "prefill failed"):
                gen._warmup_prefill(torch.ones(1, 3), torch.tensor([3]), speculative=True)
        self.assertEqual(gen._decode_and_generate.call_count, 2)
        self.assertEqual(gen.model.clear_mtp_cache.call_count, 2)
        self.assertEqual(gen.model.decoder._disable_cache.call_count, 2)
        self.assertIs(gen.model.decoder._prefill_cache, prefix_cache)

    def test_cache_cleanup_releases_prompt_sized_attention_gates(self):
        from eole.tests.test_speculative_decoding import _build_model

        model, _ = _build_model(12, qwen_mtp=True)
        attention = model.decoder.transformer_layers[0].self_attn
        attention._attn_gate = torch.ones(1, 19, 16)
        model.decoder._disable_cache()
        self.assertFalse(hasattr(attention, "_attn_gate"))

        attention = model.mtp_heads[0].layer.self_attn
        model._mtp_cache_attention = attention
        model._mtp_cache_old = None, None, None
        attention._attn_gate = torch.ones(1, 19, 16)
        model.clear_mtp_cache()
        self.assertFalse(hasattr(attention, "_attn_gate"))


if __name__ == "__main__":
    unittest.main()
