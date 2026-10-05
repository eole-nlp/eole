import unittest
from unittest.mock import patch
from types import SimpleNamespace
from eole.predict import GeneratorLM
from eole.predict.greedy_search import GreedySearchLM
from eole.predict.beam_search import BeamSearchLM
import torch


def _decode_strategy_choice(top_k, top_p, temperature, beam_size, self_speculative_decoding):
    """Build a bare GeneratorLM and return the decode_strategy class chosen
    by ``predict_batch`` for the given decoding settings, without running
    any actual decoding."""
    gen = GeneratorLM.__new__(GeneratorLM)
    gen._tgt_pad_idx = 0
    gen._tgt_bos_idx = 1
    gen._tgt_eos_idx = [2]
    gen._tgt_unk_idx = 3
    gen._tgt_start_with = 1
    gen.n_best = 1
    gen.global_scorer = SimpleNamespace(has_cov_pen=False)
    gen.min_length = 0
    gen.block_ngram_repeat = 0
    gen._exclusion_idxs = set()
    gen.replace_unk = False
    gen.temperature = temperature
    gen.top_k = top_k
    gen.top_p = top_p
    gen.beam_size = beam_size
    gen.ban_unk_token = False
    gen.add_estimator = False
    gen.dump_beam = ""
    gen.stepwise_penalty = False
    gen.ratio = -0.0
    gen.self_speculative_decoding = self_speculative_decoding
    gen.max_length = 10
    gen.estim_only = False

    captured = {}

    def fake_predict_batch_with_strategy(batch, decode_strategy, streamer=None):
        captured["decode_strategy"] = decode_strategy
        return None

    with patch.object(gen, "_predict_batch_with_strategy", side_effect=fake_predict_batch_with_strategy):
        batch = {"srclen": torch.ones(2, dtype=torch.int)}
        gen.predict_batch(batch, attn_debug=False)
    return type(captured["decode_strategy"])


class TestGeneratorLMDecodeStrategySelection(unittest.TestCase):
    """self_speculative_decoding only ever runs inside GreedySearchLM's
    decode loop, so predict_batch must route to it even when top_k/top_p
    are left at their sampling-disabled defaults (e.g. the user only set
    temperature=0 and beam_size=1), otherwise the feature is silently
    dropped to BeamSearchLM and no speedup is observed."""

    def test_speculative_decoding_forces_greedy_strategy_with_default_top_k(self):
        strategy_cls = _decode_strategy_choice(
            top_k=0, top_p=0, temperature=0.0, beam_size=1, self_speculative_decoding=True
        )
        self.assertIs(strategy_cls, GreedySearchLM)

    def test_default_decoding_without_speculative_uses_beam_search(self):
        strategy_cls = _decode_strategy_choice(
            top_k=0, top_p=0, temperature=1.0, beam_size=5, self_speculative_decoding=False
        )
        self.assertIs(strategy_cls, BeamSearchLM)

    def test_top_k_one_uses_greedy_strategy_regardless_of_speculative_flag(self):
        strategy_cls = _decode_strategy_choice(
            top_k=1, top_p=0, temperature=1.0, beam_size=1, self_speculative_decoding=False
        )
        self.assertIs(strategy_cls, GreedySearchLM)


class TestGeneratorLM(unittest.TestCase):
    def test_split_src_to_prevent_padding_target_prefix_is_none_when_equal_size(  # noqa: E501
        self,
    ):
        src = torch.randint(0, 10, (6, 5, 1))
        src_len = 5 * torch.ones(5, dtype=torch.int)
        (
            src,
            src_len,
            target_prefix,
        ) = GeneratorLM.split_src_to_prevent_padding(src, src_len)
        self.assertIsNone(target_prefix)

    def test_split_src_to_prevent_padding_target_prefix_is_ok_when_different_size(  # noqa: E501
        self,
    ):
        default_length = 5
        src = torch.randint(0, 10, (6, default_length, 1))
        src_len = default_length * torch.ones(6, dtype=torch.int)
        new_length = 4
        src_len[1] = new_length
        (
            src,
            src_len,
            target_prefix,
        ) = GeneratorLM.split_src_to_prevent_padding(src, src_len)
        self.assertTupleEqual(src.shape, (6, new_length, 1))
        self.assertTupleEqual(target_prefix.shape, (6, 1, 1))
        self.assertTrue(src_len.equal(new_length * torch.ones(6, dtype=torch.int)))
