"""Distribution, penalties, and cache boundaries for one-hot speculative sampling."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from eole.predict.speculative_sampling import apply_token_penalties, sample_target_chunk
from eole.predict.generator import GeneratorLM


class TestSpeculativeSampling(unittest.TestCase):
    def test_one_hot_acceptance_preserves_target_distribution(self):
        torch.manual_seed(20)
        n = 30000
        scores = torch.tensor([[0.3, 0.7], [0.8, 0.2]]).log().expand(n, -1, -1)
        drafts = torch.zeros(n, 1, dtype=torch.long)
        sampled = sample_target_chunk(scores, drafts, torch.empty(n, 0, dtype=torch.long), 1.0, 0, 0.0)
        accepted = sampled[:, 0].eq(0)
        self.assertAlmostEqual(accepted.float().mean().item(), 0.3, delta=0.012)
        self.assertTrue(sampled[~accepted, 0].eq(1).all())
        self.assertAlmostEqual(sampled[accepted, 1].eq(0).float().mean().item(), 0.8, delta=0.018)

    def test_penalties_include_hypothetical_prefix_not_future_drafts(self):
        scores = torch.zeros(1, 3, 4)
        history = torch.tensor([[1, 1]])
        drafts = torch.tensor([[2, 1]])
        with patch("eole.predict.greedy_search.sample_with_temperature") as sampler:
            sampler.side_effect = lambda x, *args: (x.argmax(-1, keepdim=True), None)
            sample_target_chunk(scores, drafts, history, 1.0, 20, 0.95, 1.0, 0.5)
        actual = sampler.call_args.args[0]
        torch.testing.assert_close(
            actual, torch.tensor([[0.0, -2.0, 0.0, 0.0], [0.0, -2.0, -1.5, 0.0], [0.0, -2.5, -1.5, 0.0]])
        )
        self.assertTrue(scores.eq(0).all())

    def test_seeded_chunk_is_reproducible(self):
        logits = torch.randn(1, 4, 8)
        drafts = torch.tensor([[1, 2, 3]])
        history = torch.tensor([[3, 3]])
        torch.manual_seed(8)
        first = sample_target_chunk(logits, drafts, history, 0.6, 4, 0.95, 1.0, 0.2)
        torch.manual_seed(8)
        second = sample_target_chunk(logits, drafts, history, 0.6, 4, 0.95, 1.0, 0.2)
        torch.testing.assert_close(first, second)

    def test_penalties_reject_unsupported_beam_search(self):
        from eole.config.inference import InferenceConfig

        with self.assertRaisesRegex(ValueError, "beam_size=1"):
            InferenceConfig(presence_penalty=1.0, beam_size=4)
        self.assertEqual(InferenceConfig(presence_penalty=1.0, beam_size=1).presence_penalty, 1.0)

    def test_lm_penalties_exclude_unequal_prompt_tails_and_padding(self):
        from eole.predict.greedy_search import GreedySearchLM
        from eole.tests.test_greedy_search import GlobalScorerStub

        strategy = GreedySearchLM(
            0,
            1,
            2,
            3,
            1,
            1,
            2,
            GlobalScorerStub(),
            0,
            0,
            set(),
            False,
            30,
            1.0,
            1,
            0.0,
            1,
            False,
            presence_penalty=1.0,
            frequency_penalty=0.5,
        )
        # The common prompt stays outside alive_seq. Only the longer prompt's
        # tail is forced through decoding; BOS is removed by initialize().
        strategy.initialize(
            torch.tensor([[4, 5], [6, 7]]),
            torch.tensor([2, 2]),
            target_prefix=torch.tensor([[1, 4, 5], [1, 0, 0]]),
        )
        torch.testing.assert_close(strategy.alive_seq, torch.tensor([[1], [1]]))
        strategy.alive_seq = torch.tensor([[1, 4, 5, 6, 6], [1, 7, 0, 7, 8]])
        with patch("eole.predict.greedy_search.sample_with_temperature") as sampler:
            sampler.return_value = (torch.tensor([[6], [7]]), torch.zeros(2, 1))
            strategy._pick(torch.zeros(2, 10))
        expected = torch.zeros(2, 10)
        expected[0, 6] = -2.0
        expected[1, 7] = -2.0
        expected[1, 8] = -1.5
        torch.testing.assert_close(sampler.call_args.args[0], expected)

    def test_zero_penalties_leave_scores_unchanged(self):
        scores = torch.randn(2, 5)
        self.assertIs(apply_token_penalties(scores, torch.zeros(2, 0, dtype=torch.long)), scores)

    def test_zero_probability_draft_is_never_accepted(self):
        scores = torch.tensor([[[-float("inf"), 0.0], [0.0, -float("inf")]]])
        actual = sample_target_chunk(scores, torch.tensor([[0]]), torch.empty(1, 0, dtype=torch.long), 0.6, 0, 0.95)
        torch.testing.assert_close(actual[:, 0], torch.tensor([1]))

    def test_topk_one_keeps_greedy_selection(self):
        logits = torch.tensor([[[3.0, 1.0, 2.0], [0.0, 5.0, 1.0]]])
        actual = sample_target_chunk(logits, torch.tensor([[0]]), torch.empty(1, 0, dtype=torch.long), 1.0, 1, 0.0)
        torch.testing.assert_close(actual, torch.tensor([[0, 1]]))

    def test_sampled_correction_and_eos_commit_only_emitted_prefix(self):
        # Exercise the actual generator round with a sampled correction, then
        # with an accepted EOS before later matching proposals.
        for selected, expected_count, expected_commit in (
            ([4, 1, 2, 3], 1, 1),
            ([2, 1, 3, 4], 1, 1),
            ([1, 4, 2, 3], 2, 2),
            ([1, 2, 3, 4], 2, 2),
        ):
            with self.subTest(selected=selected):
                decoder = torch.nn.Module()
                decoder.cache_seqlens = torch.tensor([14], dtype=torch.int32)
                model = SimpleNamespace(
                    decoder=decoder,
                    draft_mtp_tokens=Mock(return_value=[torch.tensor([[x]]) for x in (1, 2, 3)]),
                    commit_mtp_draft=Mock(),
                    set_mtp_context=Mock(),
                )
                strategy = SimpleNamespace(
                    current_predictions=torch.tensor([0]),
                    max_length=10,
                    src_len=torch.tensor([3]),
                    target_prefix=None,
                    eos_t=torch.tensor([2]),
                    alive_seq=torch.tensor([[0]]),
                    presence_penalty=0.0,
                    frequency_penalty=0.0,
                    advance_speculative=Mock(side_effect=lambda ids, lp, n, fin: ids[:, :n]),
                )
                predictor = SimpleNamespace(
                    model=model,
                    self_speculative_num_tokens=3,
                    _mtp_profile_enabled=False,
                    top_k=20,
                    top_p=0.95,
                    temperature=0.6,
                    _speculative_drafted_tokens=0,
                    _speculative_accepted_tokens=0,
                    _speculative_drafted_by_position=[0] * 3,
                    _speculative_accepted_by_position=[0] * 3,
                    _decode_and_generate=Mock(return_value=(torch.randn(1, 4, 5), None, torch.randn(1, 4, 4))),
                )
                streamer = Mock()
                with patch(
                    "eole.predict.speculative_sampling.sample_target_chunk", return_value=torch.tensor([selected])
                ), patch("eole.predict.generator.GatedDeltaNet.commit_speculation_group") as commit:
                    step, _, position = GeneratorLM._speculative_draft_verify(
                        predictor, strategy, torch.zeros(1, 1, 4), 7, 1, streamer
                    )
                self.assertEqual(strategy.advance_speculative.call_args.args[3], selected[expected_count - 1] == 2)
                self.assertEqual(streamer.put.call_count, expected_count)
                self.assertEqual(step, 1 + expected_count)
                model.commit_mtp_draft.assert_called_once_with(expected_commit)
                commit.assert_called_once_with([], expected_commit)
                self.assertEqual(decoder.cache_seqlens.item(), 10 + expected_commit)
                self.assertEqual(position, 7 + expected_count)
                self.assertEqual(model.set_mtp_context.call_args.args[0].size(1), expected_count)
