import unittest
from eole.predict.greedy_search import GreedySearch, sample_topk, sample_topp, sample_with_temperature

import torch


class TestSamplingFilters(unittest.TestCase):
    def test_filters_assign_zero_probability_to_excluded_tokens(self):
        for dtype in (torch.float16, torch.float32, torch.float64):
            logits = torch.tensor([[-20000, -20016, -20032], [-20032, -20000, -20016]], dtype=dtype)
            for filter_fn, threshold, excluded in (
                (sample_topk, 2, [[False, False, True], [True, False, False]]),
                (sample_topp, 0.9, [[False, True, True], [True, False, True]]),
            ):
                with self.subTest(dtype=dtype, filter=filter_fn.__name__):
                    filtered = filter_fn(logits, threshold)
                    mask = torch.tensor(excluded)
                    self.assertTrue(torch.isneginf(filtered[mask]).all())
                    self.assertTrue(torch.equal(filtered[~mask], logits[~mask]))
                    self.assertTrue((filtered.softmax(-1)[mask] == 0).all())
                    self.assertEqual(filtered.dtype, dtype)

    def test_filters_do_not_restore_forbidden_tokens(self):
        logits = torch.tensor([[-20000.0, -20016.0, -float("inf")]])
        for filter_fn, threshold in ((sample_topk, 2), (sample_topp, 0.9)):
            with self.subTest(filter=filter_fn.__name__):
                filtered = filter_fn(logits, threshold)
                self.assertTrue(torch.isneginf(filtered[0, 2]))
                self.assertEqual(filtered.softmax(-1)[0, 2].item(), 0)

    def test_temperature_sampling_never_selects_forbidden_token(self):
        logits = torch.tensor([[-2.0, -3.0, -float("inf")]]).repeat(8, 1)
        for top_k, top_p in ((2, 0.0), (0, 0.9), (2, 0.9)):
            with self.subTest(top_k=top_k, top_p=top_p):
                ids, scores = sample_with_temperature(logits, temperature=0.0001, top_k=top_k, top_p=top_p)
                self.assertTrue((ids != 2).all())
                self.assertTrue(torch.isfinite(scores).all())


class GlobalScorerStub(object):
    alpha = 0
    beta = 0

    def __init__(self):
        self.length_penalty = lambda x, alpha: 1.0
        self.cov_penalty = lambda cov, beta: torch.zeros((1, cov.shape[-2]), device=cov.device, dtype=torch.float)
        self.has_cov_pen = False
        self.has_len_pen = False

    def update_global_state(self, beam):
        pass

    def score(self, beam, scores):
        return scores


class TestGreedySearch(unittest.TestCase):
    BATCH_SZ = 3
    INP_SEQ_LEN = 53
    DEAD_SCORE = -1e20

    BLOCKED_SCORE = -10e20

    def test_doesnt_predict_eos_if_shorter_than_min_len(self):
        # batch 0 will always predict EOS. The other batches will predict
        # non-eos scores.
        for batch_sz in [1, 3]:
            n_words = 100
            _non_eos_idxs = [47]
            valid_score_dist = torch.log_softmax(torch.tensor([6.0, 5.0]), dim=0)
            min_length = 5
            eos_idx = 2
            lengths = torch.randint(0, 30, (batch_sz,))
            samp = GreedySearch(
                0,
                1,
                2,
                3,
                1,
                1,
                batch_sz,
                GlobalScorerStub(),
                min_length,
                False,
                set(),
                False,
                30,
                1.0,
                1,
                0,
                1,
                False,
            )
            samp.initialize(torch.zeros((1, 1)), lengths)
            all_attns = []
            for i in range(min_length + 4):
                word_probs = torch.full((batch_sz, n_words), -float("inf"))
                # "best" prediction is eos - that should be blocked
                word_probs[0, eos_idx] = valid_score_dist[0]
                # include at least one prediction OTHER than EOS
                # that is greater than -1e20
                word_probs[0, _non_eos_idxs[0]] = valid_score_dist[1]
                word_probs[1:, _non_eos_idxs[0] + i] = 0

                attns = torch.randn(1, batch_sz, 53)
                all_attns.append(attns)
                samp.advance(word_probs, attns)
                if i < min_length:
                    self.assertTrue(samp.topk_scores[0].allclose(valid_score_dist[1]))
                    self.assertTrue(samp.topk_scores[1:].eq(0).all())
                elif i == min_length:
                    # now batch 0 has ended and no others have
                    self.assertTrue(all(samp.is_finished_list[0][:]))
                    self.assertTrue(all(all([not x for x in sublist]) for sublist in samp.is_finished_list[1:][1:]))
                else:  # i > min_length
                    break

    def test_returns_correct_scores_deterministic(self):
        for batch_sz in [1, 13]:
            for temp in [1.0, 3.0]:
                n_words = 100
                _non_eos_idxs = [47, 51, 13, 88, 99]
                valid_score_dist_1 = torch.log_softmax(torch.tensor([6.0, 5.0, 4.0, 3.0, 2.0, 1.0]), dim=0)
                valid_score_dist_2 = torch.log_softmax(torch.tensor([6.0, 1.0]), dim=0)
                eos_idx = 2
                lengths = torch.randint(0, 30, (batch_sz,))
                samp = GreedySearch(
                    0,
                    1,
                    2,
                    3,
                    1,
                    1,
                    batch_sz,
                    GlobalScorerStub(),
                    0,
                    False,
                    set(),
                    False,
                    30,
                    temp,
                    1,
                    0,
                    1,
                    False,
                )
                samp.initialize(torch.zeros((1, 1)), lengths)
                # initial step
                i = 0
                word_probs = torch.full((batch_sz, n_words), -float("inf"))
                # batch 0 dies on step 0
                word_probs[0, eos_idx] = valid_score_dist_1[0]
                # include at least one prediction OTHER than EOS
                # that is greater than -1e20
                word_probs[0, _non_eos_idxs] = valid_score_dist_1[1:]
                word_probs[1:, _non_eos_idxs[0] + i] = 0

                attns = torch.randn(1, batch_sz, 53)
                samp.advance(word_probs, attns)
                self.assertTrue(all(samp.is_finished_list[0]))
                samp.update_finished()
                self.assertEqual(
                    [score for score, *_ in samp.hypotheses[0]],
                    [valid_score_dist_1[0]],
                )
                if batch_sz == 1:
                    self.assertTrue(samp.done)
                    continue
                else:
                    self.assertFalse(samp.done)

                # step 2
                i = 1
                word_probs = torch.full((batch_sz - 1, n_words), -float("inf"))
                # (old) batch 8 dies on step 1
                word_probs[7, eos_idx] = valid_score_dist_2[0]
                word_probs[0:7, _non_eos_idxs[:2]] = valid_score_dist_2
                word_probs[8:, _non_eos_idxs[:2]] = valid_score_dist_2

                attns = torch.randn(1, batch_sz, 53)
                samp.advance(word_probs, attns)

                self.assertTrue(all(samp.is_finished_list[7]))
                samp.update_finished()
                self.assertEqual(
                    [score for score, *_ in samp.hypotheses[8]],
                    [valid_score_dist_2[0]],
                )

                # step 3
                i = 2
                word_probs = torch.full((batch_sz - 2, n_words), -float("inf"))
                # everything dies
                word_probs[:, eos_idx] = 0

                attns = torch.randn(1, batch_sz, 53)
                samp.advance(word_probs, attns)

                self.assertTrue(all(all(sublist) for sublist in samp.is_finished_list))
                samp.update_finished()
                self.assertTrue(samp.done)

    def test_returns_correct_scores_non_deterministic(self):
        for batch_sz in [1, 13]:
            for temp in [1.0, 3.0]:
                n_words = 100
                _non_eos_idxs = [47, 51, 13, 88, 99]
                valid_score_dist_1 = torch.log_softmax(torch.tensor([6.0, 5.0, 4.0, 3.0, 2.0, 1.0]), dim=0)
                valid_score_dist_2 = torch.log_softmax(torch.tensor([6.0, 1.0]), dim=0)
                eos_idx = 2
                lengths = torch.randint(0, 30, (batch_sz,))
                samp = GreedySearch(
                    0,
                    1,
                    2,
                    3,
                    1,
                    1,
                    batch_sz,
                    GlobalScorerStub(),
                    0,
                    False,
                    set(),
                    False,
                    30,
                    temp,
                    2,
                    0,
                    1,
                    False,
                )
                samp.initialize(torch.zeros((1, 1)), lengths)
                # initial step
                i = 0
                for _ in range(100):
                    word_probs = torch.full((batch_sz, n_words), -float("inf"))
                    # batch 0 dies on step 0
                    word_probs[0, eos_idx] = valid_score_dist_1[0]
                    # include at least one prediction OTHER than EOS
                    # that is greater than -1e20
                    word_probs[0, _non_eos_idxs] = valid_score_dist_1[1:]
                    word_probs[1:, _non_eos_idxs[0] + i] = 0

                    attns = torch.randn(1, batch_sz, 53)
                    samp.advance(word_probs, attns)
                    if all(samp.is_finished_list[0]):
                        break
                else:
                    self.fail(
                        "Batch 0 never ended (very unlikely but maybe "
                        "due to stochasticisty. If so, please increase "
                        "the range of the for-loop."
                    )
                samp.update_finished()
                self.assertEqual([samp.topk_scores[0]], [valid_score_dist_1[0] / temp])
                if batch_sz == 1:
                    self.assertTrue(samp.done)
                    continue
                else:
                    self.assertFalse(samp.done)

                # step 2
                i = 1
                for _ in range(100):
                    word_probs = torch.full((batch_sz - 1, n_words), -float("inf"))
                    # (old) batch 8 dies on step 1
                    word_probs[7, eos_idx] = valid_score_dist_2[0]
                    word_probs[0:7, _non_eos_idxs[:2]] = valid_score_dist_2
                    word_probs[8:, _non_eos_idxs[:2]] = valid_score_dist_2

                    attns = torch.randn(1, batch_sz, 53)
                    samp.advance(word_probs, attns)
                    if all(samp.is_finished_list[7]):
                        break
                else:
                    self.fail(
                        "Batch 8 never ended (very unlikely but maybe "
                        "due to stochasticisty. If so, please increase "
                        "the range of the for-loop."
                    )

                samp.update_finished()
                self.assertEqual(
                    [score for score, *_ in samp.hypotheses[8]],
                    [valid_score_dist_2[0]],
                )

                # step 3
                i = 2
                for _ in range(250):
                    word_probs = torch.full((samp.alive_seq.shape[0], n_words), -float("inf"))
                    # everything dies
                    word_probs[:, eos_idx] = 0

                    attns = torch.randn(1, batch_sz, 53)
                    samp.advance(word_probs, attns)
                    if any(any(sublist) for sublist in samp.is_finished_list):
                        samp.update_finished()
                    if all(all(sublist) for sublist in samp.is_finished_list):
                        break
                else:
                    self.fail(
                        "All batches never ended (very unlikely but "
                        "maybe due to stochasticisty. If so, please "
                        "increase the range of the for-loop."
                    )

                self.assertTrue(samp.done)

    def test_returns_correct_scores_non_deterministic_beams(self):
        beam_size = 10
        for batch_sz in [1, 13]:
            for temp in [1.0, 3.0]:
                print(batch_sz, temp)
                n_words = 100
                _non_eos_idxs = [47, 51, 13, 88, 99]
                valid_score_dist_1 = torch.log_softmax(torch.tensor([6.0, 5.0, 4.0, 3.0, 2.0, 1.0]), dim=0)
                valid_score_dist_2 = torch.log_softmax(torch.tensor([6.0, 1.0]), dim=0)
                eos_idx = 2
                lengths = torch.randint(0, 30, (batch_sz,))
                samp = GreedySearch(
                    0,
                    1,
                    2,
                    3,
                    1,
                    1,
                    batch_sz,
                    GlobalScorerStub(),
                    0,
                    False,
                    set(),
                    False,
                    30,
                    temp,
                    50,
                    0,
                    beam_size,
                    False,
                )
                samp.initialize(torch.zeros((1, 1)), lengths)
                # initial step
                # finish one beam
                i = 0
                for _ in range(100):
                    word_probs = torch.full((batch_sz * beam_size, n_words), -float("inf"))

                    word_probs[beam_size - 2, eos_idx] = valid_score_dist_1[0]
                    # include at least one prediction OTHER than EOS
                    # that is greater than -1e20
                    word_probs[beam_size - 2, _non_eos_idxs] = valid_score_dist_1[1:]
                    word_probs[beam_size - 2 + 1 :, _non_eos_idxs[0] + i] = 0
                    word_probs[: beam_size - 2, _non_eos_idxs[0] + i] = 0

                    attns = torch.randn(1, batch_sz, 53)
                    samp.advance(word_probs, attns)
                    if all(samp.is_finished_list[beam_size - 2]):
                        self.assertFalse(any(any(sublist) for sublist in samp.is_finished_list[: beam_size - 2]))
                        self.assertFalse(any(samp.is_finished_list[beam_size - 2 + 1]))
                        break
                else:
                    self.fail(
                        "Batch 0 never ended (very unlikely but maybe "
                        "due to stochasticisty. If so, please increase "
                        "the range of the for-loop."
                    )
                samp.update_finished()
                self.assertEqual([samp.topk_scores[beam_size - 2]], [valid_score_dist_1[0] / temp])

                # step 2
                # finish example in last batch
                i = 1
                for _ in range(100):
                    word_probs = torch.full((batch_sz * beam_size - 1, n_words), -float("inf"))
                    # (old) batch 8 dies on step 1
                    word_probs[(batch_sz - 1) * beam_size + 7, eos_idx] = valid_score_dist_2[0]
                    word_probs[: (batch_sz - 1) * beam_size + 7, _non_eos_idxs[:2]] = valid_score_dist_2
                    word_probs[(batch_sz - 1) * beam_size + 8 :, _non_eos_idxs[:2]] = valid_score_dist_2

                    attns = torch.randn(1, batch_sz, 53)
                    samp.advance(word_probs, attns)
                    if all(samp.is_finished_list[(batch_sz - 1) * beam_size + 7]):
                        break
                else:
                    self.fail(
                        "Batch 8 never ended (very unlikely but maybe "
                        "due to stochasticisty. If so, please increase "
                        "the range of the for-loop."
                    )

                samp.update_finished()
                self.assertEqual(
                    [score for score, *_ in samp.hypotheses[batch_sz - 1][-1:]],
                    [valid_score_dist_2[0]],
                )

                # step 3
                i = 2
                for _ in range(250):
                    word_probs = torch.full((samp.alive_seq.shape[0], n_words), -float("inf"))
                    # everything dies
                    word_probs[:, eos_idx] = 0

                    attns = torch.randn(1, batch_sz, 53)
                    samp.advance(word_probs, attns)
                    if any(any(sublist) for sublist in samp.is_finished_list):
                        samp.update_finished()
                    if all(all(sublist) for sublist in samp.is_finished_list):
                        break
                else:
                    self.fail(
                        "All batches never ended (very unlikely but "
                        "maybe due to stochasticisty. If so, please "
                        "increase the range of the for-loop."
                    )

                self.assertTrue(samp.done)

    def test_returns_correct_scores_non_deterministic_topp(self):
        for batch_sz in [1, 13]:
            for temp in [1.0, 0.3]:
                n_words = 100
                _non_eos_idxs = [47, 51, 13, 88, 99]
                valid_score_dist_1 = torch.log_softmax(torch.tensor([6.0, 5.0, 4.0, 3.0, 2.0, 1.0]), dim=0)
                valid_score_dist_2 = torch.log_softmax(torch.tensor([6.0, 1.0]), dim=0)
                eos_idx = 2
                lengths = torch.randint(0, 30, (batch_sz,))
                samp = GreedySearch(
                    0,
                    1,
                    2,
                    3,
                    1,
                    1,
                    batch_sz,
                    GlobalScorerStub(),
                    0,
                    False,
                    set(),
                    False,
                    -1,
                    temp,
                    50,
                    0.5,
                    1,
                    False,
                )
                samp.initialize(torch.zeros((1, 1)), lengths)
                # initial step
                i = 0
                for _ in range(100):
                    word_probs = torch.full((batch_sz, n_words), -float("inf"))
                    # batch 0 dies on step 0
                    word_probs[0, eos_idx] = valid_score_dist_1[0]
                    # include at least one prediction OTHER than EOS
                    # that is greater than -1e20
                    word_probs[0, _non_eos_idxs] = valid_score_dist_1[1:]
                    word_probs[1:, _non_eos_idxs[0] + i] = 0

                    attns = torch.randn(1, batch_sz, 53)
                    samp.advance(word_probs, attns)
                    if all(samp.is_finished_list[0]):
                        break
                else:
                    self.fail(
                        "Batch 0 never ended (very unlikely but maybe "
                        "due to stochasticisty. If so, please increase "
                        "the range of the for-loop."
                    )
                samp.update_finished()
                self.assertEqual(
                    [score for score, *_ in samp.hypotheses[0]],
                    [valid_score_dist_1[0]],
                )
                if batch_sz == 1:
                    self.assertTrue(samp.done)
                    continue
                else:
                    self.assertFalse(samp.done)

                # step 2
                i = 1
                for _ in range(200):
                    word_probs = torch.full((batch_sz - 1, n_words), -float("inf"))
                    # (old) batch 8 dies on step 1
                    word_probs[7, eos_idx] = valid_score_dist_2[0]
                    word_probs[0:7, _non_eos_idxs[:2]] = valid_score_dist_2
                    word_probs[8:, _non_eos_idxs[:2]] = valid_score_dist_2

                    attns = torch.randn(1, batch_sz, 53)
                    samp.advance(word_probs, attns)
                    if all(samp.is_finished_list[7]):
                        break
                else:
                    self.fail(
                        "Batch 8 never ended (very unlikely but maybe "
                        "due to stochasticisty. If so, please increase "
                        "the range of the for-loop."
                    )

                samp.update_finished()
                self.assertEqual(
                    [score for score, *_ in samp.hypotheses[8]],
                    [valid_score_dist_2[0]],
                )

                # step 3
                i = 2
                for _ in range(250):
                    word_probs = torch.full((samp.alive_seq.shape[0], n_words), -float("inf"))
                    # everything dies
                    word_probs[:, eos_idx] = 0

                    attns = torch.randn(1, batch_sz, 53)
                    samp.advance(word_probs, attns)
                    if any(any(sublist) for sublist in samp.is_finished_list):
                        samp.update_finished()
                    if all(all(sublist) for sublist in samp.is_finished_list):
                        break
                else:
                    self.fail(
                        "All batches never ended (very unlikely but "
                        "maybe due to stochasticisty. If so, please "
                        "increase the range of the for-loop."
                    )

                self.assertTrue(samp.done)
