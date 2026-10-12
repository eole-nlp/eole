import unittest

import torch

from eole.predict.greedy_search import GreedySearch, GreedySearchLM
from eole.tests.test_greedy_search import GlobalScorerStub


def strategy(device, minimum=0, maximum=5, batch=1):
    search = GreedySearch(
        0, 1, [2, 4], 3, 1, 1, batch, GlobalScorerStub(), minimum, False, set(), False, maximum, 1.0, 1, 0, 1, False
    )
    search.initialize(torch.zeros((batch, 1), device=device), torch.ones(batch, dtype=torch.long, device=device))
    return search


class TestAsyncGreedySearch(unittest.TestCase):
    def test_cpu_rejected(self):
        with self.assertRaisesRegex(ValueError, "CUDA sequences"):
            strategy("cpu").enable_async_finish()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for asynchronous copies")
    def test_minimum_multiple_eos_and_maximum(self):
        for minimum in (0, 2):
            for token in (2, 4, 6):
                reference = strategy("cuda", minimum)
                asynchronous = strategy("cuda", minimum)
                asynchronous.enable_async_finish()
                for step in range(5):
                    scores = torch.full((1, 8), -10.0, device="cuda")
                    scores[0, 5] = -1.0
                    scores[0, token] = 0.0
                    reference.advance(scores.clone(), None)
                    asynchronous.advance(scores.clone(), None)
                    # Queue work after the asynchronous copy, as lookahead does.
                    later = torch.ones(32, device="cuda").square()
                    asynchronous.resolve_async_finish()
                    self.assertEqual(reference.is_finished_list, asynchronous.is_finished_list)
                    self.assertTrue(torch.equal(reference.alive_seq, asynchronous.alive_seq))
                    self.assertEqual(later.shape, (32,))
                    if step == 4:
                        self.assertEqual(asynchronous.is_finished_list, [[True]])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for asynchronous copies")
    def test_eos_finalization(self):
        for token in (2, 4):
            search = strategy("cuda")
            search.enable_async_finish()
            scores = torch.full((1, 8), -10.0, device="cuda")
            scores[0, token] = 0.0
            search.advance(scores, None)
            search.resolve_async_finish()
            search.update_finished()
            self.assertTrue(search.done)
            self.assertEqual(search.predictions[0][0].tolist(), [token])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for asynchronous copies")
    def test_batch_shrinks_after_staggered_eos(self):
        reference = strategy("cuda", maximum=8, batch=4)
        asynchronous = strategy("cuda", maximum=8, batch=4)
        asynchronous.enable_async_finish()
        sizes = []
        for step in range(8):
            count = reference.alive_seq.shape[0]
            sizes.append(count)
            scores = torch.full((count, 8), -10.0, device="cuda")
            scores[:, 5] = 0.0
            mask = len(reference) >= reference.original_batch_idx + 3
            scores[mask, 2] = 1.0
            reference.advance(scores.clone(), None)
            asynchronous.advance(scores.clone(), None)
            asynchronous.resolve_async_finish()
            self.assertEqual(reference.is_finished_list, asynchronous.is_finished_list)
            if any(any(row) for row in reference.is_finished_list):
                reference.update_finished()
                asynchronous.update_finished()
            self.assertEqual(reference.done, asynchronous.done)
            self.assertTrue(torch.equal(reference.alive_seq, asynchronous.alive_seq))
            if reference.done:
                break
        self.assertEqual(set(sizes), {1, 2, 3, 4})
        self.assertEqual(
            [x[0].tolist() for x in reference.predictions], [x[0].tolist() for x in asynchronous.predictions]
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for asynchronous copies")
    def test_stream_token_is_available_on_cpu(self):
        search = strategy("cuda")
        search.enable_async_finish(streaming=True)
        scores = torch.full((1, 8), -10.0, device="cuda")
        scores[0, 6] = 0.0
        search.advance(scores, None)
        search.resolve_async_finish()
        self.assertEqual(search._stream_token_host.device.type, "cpu")
        self.assertEqual(search._stream_token_host.tolist(), [[6]])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for asynchronous copies")
    def test_static_batch_preserves_predictions_and_scores(self):
        reference = strategy("cuda", maximum=6, batch=4)
        asynchronous = strategy("cuda", maximum=6, batch=4)
        for search in (reference, asynchronous):
            search.static_batch_size = True
        asynchronous.enable_async_finish()
        for step in range(6):
            scores = torch.full((4, 8), -10.0, device="cuda")
            scores[:, 5] = -1.0
            scores[step >= torch.arange(4, device="cuda") + 1, 2] = 0.0
            reference.advance(scores.clone(), None)
            asynchronous.advance(scores.clone(), None)
            asynchronous.resolve_async_finish()
            self.assertEqual(reference.is_finished_list, asynchronous.is_finished_list)
            if any(row[0] for row in reference.is_finished_list):
                reference.update_finished()
                asynchronous.update_finished()
            self.assertEqual(reference.done, asynchronous.done)
            if reference.done:
                break
        self.assertTrue(reference.done)
        for ref, actual in zip(reference.predictions, asynchronous.predictions):
            torch.testing.assert_close(ref[0], actual[0])
        for ref, actual in zip(reference.scores, asynchronous.scores):
            torch.testing.assert_close(ref[0], actual[0])


class TestPipelineRouting(unittest.TestCase):
    def test_greedy_beam_and_mtp_routing(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from eole.predict.generator import GeneratorLM
        from eole.predict.beam_search import BeamSearchLM

        predictor = SimpleNamespace(add_estimator=False)
        greedy = GreedySearchLM.__new__(GreedySearchLM)
        greedy.temperature = 1.0
        greedy.top_k = 1
        greedy.parallel_paths = 1
        greedy.batch_size = 1
        greedy.return_attention = False
        cuda = SimpleNamespace(is_cuda=True)
        with patch.dict("os.environ", {"EOLE_EOS_PIPELINE": "1"}):
            for mtp in (False, True):
                self.assertEqual(GeneratorLM._can_pipeline_eos(predictor, greedy, cuda, mtp, None), not mtp)
                beam = BeamSearchLM.__new__(BeamSearchLM)
                self.assertFalse(GeneratorLM._can_pipeline_eos(predictor, beam, cuda, mtp, None))
            greedy.batch_size = 4
            self.assertTrue(GeneratorLM._can_pipeline_eos(predictor, greedy, cuda, False, None))
            self.assertFalse(GeneratorLM._can_pipeline_eos(predictor, greedy, cuda, False, object()))
            greedy.batch_size = 1
            self.assertTrue(GeneratorLM._can_pipeline_eos(predictor, greedy, cuda, False, object()))
            self.assertFalse(
                GeneratorLM._can_pipeline_eos(predictor, greedy, SimpleNamespace(is_cuda=False), False, None)
            )
            greedy.return_attention = True
            self.assertFalse(GeneratorLM._can_pipeline_eos(predictor, greedy, cuda, False, None))
        with patch.dict("os.environ", {"EOLE_EOS_PIPELINE": "0"}):
            self.assertFalse(GeneratorLM._can_pipeline_eos(predictor, greedy, cuda, False, None))

    def test_exception_drains_pending_copy(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from eole.predict.generator import GeneratorLM

        decoder = Mock()
        decoder.modules.return_value = []
        predictor = SimpleNamespace(model=SimpleNamespace(decoder=decoder))
        predictor._predict_batch_with_strategy_impl = Mock(side_effect=RuntimeError("decode failed"))
        search = SimpleNamespace(_finish_pending=True, resolve_async_finish=Mock())
        with self.assertRaisesRegex(RuntimeError, "decode failed"):
            GeneratorLM._predict_batch_with_strategy(predictor, {}, search)
        search.resolve_async_finish.assert_called_once()
        decoder._disable_cache.assert_called_once()
