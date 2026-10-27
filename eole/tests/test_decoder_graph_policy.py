"""Capture failure and request routing policy also run on CPU-only CI."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from eole.decoders.transformer import TransformerDecoder
from eole.predict.generator import GeneratorLM
from eole.predict.inference import Inference
from eole.predict.greedy_search import GreedySearchLM
from eole.utils.decoder_graph import DecoderGraph


class TestCaptureFailurePolicy(unittest.TestCase):
    def setUp(self):
        self.state = torch.zeros(())
        self.emb = torch.ones(1, 1, 4)
        self.mask = torch.zeros(1, 1, 1, dtype=torch.bool)

        def function(emb, *, tgt_pad_mask):
            self.state.add_(1)
            return emb + self.state

        self.fallback = Mock(side_effect=function)
        self.runner = DecoderGraph(function, self.fallback)
        self.graph = Mock()
        self.stream = SimpleNamespace(device=torch.device("cpu"))

    def fail_capture(self, error):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("torch.cuda.device", return_value=contextlib.nullcontext()))
            stack.enter_context(patch("torch.cuda.Stream", return_value=self.stream))
            stack.enter_context(patch("torch.cuda.current_stream", return_value=self.stream))
            stream_context = stack.enter_context(patch("torch.cuda.stream"))
            stack.enter_context(patch("torch.cuda.CUDAGraph", return_value=self.graph))
            capture = stack.enter_context(patch("torch.cuda.graph"))
            capture.return_value.__enter__.side_effect = error
            warning = stack.enter_context(patch("eole.utils.decoder_graph.logger.warning"))
            for _ in range(2):
                self.runner(self.emb, tgt_pad_mask=self.mask)
            output = self.runner(self.emb, tgt_pad_mask=self.mask)
            torch.testing.assert_close(output, self.emb + 3)
            self.graph.reset.assert_called_once()
            stream_context.return_value.__exit__.assert_called_once()
            capture.assert_called_once_with(self.graph, stream=self.stream, capture_error_mode="thread_local")
            self.runner.reset()
            output = self.runner(self.emb, tgt_pad_mask=self.mask)
            torch.testing.assert_close(output, self.emb + 4)
            capture.assert_called_once()
            warning.assert_called_once()
            self.assertTrue(self.runner.failed)
            self.assertEqual(self.fallback.call_count, 2)

    def test_failure_is_not_retried_after_cache_reset(self):
        self.fail_capture(RuntimeError("capture unsupported"))

    def test_capture_oom_releases_graph_and_tries_uncaptured_execution(self):
        self.fail_capture(torch.cuda.OutOfMemoryError("capture memory overhead"))

    def test_genuine_fallback_oom_propagates(self):
        for _ in range(2):
            self.runner(self.emb, tgt_pad_mask=self.mask)
        self.runner.failed_signatures.add(self.runner.signature)
        self.fallback.side_effect = torch.cuda.OutOfMemoryError("inference allocation")
        with self.assertRaisesRegex(torch.cuda.OutOfMemoryError, "inference allocation"):
            self.runner(self.emb, tgt_pad_mask=self.mask)

    def test_replay_failure_does_not_retry_possibly_mutated_state(self):
        for _ in range(2):
            self.runner(self.emb, tgt_pad_mask=self.mask)
        self.runner.graph = self.graph
        self.runner.inputs = (self.emb.clone(), self.mask.clone())
        self.graph.replay.side_effect = RuntimeError("execution failed")
        with self.assertRaisesRegex(RuntimeError, "execution failed"):
            self.runner(self.emb, tgt_pad_mask=self.mask)
        self.fallback.assert_not_called()


class TestWarmupEligibility(unittest.TestCase):
    @torch.no_grad()
    def test_non_replay_requests_keep_legacy_warmup(self):
        for variant in ("cpu", "fp32", "non_flash", "dynamic", "cross", "attention"):
            with self.subTest(variant=variant):
                decoder = SimpleNamespace(
                    _decode_graph_enabled=True,
                    _decode_graph_request_allowed=True,
                    _verify_graph_request_allowed=False,
                    training=False,
                    flash=variant != "non_flash",
                    dynamic_shapes=variant == "dynamic",
                    with_cross_attn=variant == "cross",
                    _speculative_forward=False,
                    cache_seqlens=torch.zeros(1, dtype=torch.int32),
                    hidden_size=4,
                    compiled_shapes=set(),
                    _forward_compile=Mock(),
                )
                decoder._can_replay_decoder = lambda emb, kwargs, **options: TransformerDecoder._can_replay_decoder(
                    decoder, emb, kwargs, **options
                )
                emb = torch.zeros(1, 8, 4, dtype=torch.float32 if variant == "fp32" else torch.float16)
                TransformerDecoder._compile_decoder(
                    decoder,
                    emb=emb,
                    tgt_pad_mask=torch.zeros(1, 1, 8, dtype=torch.bool),
                    return_attn=variant == "attention",
                )
                decoder._forward_compile.assert_called_once()

    def test_bound_decode_compiles_without_capture_for_effective_batch(self):
        for batch_size in (1, 4):
            with self.subTest(batch_size=batch_size):
                decoder = SimpleNamespace(
                    _can_replay_decoder=Mock(return_value=True),
                    _decode_graph_request_allowed=True,
                    cache_seqlens=torch.zeros(batch_size, dtype=torch.int32),
                    hidden_size=4,
                    _replay_decoder=Mock(),
                    _forward_compile=Mock(),
                )
                emb = torch.zeros(1, 8, 4, dtype=torch.bfloat16)
                mask = torch.zeros(1, 1, 8, dtype=torch.bool)
                TransformerDecoder._compile_decoder(decoder, emb=emb, tgt_pad_mask=mask)
                self.assertTrue(decoder._can_replay_decoder.call_args.kwargs["for_warmup"])
                dummy, kwargs = decoder._replay_decoder.call_args.args
                self.assertEqual(dummy.shape, (batch_size, 1, 4))
                self.assertEqual(dummy.dtype, emb.dtype)
                self.assertEqual(dummy.device, emb.device)
                self.assertEqual(kwargs["tgt_pad_mask"].shape, (batch_size, 1, 1))
                self.assertFalse(kwargs["tgt_pad_mask"].any())
                self.assertFalse(decoder._replay_decoder.call_args.kwargs["capture"])
                decoder._forward_compile.assert_not_called()

    def test_mtp_keeps_separate_verifier_warmup(self):
        decoder = SimpleNamespace(
            _can_replay_decoder=Mock(return_value=True),
            _decode_graph_request_allowed=False,
            _replay_decoder=Mock(),
            _forward_compile=Mock(),
        )
        TransformerDecoder._compile_decoder(
            decoder,
            emb=torch.zeros(1, 8, 4),
            tgt_pad_mask=torch.zeros(1, 1, 8, dtype=torch.bool),
        )
        decoder._replay_decoder.assert_not_called()
        decoder._forward_compile.assert_not_called()


class TestEffectiveSpeculationPolicy(unittest.TestCase):
    def test_mtp_verifier_is_warmed_before_generation_in_all_supported_modes(self):
        class StopBeforeInference(Exception):
            pass

        class Strategy(GreedySearchLM):
            parallel_paths = 1
            return_attention = False
            target_prefix = None

            def __init__(self):
                pass

            def initialize(self, src, src_len, target_prefix):
                return None, src

        for mode in ("0", "1", "2"):
            for fails in (False, True):
                with self.subTest(mode=mode, fails=fails):
                    decoder = Mock()
                    decoder.hidden_size = 4
                    decoder.modules.return_value = []
                    decoder._can_replay_decoder.return_value = True

                    def verify(emb, **kwargs):
                        self.assertTrue(decoder._speculative_forward)
                        length = (4, 3, 2)[decoder.call_count - 1]
                        self.assertEqual(emb.shape, (1, length, 4))
                        self.assertEqual(kwargs["tgt_pad_mask"].shape, (1, 1, length))
                        if fails:
                            raise RuntimeError("verifier warmup failed")

                    decoder.side_effect = verify
                    predictor = SimpleNamespace(
                        dynamic_shapes=False,
                        self_speculative_decoding=True,
                        beam_size=1,
                        n_best=1,
                        top_k=1,
                        temperature=1.0,
                        min_length=0,
                        ban_unk_token=False,
                        block_ngram_repeat=0,
                        model=SimpleNamespace(
                            decoder=decoder,
                            mtp_heads=[SimpleNamespace(emb_norm=True)],
                            tgt_emb=Mock(return_value=torch.zeros(1, 8, 4)),
                        ),
                        self_speculative_num_tokens=3,
                        max_length=128,
                        _tgt_pad_idx=0,
                        warmup_time=[],
                        _gold_score=Mock(return_value=(0, None)),
                        _warmup_prefill=Mock(),
                        _log=Mock(),
                        _log_inference_backends=Mock(side_effect=StopBeforeInference),
                    )
                    predictor._warmup_speculative_decoder = lambda emb, length: GeneratorLM._warmup_speculative_decoder(
                        predictor, emb, length
                    )
                    batch = {
                        "src": torch.ones(1, 8, dtype=torch.long),
                        "srclen": torch.tensor([8]),
                        "left_pad": True,
                    }
                    with (
                        patch("eole.predict.generator.EOLE_TORCH_COMPILE", True),
                        patch("eole.predict.generator.EOLE_COMPILE_MODE", mode),
                        self.assertRaises(RuntimeError if fails else StopBeforeInference),
                    ):
                        GeneratorLM._predict_batch_with_strategy_impl(predictor, batch, Strategy())
                    self.assertEqual(decoder.call_count, 1 if fails else 3)
                    if fails:
                        decoder._replay_decoder.assert_not_called()
                    else:
                        decoder._replay_decoder.assert_called_once()
                        self.assertFalse(decoder._replay_decoder.call_args.kwargs["capture"])
                    self.assertFalse(decoder._speculative_forward)
                    predictor._warmup_prefill.assert_called_once()
                    self.assertFalse(decoder._decode_graph_request_allowed)
                    self.assertTrue(decoder._verify_graph_request_allowed)
                    self.assertEqual(len(predictor.warmup_time), 0 if fails else 1)

    def test_decode_count_excludes_first_token_for_eos_and_length_stops(self):
        predictor = SimpleNamespace(decode_timings=[])
        # One EOS-ended output and one length-ended output; second-best
        # hypotheses are not delivered in the first-best throughput count.
        predictions = [[torch.tensor([10, 11, 12, 99]), torch.arange(20)], [torch.arange(32)]]
        Inference._record_decode_timing(predictor, predictions, 0.5)
        self.assertEqual(predictor.decode_timings, [(34, 0.5)])
        Inference._record_decode_timing(predictor, [[torch.tensor([99])]], 0.0)
        self.assertEqual(predictor.decode_timings[-1], (0, 0.0))

    def test_reduced_verifier_lengths_still_warm_single_token_fallback(self):
        for length in (1, 2, 3, 4):
            with self.subTest(length=length):
                decoder = Mock(hidden_size=4, _speculative_forward=False)
                decoder.modules.return_value = []
                decoder._can_replay_decoder.return_value = True
                predictor = SimpleNamespace(model=SimpleNamespace(decoder=decoder))
                GeneratorLM._warmup_speculative_decoder(predictor, torch.zeros(1, 8, 4), length)
                self.assertEqual([call.args[0].size(1) for call in decoder.call_args_list], list(range(length, 1, -1)))
                self.assertFalse(decoder._speculative_forward)
                decoder._replay_decoder.assert_called_once()
                self.assertEqual(decoder._replay_decoder.call_args.args[0].shape, (1, 1, 4))
                self.assertFalse(decoder._replay_decoder.call_args.kwargs["capture"])

    def test_effective_strategy_selects_request_graph_policy(self):
        class StopBeforeInference(Exception):
            pass

        class Strategy(GreedySearchLM):
            def __init__(self, paths, attention):
                self.parallel_paths = paths
                self.return_attention = attention
                self.target_prefix = None

            def initialize(self, src, src_len, target_prefix):
                return None, src

        cases = (
            ("batch", 4, 1, 1, False, False, False, True, False),
            ("sampling", 1, 1, 40, False, False, False, False, True),
            ("no_heads", 1, 1, 1, False, False, True, True, False),
            ("attention", 1, 1, 1, True, False, False, False, False),
            ("images", 1, 1, 1, False, True, False, False, False),
            ("beam", 1, 2, 1, False, False, False, False, False),
            ("mtp", 1, 1, 1, False, False, False, False, True),
        )
        for (
            name,
            size,
            paths,
            top_k,
            attention,
            images,
            no_heads,
            normal,
            verify,
        ) in cases:
            with self.subTest(name=name):
                decoder = SimpleNamespace(init_state=Mock())
                predictor = SimpleNamespace(
                    dynamic_shapes=False,
                    self_speculative_decoding=True,
                    beam_size=paths,
                    n_best=1,
                    top_k=top_k,
                    temperature=1.0,
                    min_length=0,
                    ban_unk_token=False,
                    block_ngram_repeat=0,
                    model=SimpleNamespace(
                        decoder=decoder,
                        mtp_heads=[] if no_heads else [SimpleNamespace(emb_norm=True)],
                    ),
                    self_speculative_num_tokens=3,
                    max_length=128,
                    _gold_score=Mock(return_value=(0, None)),
                    _log=Mock(),
                    _log_inference_backends=Mock(side_effect=StopBeforeInference),
                )
                batch = {
                    "src": torch.ones(size, 8, dtype=torch.long),
                    "srclen": torch.full((size,), 8),
                    "left_pad": True,
                }
                if images:
                    batch["images"] = torch.ones(1)
                with patch("eole.predict.generator.EOLE_TORCH_COMPILE", False), self.assertRaises(StopBeforeInference):
                    GeneratorLM._predict_batch_with_strategy_impl(predictor, batch, Strategy(paths, attention))
                self.assertEqual(decoder._decode_graph_request_allowed, normal)
                self.assertEqual(decoder._verify_graph_request_allowed, verify)
                predictor._log_inference_backends.assert_called_once_with(speculative=verify)
