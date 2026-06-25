import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from eole.scorers import get_scorers_cls
from eole.scorers.eole_metricx import EoleMetricXQEScorer, EoleMetricXScorer
from eole.utils.metricx_scorer import metricx_input_templates


class FakeTokenizer:
    def tokenize_string(self, text, side="src", is_train=False):
        return text.split()


class FakeVocab:
    def lookup_token(self, piece):
        if piece == "</s>":
            return 1
        if piece == "<pad>":
            return 0
        return 2


class FakeRuntime:
    def __init__(self, scores, supported_input_modes=None, scoring_type="metricx"):
        vocab = FakeVocab()
        self.vocabs = {
            "src": vocab,
            "tgt": vocab,
            "specials": {"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>", "unk_token": "<unk>"},
            "decoder_start_token": "<pad>",
        }
        self.eos_id = 1
        self.max_length = 64
        self.input_templates = metricx_input_templates("24")
        self.scoring_type = scoring_type
        self.supported_input_modes = supported_input_modes or ["reference", "qe"]
        self.model_config = SimpleNamespace(architecture="transformer_encoder_decoder_scorer")
        self._scores = list(scores)
        self.rows = []
        self.validated_modes = []

    def validate_input_mode(self, input_mode):
        self.validated_modes.append(input_mode)
        if input_mode not in self.supported_input_modes:
            raise ValueError(f"unsupported mode {input_mode}")

    def predict_scores(self, rows):
        self.rows.extend(rows)
        return torch.tensor(self._scores[: len(rows)], dtype=torch.float32)


class TestEoleMetricXScorers(unittest.TestCase):
    def test_metricx_scorers_are_registered(self):
        scorers = get_scorers_cls(["EOLE-METRICX", "EOLE-METRICX-QE"])

        self.assertIs(scorers["EOLE-METRICX"], EoleMetricXScorer)
        self.assertIs(scorers["EOLE-METRICX-QE"], EoleMetricXQEScorer)

    def test_reference_scorer_aggregates_and_forces_reference_mode(self):
        runtime = FakeRuntime(scores=[0.2, 0.4, 0.8])
        scorer = EoleMetricXScorer(SimpleNamespace(metricx_model="dummy", metricx_batch_size=8))

        with (
            patch("eole.scorers.eole_metricx._resolve_model_dir", return_value="dummy"),
            patch.object(EoleMetricXScorer, "_load_model", return_value=runtime),
            patch("eole.scorers.eole_metricx.build_scorer_sentencepiece_transform", return_value=FakeTokenizer()),
        ):
            score = scorer.compute_score(["h1", "h2", "h3"], ["r1", "r2", "r3"], ["s1", "s2", "s3"])

        self.assertEqual(runtime.validated_modes, ["reference"])
        self.assertAlmostEqual(score, (0.2 + 0.4 + 0.8) / 3, places=6)

    def test_qe_scorer_aggregates_and_forces_qe_mode(self):
        runtime = FakeRuntime(scores=[0.2, 0.4])
        scorer = EoleMetricXQEScorer(SimpleNamespace(metricx_model="dummy", metricx_batch_size=8))

        with (
            patch("eole.scorers.eole_metricx._resolve_model_dir", return_value="dummy"),
            patch.object(EoleMetricXQEScorer, "_load_model", return_value=runtime),
            patch("eole.scorers.eole_metricx.build_scorer_sentencepiece_transform", return_value=FakeTokenizer()),
        ):
            score = scorer.compute_score(["h1", "h2"], None, ["s1", "s2"])

        self.assertEqual(runtime.validated_modes, ["qe"])
        self.assertAlmostEqual(score, 0.3, places=6)

    def test_passes_newline_sentinel_setting_to_tokenizer(self):
        runtime = FakeRuntime(scores=[0.2])
        scorer = EoleMetricXScorer(
            SimpleNamespace(metricx_model="dummy", metricx_batch_size=8, metricx_replace_newline_sentinel=False)
        )

        with (
            patch("eole.scorers.eole_metricx._resolve_model_dir", return_value="dummy"),
            patch.object(EoleMetricXScorer, "_load_model", return_value=runtime),
            patch(
                "eole.scorers.eole_metricx.build_scorer_sentencepiece_transform",
                return_value=FakeTokenizer(),
            ) as build_tokenizer,
        ):
            scorer.compute_score(["h1"], ["r1"], ["s1"])

        self.assertFalse(build_tokenizer.call_args.kwargs["replace_newline_sentinel"])

    def test_reference_scorer_requires_references(self):
        runtime = FakeRuntime(scores=[0.2])
        scorer = EoleMetricXScorer(SimpleNamespace(metricx_model="dummy", metricx_batch_size=8))

        with (
            patch("eole.scorers.eole_metricx._resolve_model_dir", return_value="dummy"),
            patch.object(EoleMetricXScorer, "_load_model", return_value=runtime),
        ):
            with self.assertRaisesRegex(ValueError, "requires references"):
                scorer.compute_score(["h1"], None, ["s1"])

    def test_rejects_incompatible_mode(self):
        runtime = FakeRuntime(scores=[0.2], supported_input_modes=["reference"])
        scorer = EoleMetricXQEScorer(SimpleNamespace(metricx_model="dummy", metricx_batch_size=8))

        with (
            patch("eole.scorers.eole_metricx._resolve_model_dir", return_value="dummy"),
            patch.object(EoleMetricXQEScorer, "_load_model", return_value=runtime),
        ):
            with self.assertRaisesRegex(ValueError, "unsupported mode qe"):
                scorer.compute_score(["h1"], None, ["s1"])

    def test_uses_cpu_device_for_configured_cpu(self):
        scorer = EoleMetricXScorer(SimpleNamespace(metricx_model="dummy", metricx_batch_size=8, metricx_device="cpu"))

        self.assertEqual(scorer._resolve_device(), torch.device("cpu"))

    def test_defaults_to_fp32_compute_dtype(self):
        scorer = EoleMetricXScorer(SimpleNamespace(metricx_model="dummy", metricx_batch_size=8, metricx_device="cuda"))

        with patch("eole.scorers.eole_metricx.EncoderDecoderScoringModel.from_model_dir") as from_model_dir:
            from_model_dir.return_value = object()
            scorer._load_model("dummy")

        self.assertEqual(from_model_dir.call_args.args[0], "dummy")
        self.assertEqual(from_model_dir.call_args.kwargs["device"], torch.device("cuda"))
        self.assertEqual(from_model_dir.call_args.kwargs["compute_dtype"], torch.float32)

    def test_uses_fp32_compute_dtype_on_mps_by_default(self):
        scorer = EoleMetricXScorer(SimpleNamespace(metricx_model="dummy", metricx_batch_size=8, metricx_device="mps"))

        with patch("eole.scorers.eole_metricx.EncoderDecoderScoringModel.from_model_dir") as from_model_dir:
            from_model_dir.return_value = object()
            scorer._load_model("dummy")

        self.assertEqual(from_model_dir.call_args.args[0], "dummy")
        self.assertEqual(from_model_dir.call_args.kwargs["device"], torch.device("mps"))
        self.assertEqual(from_model_dir.call_args.kwargs["compute_dtype"], torch.float32)

    def test_explicit_compute_dtype_overrides_default(self):
        for configured_dtype, torch_dtype in (
            ("fp32", torch.float32),
            ("fp16", torch.float16),
            ("bf16", torch.bfloat16),
        ):
            scorer = EoleMetricXScorer(
                SimpleNamespace(
                    metricx_model="dummy",
                    metricx_batch_size=8,
                    metricx_device="mps",
                    metricx_compute_dtype=configured_dtype,
                )
            )

            self.assertEqual(scorer._resolve_compute_dtype(torch.device("mps")), torch_dtype)


if __name__ == "__main__":
    unittest.main()
