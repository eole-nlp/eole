import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from eole.config.models import TransformerEncoderDecoderScorerModelConfig
from eole.models.model import EncoderDecoderScoringModel, get_model_class
from eole.predict import get_infer_class
from eole.predict.encoder_decoder_scorer import EncoderDecoderScorer
from eole.utils.metricx_scorer import metricx_input_templates
from eole.utils.scorer import build_template_inputs, encode_scorer_inputs


class FakeTokenizer:
    def __init__(self):
        self.texts = []

    def tokenize_string(self, text, side="src", is_train=False):
        self.texts.append(text)
        return text.split()


class FakeVocab:
    def lookup_token(self, piece):
        if piece == "</s>":
            return 1
        if piece == "<pad>":
            return 0
        return 2

    def __len__(self):
        return 10


class FakeModel:
    def __init__(self, scores, *, version="24", supported_input_modes=None):
        vocab = FakeVocab()
        self.vocabs = {
            "src": vocab,
            "tgt": vocab,
            "specials": {"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>", "unk_token": "<unk>"},
            "decoder_start_token": "<pad>",
        }
        self.eos_id = 1
        self.max_length = 64
        self.input_templates = metricx_input_templates(version)
        self.supported_input_modes = supported_input_modes or ["reference", "qe"]
        self.strip_eos = True
        self._scores = list(scores)
        self._offset = 0
        self.rows = []

    def predict_scores(self, rows):
        self.rows.extend(rows)
        n = len(rows)
        out = torch.tensor(self._scores[self._offset : self._offset + n], dtype=torch.float32)
        self._offset += n
        return out

    def validate_input_mode(self, input_mode):
        if input_mode not in self.supported_input_modes:
            raise ValueError(f"unsupported mode {input_mode}")


def _write(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


class TestMetricXFormatting(unittest.TestCase):
    def test_metricx24_reference_format(self):
        self.assertEqual(
            build_template_inputs(["src"], ["hyp"], ["ref"], metricx_input_templates("24"), input_mode="reference")[0],
            "source: src candidate: hyp reference: ref",
        )

    def test_metricx24_qe_format(self):
        self.assertEqual(
            build_template_inputs(["src"], ["hyp"], None, metricx_input_templates("24"), input_mode="qe")[0],
            "source: src candidate: hyp",
        )

    def test_metricx23_reference_format(self):
        self.assertEqual(
            build_template_inputs(["src"], ["hyp"], ["ref"], metricx_input_templates("23"), input_mode="reference")[0],
            "candidate: hyp reference: ref",
        )

    def test_metricx23_qe_format(self):
        self.assertEqual(
            build_template_inputs(["src"], ["hyp"], None, metricx_input_templates("23"), input_mode="qe")[0],
            "candidate: hyp source: src",
        )

    def test_metricx_encoding_removes_eos(self):
        class EosTokenizer:
            def tokenize_string(self, text, side="src", is_train=False):
                return ["a", "</s>"]

        model = FakeModel(scores=[])
        self.assertEqual(encode_scorer_inputs(["a"], EosTokenizer(), model), [[2]])


class TestEncoderDecoderScorer(unittest.TestCase):
    def test_predict_writes_segment_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            src_path = os.path.join(tmp, "src.txt")
            tgt_path = os.path.join(tmp, "tgt.txt")
            ref_path = os.path.join(tmp, "ref.txt")
            _write(src_path, ["s1", "s2"])
            _write(tgt_path, ["h1", "h2"])
            _write(ref_path, ["r1", "r2"])
            model = FakeModel(scores=[1.0, 2.0])
            config = SimpleNamespace(
                src=src_path,
                tgt=tgt_path,
                ref=ref_path,
                model_path=[tmp],
                output=os.path.join(tmp, "scores.txt"),
                batch_size=2,
                report_time=False,
                with_score=True,
            )
            scorer = EncoderDecoderScorer(model, model.vocabs, config, model_config=SimpleNamespace())
            tokenizer = FakeTokenizer()

            with patch(
                "eole.predict.encoder_decoder_scorer.build_scorer_sentencepiece_transform",
                return_value=tokenizer,
            ):
                scores_out, estims_out, preds_out = scorer._predict(
                    infer_iter=None, transforms={}, attn_debug=False, align_debug=False
                )

            self.assertEqual(scores_out, [[]])
            self.assertEqual(preds_out, [[]])
            self.assertEqual(estims_out, [[1.0, 2.0]])
            self.assertEqual(model.rows[0]["input_ids"], [2, 2, 2, 2, 2, 2])
            self.assertEqual(tokenizer.texts[0], "source: s1 candidate: h1 reference: r1")

    def test_predict_system_score_level_outputs_mean_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            src_path = os.path.join(tmp, "src.txt")
            tgt_path = os.path.join(tmp, "tgt.txt")
            _write(src_path, ["s1", "s2"])
            _write(tgt_path, ["h1", "h2"])
            model = FakeModel(scores=[1.0, 3.0])
            config = SimpleNamespace(
                src=src_path,
                tgt=tgt_path,
                ref=None,
                model_path=[tmp],
                output=os.path.join(tmp, "scores.txt"),
                batch_size=2,
                report_time=False,
                with_score=True,
                score_level="system",
            )
            scorer = EncoderDecoderScorer(model, model.vocabs, config, model_config=SimpleNamespace())
            tokenizer = FakeTokenizer()

            with patch(
                "eole.predict.encoder_decoder_scorer.build_scorer_sentencepiece_transform",
                return_value=tokenizer,
            ):
                _, estims_out, _ = scorer._predict(infer_iter=None, transforms={}, attn_debug=False, align_debug=False)

            self.assertEqual(estims_out, [[2.0]])
            self.assertEqual(tokenizer.texts[0], "source: s1 candidate: h1")

    def test_missing_reference_errors_when_qe_mode_is_unsupported(self):
        with tempfile.TemporaryDirectory() as tmp:
            src_path = os.path.join(tmp, "src.txt")
            tgt_path = os.path.join(tmp, "tgt.txt")
            _write(src_path, ["s1"])
            _write(tgt_path, ["h1"])
            model = FakeModel(scores=[1.0], supported_input_modes=["reference"])
            config = SimpleNamespace(
                src=src_path,
                tgt=tgt_path,
                ref=None,
                model_path=[tmp],
                output=os.path.join(tmp, "scores.txt"),
                batch_size=2,
                report_time=False,
                with_score=True,
            )
            scorer = EncoderDecoderScorer(model, model.vocabs, config, model_config=SimpleNamespace())

            with self.assertRaisesRegex(ValueError, "unsupported mode qe"):
                scorer._predict(infer_iter=None, transforms={}, attn_debug=False, align_debug=False)

    def test_reference_mode_errors_when_reference_mode_is_unsupported(self):
        with tempfile.TemporaryDirectory() as tmp:
            src_path = os.path.join(tmp, "src.txt")
            tgt_path = os.path.join(tmp, "tgt.txt")
            ref_path = os.path.join(tmp, "ref.txt")
            _write(src_path, ["s1"])
            _write(tgt_path, ["h1"])
            _write(ref_path, ["r1"])
            model = FakeModel(scores=[1.0], supported_input_modes=["qe"])
            config = SimpleNamespace(
                src=src_path,
                tgt=tgt_path,
                ref=ref_path,
                model_path=[tmp],
                output=os.path.join(tmp, "scores.txt"),
                batch_size=2,
                report_time=False,
                with_score=True,
            )
            scorer = EncoderDecoderScorer(model, model.vocabs, config, model_config=SimpleNamespace())

            with self.assertRaisesRegex(ValueError, "unsupported mode reference"):
                scorer._predict(infer_iter=None, transforms={}, attn_debug=False, align_debug=False)

    def test_config_and_dispatch_for_encoder_decoder_scorer(self):
        config = TransformerEncoderDecoderScorerModelConfig(
            hidden_size=8,
            heads=2,
            transformer_ff=16,
            encoder={"layers": 1},
            decoder={"layers": 1},
        )

        self.assertEqual(config.architecture, "transformer_encoder_decoder_scorer")
        self.assertEqual(config.scoring_type, "token_regression")
        self.assertEqual(config.score_position, 0)
        self.assertEqual(config.decoder_input_length, 2)
        self.assertIs(get_model_class(config), EncoderDecoderScoringModel)
        self.assertIs(get_infer_class(config), EncoderDecoderScorer)

    def test_config_rejects_score_position_outside_decoder_input_length(self):
        with self.assertRaisesRegex(ValueError, "score_position must be < decoder_input_length"):
            TransformerEncoderDecoderScorerModelConfig(
                hidden_size=8,
                heads=2,
                transformer_ff=16,
                score_position=2,
                decoder_input_length=2,
                encoder={"layers": 1},
                decoder={"layers": 1},
            )


if __name__ == "__main__":
    unittest.main()
