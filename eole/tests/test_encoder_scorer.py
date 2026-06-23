import os
import tempfile
import unittest
from types import SimpleNamespace

import torch

from eole.predict.encoder_scorer import EncoderScorer, _resolve_sentencepiece
from eole.inference_engine import InferenceEngine
from eole.models.model import EncoderScoringModel
from eole.config.models import TransformerEncoderScorerModelConfig
from eole.modules.estimator import FeedForward
from eole.utils.encoder_scorer import build_segment_rows


class FakeTokenizer:
    def tokenize_string(self, text, side="src", is_train=False):
        return text.strip().split()


class FakeVocab:
    def lookup_token(self, piece):
        return 2 if piece else 0

    def __len__(self):
        return 10


class FakeModel:
    def __init__(self, scores, raises_oom_until=None):
        src_vocab = FakeVocab()
        self.bos_id = 0
        self.eos_id = 1
        self.max_length = 64
        self.vocabs = {
            "src": src_vocab,
            "tgt": src_vocab,
            "specials": {"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>", "unk_token": "<unk>"},
            "decoder_start_token": "<s>",
        }
        self._scores = list(scores)
        self._raises_oom_until = raises_oom_until or 0
        self._oom_calls = 0
        self._offset = 0
        self.calls = []
        self.requires_reference = False
        self.input_segments = []

    def predict_scores(self, rows):
        self.calls.append(len(rows))
        if self._oom_calls < self._raises_oom_until:
            self._oom_calls += 1
            raise torch.cuda.OutOfMemoryError("simulated OOM")
        n = len(rows)
        out = torch.tensor(self._scores[self._offset : self._offset + n], dtype=torch.float32)
        self._offset += n
        return out


def _write(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _encode_texts_for_test(texts, tokenizer, model):
    return [[model.bos_id, *[2 for _ in tokenizer.tokenize_string(text)], model.eos_id] for text in texts]


class TestEncoderScorer(unittest.TestCase):
    def test_resolves_sentencepiece_from_dict(self):
        tx = FakeTokenizer()
        self.assertIs(_resolve_sentencepiece({"sentencepiece": tx, "filtertoolong": object()}), tx)

    def test_resolves_sentencepiece_from_iterable(self):
        tx = FakeTokenizer()
        self.assertIs(_resolve_sentencepiece([object(), tx]), tx)

    def test_resolves_sentencepiece_missing_returns_none(self):
        self.assertIsNone(_resolve_sentencepiece([object()]))
        self.assertIsNone(_resolve_sentencepiece({"filtertoolong": object()}))

    def test_predict_writes_one_score_per_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            src_path = os.path.join(tmp, "src.txt")
            tgt_path = os.path.join(tmp, "tgt.txt")
            out_path = os.path.join(tmp, "scores.txt")
            _write(src_path, ["hello world", "foo bar baz", "x"])
            _write(tgt_path, ["bonjour monde", "qux quux quuux", "y"])
            model = FakeModel(scores=[0.1, 0.5, 0.9])
            config = SimpleNamespace(
                src=src_path, tgt=tgt_path, output=out_path, batch_size=2, report_time=False, with_score=True
            )
            scorer = EncoderScorer(model, model.vocabs, config, model_config=SimpleNamespace())
            scores_out, estims_out, preds_out = scorer._predict(
                infer_iter=None, transforms={"sentencepiece": FakeTokenizer()}, attn_debug=False, align_debug=False
            )
            self.assertEqual(scores_out, [[]])
            self.assertEqual(preds_out, [[]])
            self.assertEqual(len(estims_out[0]), 3)
            self.assertAlmostEqual(estims_out[0][0], 0.1, places=6)
            self.assertAlmostEqual(estims_out[0][2], 0.9, places=6)

    def test_predict_system_score_level_outputs_mean_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            src_path = os.path.join(tmp, "src.txt")
            tgt_path = os.path.join(tmp, "tgt.txt")
            out_path = os.path.join(tmp, "scores.txt")
            _write(src_path, ["a", "b", "c"])
            _write(tgt_path, ["x", "y", "z"])
            model = FakeModel(scores=[0.1, 0.5, 0.9])
            config = SimpleNamespace(
                src=src_path,
                tgt=tgt_path,
                output=out_path,
                batch_size=2,
                report_time=False,
                with_score=True,
                score_level="system",
            )
            scorer = EncoderScorer(model, model.vocabs, config, model_config=SimpleNamespace())
            scores_out, estims_out, preds_out = scorer._predict(
                infer_iter=None, transforms={"sentencepiece": FakeTokenizer()}, attn_debug=False, align_debug=False
            )

            self.assertEqual(scores_out, [[]])
            self.assertEqual(preds_out, [[]])
            self.assertEqual(len(estims_out[0]), 1)
            self.assertAlmostEqual(estims_out[0][0], 0.5, places=6)

    def test_system_score_level_writes_single_score_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "scores.txt")
            engine = InferenceEngine(SimpleNamespace(with_score=True))

            engine._write_predictions_to_file([], [0.5], [], out_path)

            with open(out_path, "r", encoding="utf-8") as f:
                self.assertEqual(f.read(), "0.5\n")

    def test_oom_fallback_halves_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            src_path = os.path.join(tmp, "src.txt")
            tgt_path = os.path.join(tmp, "tgt.txt")
            out_path = os.path.join(tmp, "scores.txt")
            _write(src_path, ["a"] * 8)
            _write(tgt_path, ["b"] * 8)
            model = FakeModel(scores=[0.1] * 8, raises_oom_until=1)
            config = SimpleNamespace(
                src=src_path, tgt=tgt_path, output=out_path, batch_size=8, report_time=False, with_score=True
            )
            scorer = EncoderScorer(model, model.vocabs, config, model_config=SimpleNamespace())
            scorer._predict(
                infer_iter=None, transforms={"sentencepiece": FakeTokenizer()}, attn_debug=False, align_debug=False
            )
            # First call raises OOM at batch=8, then halves to 4 → 4 → 4 (succeeds at half)
            self.assertEqual(model.calls, [8, 4, 4])

    def test_fallback_only_for_comet_scoring_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            src_path = os.path.join(tmp, "src.txt")
            tgt_path = os.path.join(tmp, "tgt.txt")
            out_path = os.path.join(tmp, "scores.txt")
            _write(src_path, ["a"])
            _write(tgt_path, ["b"])
            model = FakeModel(scores=[0.1])
            model.scoring_type = "pooled_regression"
            config = SimpleNamespace(
                src=src_path,
                tgt=tgt_path,
                output=out_path,
                batch_size=2,
                report_time=False,
                model_path=[tmp],
                with_score=True,
            )
            scorer = EncoderScorer(model, model.vocabs, config, model_config=SimpleNamespace())
            with self.assertRaisesRegex(ValueError, "scoring_type='pooled_regression'"):
                scorer._predict(infer_iter=None, transforms={}, attn_debug=False, align_debug=False)

    def test_mismatched_lengths_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            src_path = os.path.join(tmp, "src.txt")
            tgt_path = os.path.join(tmp, "tgt.txt")
            out_path = os.path.join(tmp, "scores.txt")
            _write(src_path, ["a", "b"])
            _write(tgt_path, ["c"])
            model = FakeModel(scores=[0.1])
            config = SimpleNamespace(
                src=src_path, tgt=tgt_path, output=out_path, batch_size=2, report_time=False, with_score=True
            )
            scorer = EncoderScorer(model, model.vocabs, config, model_config=SimpleNamespace())
            with self.assertRaises(ValueError):
                scorer._predict(
                    infer_iter=None, transforms={"sentencepiece": FakeTokenizer()}, attn_debug=False, align_debug=False
                )

    def test_reference_model_requires_ref_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            src_path = os.path.join(tmp, "src.txt")
            tgt_path = os.path.join(tmp, "tgt.txt")
            out_path = os.path.join(tmp, "scores.txt")
            _write(src_path, ["a"])
            _write(tgt_path, ["b"])
            model = FakeModel(scores=[0.1])
            model.requires_reference = True
            config = SimpleNamespace(
                src=src_path, tgt=tgt_path, output=out_path, batch_size=2, report_time=False, with_score=True
            )
            scorer = EncoderScorer(model, model.vocabs, config, model_config=SimpleNamespace())
            with self.assertRaisesRegex(ValueError, "Provide --ref"):
                scorer._predict(
                    infer_iter=None, transforms={"sentencepiece": FakeTokenizer()}, attn_debug=False, align_debug=False
                )

    def test_reference_model_with_ref_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            src_path = os.path.join(tmp, "src.txt")
            tgt_path = os.path.join(tmp, "tgt.txt")
            ref_path = os.path.join(tmp, "ref.txt")
            out_path = os.path.join(tmp, "scores.txt")
            _write(src_path, ["a", "b"])
            _write(tgt_path, ["c", "d"])
            _write(ref_path, ["e", "f"])
            model = FakeModel(scores=[0.1, 0.2])
            model.requires_reference = True
            config = SimpleNamespace(
                src=src_path,
                tgt=tgt_path,
                ref=ref_path,
                output=out_path,
                batch_size=2,
                report_time=False,
                with_score=True,
            )
            scorer = EncoderScorer(model, model.vocabs, config, model_config=SimpleNamespace())
            scores_out, estims_out, preds_out = scorer._predict(
                infer_iter=None, transforms={"sentencepiece": FakeTokenizer()}, attn_debug=False, align_debug=False
            )
            self.assertEqual(scores_out, [[]])
            self.assertEqual(preds_out, [[]])
            self.assertEqual(len(estims_out[0]), 2)

    def test_reference_model_ref_length_mismatch_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            src_path = os.path.join(tmp, "src.txt")
            tgt_path = os.path.join(tmp, "tgt.txt")
            ref_path = os.path.join(tmp, "ref.txt")
            out_path = os.path.join(tmp, "scores.txt")
            _write(src_path, ["a", "b"])
            _write(tgt_path, ["c", "d"])
            _write(ref_path, ["e"])
            model = FakeModel(scores=[0.1, 0.2])
            model.requires_reference = True
            config = SimpleNamespace(
                src=src_path,
                tgt=tgt_path,
                ref=ref_path,
                output=out_path,
                batch_size=2,
                report_time=False,
                with_score=True,
            )
            scorer = EncoderScorer(model, model.vocabs, config, model_config=SimpleNamespace())
            with self.assertRaisesRegex(ValueError, "src/tgt/ref line counts differ"):
                scorer._predict(
                    infer_iter=None, transforms={"sentencepiece": FakeTokenizer()}, attn_debug=False, align_debug=False
                )

    def test_build_segment_rows_includes_ref_for_ref_input_segment(self):
        model = FakeModel(scores=[0.1])
        model.input_segments = ["ref"]

        rows = build_segment_rows(["mt"], ["src"], ["ref"], FakeTokenizer(), model, _encode_texts_for_test)

        self.assertIn("ref_ids", rows[0])

    def test_build_segment_rows_rejects_src_length_mismatch(self):
        model = FakeModel(scores=[0.1])

        with self.assertRaisesRegex(ValueError, "texts_srcs"):
            build_segment_rows(["mt", "mt2"], ["src"], None, FakeTokenizer(), model, _encode_texts_for_test)

    def test_build_segment_rows_rejects_ref_length_mismatch(self):
        model = FakeModel(scores=[0.1])

        with self.assertRaisesRegex(ValueError, "texts_refs"):
            build_segment_rows(["mt", "mt2"], ["src", "src2"], ["ref"], FakeTokenizer(), model, _encode_texts_for_test)

    def test_xcomet_config_sets_reference_requirement_from_segments(self):
        config = TransformerEncoderScorerModelConfig(
            class_identifier="xcomet_metric",
            requires_reference=False,
            input_segments=["mt", "src", "ref"],
        )

        self.assertTrue(config.requires_reference)
        self.assertEqual(config.error_labels, ["minor", "major", "critical"])

    def test_xcomet_config_rejects_custom_error_labels(self):
        with self.assertRaisesRegex(ValueError, "error_labels"):
            TransformerEncoderScorerModelConfig(
                class_identifier="xcomet_metric",
                error_labels=["minor", "major"],
            )

    def test_xcomet_config_rejects_malformed_input_weights(self):
        with self.assertRaisesRegex(ValueError, "input_weights_spans"):
            TransformerEncoderScorerModelConfig(
                class_identifier="xcomet_metric",
                input_weights_spans=[1.0],
            )

    def test_xcomet_config_rejects_malformed_score_weights(self):
        with self.assertRaisesRegex(ValueError, "score_weights"):
            TransformerEncoderScorerModelConfig(
                class_identifier="xcomet_metric",
                score_weights=[0.33, 0.33, 0.34],
            )

    def test_pooled_regression_uses_configured_segment_ids(self):
        model = EncoderScoringModel.__new__(EncoderScoringModel)
        model.scoring_type = "pooled_regression"
        model.input_segments = ["mt"]
        model._sentence_embed_from_ids = lambda seg: torch.tensor([[float(len(ids))] for ids in seg])
        model.estimator = lambda rep: rep

        scores = model.predict_scores([{"src_ids": [0], "mt_ids": [0, 2, 1]}, {"src_ids": [0], "mt_ids": [0, 2, 2, 1]}])

        self.assertEqual(scores.tolist(), [3.0, 4.0])

    def test_pooled_regression_errors_when_configured_segment_missing(self):
        model = EncoderScoringModel.__new__(EncoderScoringModel)
        model.scoring_type = "pooled_regression"
        model.input_segments = ["ref"]

        with self.assertRaisesRegex(ValueError, "requires 'ref' segment ids"):
            model.predict_scores([{"src_ids": [0], "mt_ids": [0, 2, 1]}])

    def test_checkpoint_key_mapping_for_converted_comet_weights(self):
        model = EncoderScoringModel.__new__(EncoderScoringModel)

        self.assertEqual(
            model._checkpoint_key_for_param("src_emb.word_embeddings.weight"),
            "encoder.embeddings.word_embeddings.weight",
        )
        self.assertEqual(
            model._checkpoint_key_for_param("representation.layerwise_attention.scalar_parameters.0"),
            "layerwise_attention.scalar_parameters.0",
        )
        self.assertEqual(
            model._checkpoint_key_for_param("encoder.transformer_layers.0.weight"),
            "encoder.transformer_layers.0.weight",
        )

    def test_strict_checkpoint_validation_raises_on_missing_or_extra_keys(self):
        model = EncoderScoringModel.__new__(EncoderScoringModel)
        model.state_dict = lambda: {
            "src_emb.word_embeddings.weight": torch.empty(1),
            "encoder.transformer_layers.0.weight": torch.empty(1),
        }
        model._last_loaded_checkpoint_keys = {
            "encoder.embeddings.word_embeddings.weight",
            "unexpected.weight",
        }

        with self.assertRaisesRegex(
            RuntimeError, "missing keys: encoder.transformer_layers.0.weight; extra keys: unexpected.weight"
        ):
            model._validate_strict_checkpoint_load({"encoder.embeddings.word_embeddings.weight": True})

    def test_strict_checkpoint_validation_accepts_complete_remapped_load(self):
        model = EncoderScoringModel.__new__(EncoderScoringModel)
        model.state_dict = lambda: {
            "src_emb.word_embeddings.weight": torch.empty(1),
            "representation.layerwise_attention.gamma": torch.empty(1),
        }
        model._last_loaded_checkpoint_keys = {
            "encoder.embeddings.word_embeddings.weight",
            "layerwise_attention.gamma",
        }

        model._validate_strict_checkpoint_load(
            {
                "encoder.embeddings.word_embeddings.weight": True,
                "layerwise_attention.gamma": True,
            }
        )

    def test_xcomet_mqm_scores_from_spans(self):
        model = EncoderScoringModel.__new__(EncoderScoringModel)

        scores = model._xcomet_mqm_scores(
            [
                [{"severity": "minor"}, {"severity": "major"}],
                [{"severity": "critical"}, {"severity": "critical"}, {"severity": "critical"}],
            ]
        )

        self.assertAlmostEqual(scores.tolist()[0], 19 / 25, places=6)
        self.assertEqual(scores.tolist()[1], 0.0)

    def test_xcomet_decode_groups_error_spans(self):
        vocab = FakeVocab()
        vocab.lookup_index = lambda idx: {2: "▁bad", 3: "ly"}.get(idx, "")
        model = EncoderScoringModel.__new__(EncoderScoringModel)
        model.bos_id = 0
        model.eos_id = 1
        model.pad_idx = 9
        model.vocabs = {"src": vocab}
        model.ids_to_error_label = {0: "O", 1: "I-minor", 2: "I-major", 3: "I-critical"}
        model.decoding_threshold = None

        probs = torch.tensor([[[0.9, 0.1, 0.0, 0.0], [0.1, 0.8, 0.1, 0.0], [0.1, 0.7, 0.2, 0.0], [0.8, 0.2, 0.0, 0.0]]])
        spans = model._decode_xcomet_spans(probs, torch.tensor([[0, 2, 3, 1]]), [[(0, 0), (0, 3), (3, 5), (0, 0)]])

        self.assertEqual(len(spans[0]), 1)
        self.assertEqual(spans[0][0]["severity"], "minor")
        self.assertEqual(spans[0][0]["start"], 0)
        self.assertEqual(spans[0][0]["end"], 5)
        self.assertEqual(spans[0][0]["text"], "badly")

    def test_xcomet_decode_drops_unclosed_trailing_span_to_match_unbabel(self):
        vocab = FakeVocab()
        vocab.lookup_index = lambda idx: {2: "▁bad", 3: "ly"}.get(idx, "")
        model = EncoderScoringModel.__new__(EncoderScoringModel)
        model.bos_id = 0
        model.eos_id = 1
        model.pad_idx = 9
        model.vocabs = {"src": vocab}
        model.ids_to_error_label = {0: "O", 1: "I-minor", 2: "I-major", 3: "I-critical"}
        model.decoding_threshold = None

        probs = torch.tensor([[[0.9, 0.1, 0.0, 0.0], [0.1, 0.8, 0.1, 0.0], [0.1, 0.7, 0.2, 0.0]]])
        spans = model._decode_xcomet_spans(probs, torch.tensor([[0, 2, 3]]), [[(0, 0), (0, 3), (3, 5)]])

        self.assertEqual(spans, [[]])

    def test_xcomet_predict_combines_regression_and_spans(self):
        model = EncoderScoringModel.__new__(EncoderScoringModel)
        model.class_identifier = "xcomet_metric"
        model.input_segments = ["mt", "src", "ref"]
        model.score_weights = [0.12, 0.33, 0.33, 0.22]
        model.input_weights_spans = [0.1667, 0.3333, 0.5]
        model.bos_id = 0
        model.eos_id = 1
        model.pad_idx = 9
        model.device = torch.device("cpu")
        model.vocabs = {"src": FakeVocab()}
        model._concat_segments_from_ids = lambda segments: segments[0]
        model._encode_ids_batch = lambda ids: {"input_ids": torch.tensor([[0, 2, 1]])}
        calls = []

        def _fake_forward(ids):
            calls.append(ids)
            logits = torch.tensor([[[4.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0]]])
            return {"score": torch.tensor([0.5]), "logits": logits}

        model._unified_forward_from_ids = _fake_forward
        model._decode_xcomet_spans = lambda probs, input_ids, offsets, **kwargs: [
            [{"severity": "minor", "start": 0, "end": 3, "confidence": 1.0, "text": "bad"}]
        ]

        result = model.predict_xcomet(
            [{"mt_ids": [0, 2, 1], "src_ids": [0, 2, 1], "ref_ids": [0, 2, 1], "mt_offsets": [(0, 0), (0, 3), (0, 0)]}]
        )

        self.assertEqual(len(calls), 3)
        self.assertAlmostEqual(result["scores"][0], 0.5 * 0.78 + (24 / 25) * 0.22, places=6)
        self.assertEqual(result["metadata"]["error_spans"][0][0]["severity"], "minor")


class TestDtypePropagation(unittest.TestCase):
    def test_feed_forward_casts_inputs_to_bfloat16_parameters(self):
        feed_forward = FeedForward(in_dim=2, hidden_sizes=[4], out_dim=1).to(torch.bfloat16)

        output = feed_forward(torch.ones(1, 2, dtype=torch.float32))

        self.assertEqual(output.dtype, torch.bfloat16)

    def _make_scorer(self, compute_dtype):
        from unittest.mock import MagicMock

        model = FakeModel(scores=[0.0])
        # FakeModel has no parameters(); stub one so the startup log line works.
        model.parameters = lambda: iter([torch.zeros(1, dtype=torch.float32)])
        model.to = MagicMock()
        config = SimpleNamespace(
            src=None,
            tgt=None,
            output=None,
            batch_size=512,
            compute_dtype=compute_dtype,
            report_time=False,
            with_score=True,
        )
        scorer = EncoderScorer(model, model.vocabs, config, model_config=SimpleNamespace())
        return model, scorer

    def test_requires_with_score(self):
        model = FakeModel(scores=[0.1])
        with tempfile.TemporaryDirectory() as tmp:
            src_path = os.path.join(tmp, "src.txt")
            tgt_path = os.path.join(tmp, "tgt.txt")
            out_path = os.path.join(tmp, "scores.txt")
            _write(src_path, ["a"])
            _write(tgt_path, ["b"])
            config = SimpleNamespace(src=src_path, tgt=tgt_path, output=out_path, batch_size=2, report_time=False)
            scorer = EncoderScorer(model, model.vocabs, config, model_config=SimpleNamespace())
            with self.assertRaisesRegex(ValueError, "--with_score"):
                scorer._predict(
                    infer_iter=None, transforms={"sentencepiece": FakeTokenizer()}, attn_debug=False, align_debug=False
                )

    def test_fp16_calls_model_to(self):
        model, _ = self._make_scorer(torch.float16)
        model.to.assert_called_once_with(torch.float16)

    def test_bf16_calls_model_to(self):
        model, _ = self._make_scorer(torch.bfloat16)
        model.to.assert_called_once_with(torch.bfloat16)

    def test_fp32_does_not_call_to(self):
        model, _ = self._make_scorer(torch.float32)
        model.to.assert_not_called()

    def test_no_dtype_does_not_call_to(self):
        model, _ = self._make_scorer(None)
        model.to.assert_not_called()


class TestDispatch(unittest.TestCase):
    def test_get_infer_class_routes_scoring_architecture(self):
        from eole.predict import get_infer_class
        from eole.predict.encoder_scorer import EncoderScorer as Cls

        cfg = SimpleNamespace(architecture="transformer_encoder_scorer", decoder=None, encoder=SimpleNamespace())
        self.assertIs(get_infer_class(cfg), Cls)

    def test_get_infer_class_legacy_encoder_unchanged(self):
        from eole.predict import get_infer_class, Encoder

        cfg = SimpleNamespace(architecture="transformer", decoder=None, encoder=SimpleNamespace(encoder_type="text"))
        self.assertIs(get_infer_class(cfg), Encoder)


if __name__ == "__main__":
    unittest.main()
