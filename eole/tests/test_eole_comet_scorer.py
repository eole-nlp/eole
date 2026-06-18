import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from eole.scorers.eole_comet import EoleCometScorer, EoleCometKiwiScorer, EoleXCometScorer


class FakeRuntime:
    def __init__(self, requires_reference, scores, class_identifier="regression_metric"):
        self.requires_reference = requires_reference
        self._scores = scores
        self.class_identifier = class_identifier

    def predict_scores(self, rows):
        import torch

        return torch.tensor(self._scores[: len(rows)], dtype=torch.float32)


class TestEoleCometScorers(unittest.TestCase):
    def _compute_score_with_runtime(self, scorer_cls, runtime, preds=None, refs=None, srcs=None):
        scorer = scorer_cls(SimpleNamespace(comet_model="dummy", comet_batch_size=8))
        preds = preds or ["a"]
        refs = refs or ["r"]
        srcs = srcs or ["s"]
        encoded = [[0, 1] for _ in preds]
        encode_calls = [encoded, encoded]
        if runtime.requires_reference:
            encode_calls.append(encoded)
        with (
            patch("eole.scorers.eole_comet._resolve_model_dir", return_value="dummy"),
            patch("eole.scorers.eole_comet.EncoderScoringModel.from_model_dir", return_value=runtime),
            patch.object(scorer_cls, "_build_sp_transform", return_value=object()),
            patch.object(scorer_cls, "_encode_texts", side_effect=encode_calls),
        ):
            return scorer.compute_score(preds, refs, srcs)

    def test_eole_comet_requires_references(self):
        scorer = EoleCometScorer(SimpleNamespace(comet_model="dummy", comet_batch_size=4))
        fake_runtime = FakeRuntime(requires_reference=True, scores=[0.5, 0.6])
        with (
            patch("eole.scorers.eole_comet._resolve_model_dir", return_value="dummy"),
            patch("eole.scorers.eole_comet.EncoderScoringModel.from_model_dir", return_value=fake_runtime),
            patch.object(EoleCometScorer, "_build_sp_transform", return_value=object()),
            patch.object(
                EoleCometScorer, "_encode_texts", side_effect=[[[0, 1], [0, 1]], [[0, 1], [0, 1]], [[0, 1], [0, 1]]]
            ),
        ):
            with self.assertRaises(ValueError):
                scorer.compute_score(["a", "b"], None, ["s1", "s2"])

    def test_eole_comet_kiwi_rejects_reference_model_family(self):
        scorer = EoleCometKiwiScorer(SimpleNamespace(comet_model="dummy", comet_batch_size=4))
        fake_runtime = FakeRuntime(requires_reference=True, scores=[0.5])
        with (
            patch("eole.scorers.eole_comet._resolve_model_dir", return_value="dummy"),
            patch("eole.scorers.eole_comet.EncoderScoringModel.from_model_dir", return_value=fake_runtime),
            patch.object(EoleCometKiwiScorer, "_build_sp_transform", return_value=object()),
            patch.object(EoleCometKiwiScorer, "_encode_texts", side_effect=[[[0, 1]], [[0, 1]]]),
        ):
            with self.assertRaises(ValueError):
                scorer.compute_score(["a"], ["r"], ["s"])

    def test_eole_comet_aggregation(self):
        scorer = EoleCometScorer(SimpleNamespace(comet_model="dummy", comet_batch_size=8))
        fake_runtime = FakeRuntime(requires_reference=True, scores=[0.2, 0.4, 0.8])
        with (
            patch("eole.scorers.eole_comet._resolve_model_dir", return_value="dummy"),
            patch("eole.scorers.eole_comet.EncoderScoringModel.from_model_dir", return_value=fake_runtime),
            patch.object(EoleCometScorer, "_build_sp_transform", return_value=object()),
            patch.object(
                EoleCometScorer,
                "_encode_texts",
                side_effect=[[[0, 1], [0, 1], [0, 1]], [[0, 1], [0, 1], [0, 1]], [[0, 1], [0, 1], [0, 1]]],
            ),
        ):
            score = scorer.compute_score(["a", "b", "c"], ["r1", "r2", "r3"], ["s1", "s2", "s3"])
        self.assertAlmostEqual(score, (0.2 + 0.4 + 0.8) / 3, places=6)

    def test_eole_comet_accepts_regression_model_family(self):
        fake_runtime = FakeRuntime(requires_reference=True, scores=[0.2], class_identifier="regression_metric")

        score = self._compute_score_with_runtime(EoleCometScorer, fake_runtime)

        self.assertAlmostEqual(score, 0.2, places=6)

    def test_eole_comet_rejects_xcomet_model_family(self):
        fake_runtime = FakeRuntime(requires_reference=True, scores=[0.2], class_identifier="xcomet_metric")

        with self.assertRaisesRegex(ValueError, "class_identifier='regression_metric'"):
            self._compute_score_with_runtime(EoleCometScorer, fake_runtime)

    def test_eole_comet_kiwi_accepts_referenceless_regression_model_family(self):
        fake_runtime = FakeRuntime(
            requires_reference=False, scores=[0.2], class_identifier="referenceless_regression_metric"
        )

        score = self._compute_score_with_runtime(EoleCometKiwiScorer, fake_runtime)

        self.assertAlmostEqual(score, 0.2, places=6)

    def test_eole_comet_kiwi_rejects_xcomet_model_family(self):
        fake_runtime = FakeRuntime(requires_reference=False, scores=[0.2], class_identifier="xcomet_metric")

        with self.assertRaisesRegex(ValueError, "class_identifier='referenceless_regression_metric'"):
            self._compute_score_with_runtime(EoleCometKiwiScorer, fake_runtime)

    def test_eole_comet_kiwi_rejects_regular_comet_model_family(self):
        fake_runtime = FakeRuntime(
            requires_reference=False,
            scores=[0.2],
            class_identifier="regression_metric",
        )

        with self.assertRaisesRegex(ValueError, "class_identifier='referenceless_regression_metric'"):
            self._compute_score_with_runtime(EoleCometKiwiScorer, fake_runtime)

    def test_eole_comet_uses_configured_device(self):
        scorer = EoleCometScorer(SimpleNamespace(comet_model="dummy", comet_batch_size=8, comet_device="cpu"))
        fake_runtime = FakeRuntime(requires_reference=True, scores=[0.2])
        with (
            patch("eole.scorers.eole_comet._resolve_model_dir", return_value="dummy"),
            patch("eole.scorers.eole_comet.EncoderScoringModel.from_model_dir", return_value=fake_runtime) as load_mock,
            patch.object(EoleCometScorer, "_build_sp_transform", return_value=object()),
            patch.object(EoleCometScorer, "_encode_texts", side_effect=[[[0, 1]], [[0, 1]], [[0, 1]]]),
        ):
            scorer.compute_score(["a"], ["r"], ["s"])
        load_mock.assert_called_once_with("dummy", device=torch.device("cpu"))

    def test_eole_comet_auto_selects_mps(self):
        scorer = EoleCometScorer(SimpleNamespace(comet_model="dummy", comet_batch_size=8))
        fake_runtime = FakeRuntime(requires_reference=True, scores=[0.2])
        with (
            patch("eole.scorers.eole_comet._resolve_model_dir", return_value="dummy"),
            patch("eole.scorers.eole_comet.get_device", return_value=torch.device("mps")),
            patch("eole.scorers.eole_comet.EncoderScoringModel.from_model_dir", return_value=fake_runtime) as load_mock,
            patch.object(EoleCometScorer, "_build_sp_transform", return_value=object()),
            patch.object(EoleCometScorer, "_encode_texts", side_effect=[[[0, 1]], [[0, 1]], [[0, 1]]]),
        ):
            scorer.compute_score(["a"], ["r"], ["s"])
        load_mock.assert_called_once_with("dummy", device=torch.device("mps"))

    def test_eole_xcomet_accepts_xcomet_model_family(self):
        scorer = EoleXCometScorer(SimpleNamespace(comet_model="dummy", comet_batch_size=8))
        fake_runtime = FakeRuntime(requires_reference=True, scores=[0.2, 0.6], class_identifier="xcomet_metric")
        with (
            patch("eole.scorers.eole_comet._resolve_model_dir", return_value="dummy"),
            patch("eole.scorers.eole_comet.EncoderScoringModel.from_model_dir", return_value=fake_runtime),
            patch.object(EoleXCometScorer, "_build_sp_transform", return_value=object()),
            patch.object(
                EoleXCometScorer, "_encode_texts", side_effect=[[[0, 1], [0, 1]], [[0, 1], [0, 1]], [[0, 1], [0, 1]]]
            ),
        ):
            score = scorer.compute_score(["a", "b"], ["r1", "r2"], ["s1", "s2"])

        self.assertAlmostEqual(score, 0.4, places=6)

    def test_eole_xcomet_rejects_regular_comet_model_family(self):
        scorer = EoleXCometScorer(SimpleNamespace(comet_model="dummy", comet_batch_size=8))
        fake_runtime = FakeRuntime(requires_reference=True, scores=[0.2], class_identifier="regression_metric")
        with (
            patch("eole.scorers.eole_comet._resolve_model_dir", return_value="dummy"),
            patch("eole.scorers.eole_comet.EncoderScoringModel.from_model_dir", return_value=fake_runtime),
        ):
            with self.assertRaisesRegex(ValueError, "class_identifier='xcomet_metric'"):
                scorer.compute_score(["a"], ["r"], ["s"])


if __name__ == "__main__":
    unittest.main()
