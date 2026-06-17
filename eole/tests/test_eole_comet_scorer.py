import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from eole.scorers.eole_comet import EoleCometScorer, EoleCometKiwiScorer


class FakeRuntime:
    def __init__(self, requires_reference, scores):
        self.requires_reference = requires_reference
        self._scores = scores

    def predict_scores(self, rows):
        import torch

        return torch.tensor(self._scores[: len(rows)], dtype=torch.float32)


class TestEoleCometScorers(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
