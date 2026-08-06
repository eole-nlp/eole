import unittest
from types import SimpleNamespace
from unittest.mock import patch

from eole.config.models import CustomModelConfig
from eole.config.run import TrainConfig
from eole.scorers import AVAILABLE_SCORERS, load_scorer_modules
from eole.utils import Statistics
from eole.utils.earlystopping import scorers_from_config


class DirectionalScorer:
    higher_is_better = True


class LowerIsBetterScorer:
    higher_is_better = False


class MissingDirectionScorer:
    pass


class TestEarlyStoppingScorers(unittest.TestCase):
    def _config(self, criteria, valid_metrics=None):
        return SimpleNamespace(early_stopping_criteria=criteria, valid_metrics=valid_metrics or [])

    def test_string_criterion_is_not_split_into_characters(self):
        scorers = scorers_from_config(self._config("EOLE-COMET", ["EOLE-COMET"]))

        self.assertEqual(len(scorers), 1)
        self.assertEqual(scorers[0].name, "EOLE-COMET")

    def test_real_train_config_uses_top_level_valid_metrics(self):
        config = TrainConfig(
            src_vocab="dummy",
            tgt_vocab="dummy",
            data={},
            model=CustomModelConfig(),
            valid_metrics=["EOLE-COMET"],
            training={"early_stopping_criteria": "EOLE-COMET"},
        )

        scorers = scorers_from_config(config)

        self.assertEqual(len(scorers), 1)
        self.assertEqual(scorers[0].name, "EOLE-COMET")

    def test_nested_training_config_without_valid_metrics_still_works(self):
        config = SimpleNamespace(early_stopping_criteria="EOLE-COMET")

        scorers = scorers_from_config(config)

        self.assertEqual(len(scorers), 1)
        self.assertEqual(scorers[0].name, "EOLE-COMET")

    def test_comet_family_metrics_are_higher_is_better(self):
        for metric in ("COMET", "COMET-KIWI", "EOLE-COMET", "EOLE-COMET-KIWI", "EOLE-XCOMET"):
            scorer = scorers_from_config(self._config(metric, [metric]))[0]

            self.assertTrue(scorer.is_improving(Statistics(computed_metrics={metric: 0.5})))
            scorer.update(Statistics(computed_metrics={metric: 0.5}))
            self.assertTrue(scorer.is_improving(Statistics(computed_metrics={metric: 0.6})))
            self.assertTrue(scorer.is_decreasing(Statistics(computed_metrics={metric: 0.4})))

    def test_metricx_metrics_are_lower_is_better(self):
        for metric in ("EOLE-METRICX", "EOLE-METRICX-QE"):
            scorer = scorers_from_config(self._config(metric, [metric]))[0]

            self.assertTrue(scorer.is_improving(Statistics(computed_metrics={metric: 5.0})))
            scorer.update(Statistics(computed_metrics={metric: 5.0}))
            self.assertTrue(scorer.is_improving(Statistics(computed_metrics={metric: 4.0})))
            self.assertTrue(scorer.is_decreasing(Statistics(computed_metrics={metric: 6.0})))

    def test_bleu_bleu_zh_chrf_and_ter_use_registered_metric_directions(self):
        bleu = scorers_from_config(self._config("BLEU", ["BLEU"]))[0]
        bleu_zh = scorers_from_config(self._config("BLEU_zh", ["BLEU_zh"]))[0]
        chrf = scorers_from_config(self._config("CHRF", ["CHRF"]))[0]
        ter = scorers_from_config(self._config("TER", ["TER"]))[0]

        self.assertTrue(bleu.is_improving(Statistics(computed_metrics={"BLEU": 50.0})))
        bleu.update(Statistics(computed_metrics={"BLEU": 50.0}))
        self.assertTrue(bleu.is_improving(Statistics(computed_metrics={"BLEU": 60.0})))
        self.assertTrue(bleu.is_decreasing(Statistics(computed_metrics={"BLEU": 40.0})))

        self.assertTrue(bleu_zh.is_improving(Statistics(computed_metrics={"BLEU_zh": 50.0})))
        bleu_zh.update(Statistics(computed_metrics={"BLEU_zh": 50.0}))
        self.assertTrue(bleu_zh.is_improving(Statistics(computed_metrics={"BLEU_zh": 60.0})))
        self.assertTrue(bleu_zh.is_decreasing(Statistics(computed_metrics={"BLEU_zh": 40.0})))

        self.assertTrue(chrf.is_improving(Statistics(computed_metrics={"CHRF": 50.0})))
        chrf.update(Statistics(computed_metrics={"CHRF": 50.0}))
        self.assertTrue(chrf.is_improving(Statistics(computed_metrics={"CHRF": 60.0})))
        self.assertTrue(chrf.is_decreasing(Statistics(computed_metrics={"CHRF": 40.0})))

        self.assertTrue(ter.is_improving(Statistics(computed_metrics={"TER": 50.0})))
        ter.update(Statistics(computed_metrics={"TER": 50.0}))
        self.assertTrue(ter.is_improving(Statistics(computed_metrics={"TER": 40.0})))
        self.assertTrue(ter.is_decreasing(Statistics(computed_metrics={"TER": 60.0})))

    def test_list_criteria_are_supported(self):
        scorers = scorers_from_config(
            self._config(
                ["EOLE-COMET-KIWI", "EOLE-METRICX-QE"],
                ["EOLE-COMET-KIWI", "EOLE-METRICX-QE"],
            )
        )

        self.assertEqual([scorer.name for scorer in scorers], ["EOLE-COMET-KIWI", "EOLE-METRICX-QE"])

    def test_computed_metric_criteria_must_be_valid_metrics(self):
        for metric in (
            "BLEU",
            "CHRF",
            "TER",
            "COMET",
            "COMET-KIWI",
            "EOLE-COMET",
            "EOLE-COMET-KIWI",
            "EOLE-XCOMET",
            "EOLE-METRICX",
            "EOLE-METRICX-QE",
        ):
            with self.subTest(metric=metric):
                with self.assertRaisesRegex(ValueError, f"Early stopping criterion {metric} requires it"):
                    scorers_from_config(self._config(metric))

    def test_registered_external_metric_uses_declared_direction(self):
        with patch.dict(AVAILABLE_SCORERS, {"PRIVATE-METRIC": DirectionalScorer}):
            scorer = scorers_from_config(self._config("PRIVATE-METRIC", ["PRIVATE-METRIC"]))[0]

        self.assertTrue(scorer.is_improving(Statistics(computed_metrics={"PRIVATE-METRIC": 0.5})))
        scorer.update(Statistics(computed_metrics={"PRIVATE-METRIC": 0.5}))
        self.assertTrue(scorer.is_improving(Statistics(computed_metrics={"PRIVATE-METRIC": 0.6})))
        self.assertTrue(scorer.is_decreasing(Statistics(computed_metrics={"PRIVATE-METRIC": 0.4})))

    def test_registered_external_metric_can_be_lower_is_better(self):
        with patch.dict(AVAILABLE_SCORERS, {"PRIVATE-LOWER": LowerIsBetterScorer}):
            scorer = scorers_from_config(self._config("PRIVATE-LOWER", ["PRIVATE-LOWER"]))[0]

        self.assertTrue(scorer.is_improving(Statistics(computed_metrics={"PRIVATE-LOWER": 5.0})))
        scorer.update(Statistics(computed_metrics={"PRIVATE-LOWER": 5.0}))
        self.assertTrue(scorer.is_improving(Statistics(computed_metrics={"PRIVATE-LOWER": 4.0})))
        self.assertTrue(scorer.is_decreasing(Statistics(computed_metrics={"PRIVATE-LOWER": 6.0})))

    def test_registered_external_metric_requires_valid_metrics(self):
        with patch.dict(AVAILABLE_SCORERS, {"PRIVATE-METRIC": DirectionalScorer}):
            with self.assertRaisesRegex(ValueError, "Early stopping criterion PRIVATE-METRIC requires it"):
                scorers_from_config(self._config("PRIVATE-METRIC"))

    def test_registered_external_metric_reports_valid_metrics_before_direction(self):
        with patch.dict(AVAILABLE_SCORERS, {"PRIVATE-METRIC": MissingDirectionScorer}):
            with self.assertRaisesRegex(ValueError, "Early stopping criterion PRIVATE-METRIC requires it"):
                scorers_from_config(self._config("PRIVATE-METRIC"))

    def test_registered_external_metric_requires_direction(self):
        with patch.dict(AVAILABLE_SCORERS, {"PRIVATE-METRIC": MissingDirectionScorer}):
            with self.assertRaisesRegex(ValueError, "must define higher_is_better"):
                scorers_from_config(self._config("PRIVATE-METRIC", ["PRIVATE-METRIC"]))

    def test_unknown_criterion_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "Criterion NOPE not found"):
            scorers_from_config(self._config("NOPE"))


class TestExternalScorerModules(unittest.TestCase):
    def test_load_scorer_modules_imports_configured_modules(self):
        with patch("eole.scorers.importlib.import_module") as import_module:
            load_scorer_modules(["private_scorers.unified", "private_scorers.other"])

        self.assertEqual(
            [call.args[0] for call in import_module.call_args_list],
            ["private_scorers.unified", "private_scorers.other"],
        )

    def test_load_scorer_modules_reports_config_field_on_import_error(self):
        with self.assertRaisesRegex(ValueError, "scorer_modules: cannot import 'private_scorers.missing'"):
            load_scorer_modules(["private_scorers.missing"])


if __name__ == "__main__":
    unittest.main()
