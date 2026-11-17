"""CPU checks for speculative prefix selection and service configuration."""

from types import SimpleNamespace
import unittest

from eole.predict.continuous_mtp import ContinuousMTP, verified_prefix
from eole.server.continuous import ContinuousService


class ContinuousMTPTests(unittest.TestCase):
    def test_independent_prefix_lengths(self):
        targets = [4, 5, 6, 7]
        for drafts, expected in (([0, 0, 0], (0, 1)), ([4, 0, 0], (1, 2)), ([4, 5, 6], (3, 4))):
            self.assertEqual(verified_prefix(targets, drafts, 10, set()), expected)

    def test_eos_and_output_budget_truncate_verified_prefix(self):
        self.assertEqual(verified_prefix([4, 5, 6, 7], [4, 5, 6], 2, set()), (3, 2))
        self.assertEqual(verified_prefix([4, 5, 6, 7], [4, 5, 6], 4, {5}), (3, 2))
        self.assertEqual(verified_prefix([4, 5, 6, 7], [4, 5, 6], 4, {4}), (3, 1))

    def test_existing_model_setting_selects_mtp_without_a_second_flag(self):
        engine = SimpleNamespace(predictor=SimpleNamespace(self_speculative_decoding=True))
        self.assertIs(ContinuousService(engine).runner_factory, ContinuousMTP)
        custom = object()
        self.assertIs(ContinuousService(engine, runner_factory=custom).runner_factory, custom)

    def test_statistics_survive_service_completion(self):
        stats = {}
        request = SimpleNamespace(stats=stats, request_id="a")
        runner = SimpleNamespace(statistics=lambda identifier: {"speculation_mode": "mtp", "mtp_cycles": 3})
        ContinuousService._collect_statistics(runner, request)
        self.assertEqual(stats, {"speculation_mode": "mtp", "mtp_cycles": 3})
