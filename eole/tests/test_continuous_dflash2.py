"""CPU checks for DFlash2 continuous dispatch and speculation gating."""

from types import SimpleNamespace
import unittest

from eole.predict.continuous import ContinuousGreedy
from eole.predict.continuous_dflash2 import ContinuousDFlash2
from eole.predict.continuous_mtp import ContinuousMTP
from eole.server.continuous import ContinuousService


class ContinuousDFlash2UnitTests(unittest.TestCase):
    def test_speculation_mode_prefers_dflash2_over_mtp_flag(self):
        # Configuring dflash2 forces self_speculative_decoding on the predictor,
        # so the drafter adapter must win over the MTP flag.
        predictor = SimpleNamespace(self_speculative_decoding=True, dflash2=object())
        self.assertEqual(ContinuousGreedy._speculation_mode(predictor), "dflash2")
        self.assertEqual(ContinuousGreedy._speculation_mode(SimpleNamespace(self_speculative_decoding=True)), "mtp")
        self.assertIsNone(ContinuousGreedy._speculation_mode(SimpleNamespace()))

    def test_service_dispatches_continuous_dflash2_when_drafter_configured(self):
        engine = SimpleNamespace(predictor=SimpleNamespace(self_speculative_decoding=True, dflash2=object()))
        self.assertIs(ContinuousService(engine).runner_factory, ContinuousDFlash2)
        engine = SimpleNamespace(predictor=SimpleNamespace(self_speculative_decoding=True))
        self.assertIs(ContinuousService(engine).runner_factory, ContinuousMTP)
        self.assertIs(ContinuousService(SimpleNamespace(predictor=SimpleNamespace())).runner_factory, ContinuousGreedy)

    def test_runner_classes_declare_compatible_speculation(self):
        self.assertIsNone(ContinuousGreedy.speculation)
        self.assertEqual(ContinuousMTP.speculation, "mtp")
        self.assertEqual(ContinuousDFlash2.speculation, "dflash2")

    def test_mismatched_speculation_is_rejected(self):
        predictor = SimpleNamespace(self_speculative_decoding=True, dflash2=object())
        # A predictor drafting with dflash2 may only enter runners that declare dflash2.
        mode = ContinuousGreedy._speculation_mode(predictor)
        self.assertEqual(mode, ContinuousDFlash2.speculation)
        self.assertNotEqual(mode, ContinuousMTP.speculation)


if __name__ == "__main__":
    unittest.main()
