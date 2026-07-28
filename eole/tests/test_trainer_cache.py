import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from eole.trainer import Trainer, TrainerConfig
from eole.utils.misc import clear_gpu_cache


class TestTrainerCacheClearing(unittest.TestCase):
    def test_empty_cache_steps_disabled_does_not_clear_cache(self):
        trainer = object.__new__(Trainer)
        trainer.config = TrainerConfig(empty_cache_steps=0)

        with patch("eole.trainer.clear_gpu_cache") as clear_cache:
            trainer._maybe_clear_device_cache(step=1)

        clear_cache.assert_not_called()

    def test_empty_cache_steps_interval_mismatch_does_not_clear_cache(self):
        trainer = object.__new__(Trainer)
        trainer.config = TrainerConfig(empty_cache_steps=10)

        with patch("eole.trainer.clear_gpu_cache") as clear_cache:
            trainer._maybe_clear_device_cache(step=9)

        clear_cache.assert_not_called()

    def test_empty_cache_steps_interval_match_clears_cache(self):
        trainer = object.__new__(Trainer)
        trainer.config = TrainerConfig(empty_cache_steps=10)

        with patch("eole.trainer.clear_gpu_cache") as clear_cache:
            trainer._maybe_clear_device_cache(step=10)

        clear_cache.assert_called_once_with()

    def test_training_loop_does_not_clear_cache_when_optimizer_step_is_skipped(self):
        trainer = object.__new__(Trainer)
        trainer.config = TrainerConfig(empty_cache_steps=10, average_decay=0, n_gpu=0)
        trainer.optim = SimpleNamespace(training_step=10, learning_rate=Mock(return_value=0.1))
        trainer.scoring_preparator = SimpleNamespace(transforms=None)
        trainer.model_saver = None
        trainer.report_manager = None
        trainer.earlystopper = None

        trainer._accum_batches = Mock(return_value=iter([([], 1)]))
        trainer._update_scheduled_params = Mock()
        trainer._train_step = Mock()
        trainer._maybe_clear_device_cache = Mock()

        with patch("eole.trainer.clear_gpu_cache"):
            trainer.train(train_iter=[], train_steps=10, valid_steps=1000)

        trainer._maybe_clear_device_cache.assert_not_called()

    def test_empty_cache_steps_clears_cuda_cache_when_available(self):
        torch_mock = SimpleNamespace(
            cuda=SimpleNamespace(is_available=Mock(return_value=True), empty_cache=Mock()),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=Mock(return_value=False))),
            mps=SimpleNamespace(empty_cache=Mock()),
        )

        with patch("eole.utils.misc.torch", torch_mock):
            clear_gpu_cache()

        torch_mock.cuda.empty_cache.assert_called_once_with()
        torch_mock.mps.empty_cache.assert_not_called()

    def test_empty_cache_steps_clears_mps_cache_and_graph_cache_when_available(self):
        torch_mock = SimpleNamespace(
            cuda=SimpleNamespace(is_available=Mock(return_value=False), empty_cache=Mock()),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=Mock(return_value=True))),
            mps=SimpleNamespace(empty_cache=Mock(), clear_graph_cache=Mock()),
        )

        with patch("eole.utils.misc.torch", torch_mock):
            clear_gpu_cache()

        torch_mock.mps.empty_cache.assert_called_once_with()
        torch_mock.mps.clear_graph_cache.assert_called_once_with()

    def test_empty_cache_steps_clears_mps_cache_without_graph_cache_when_unavailable(self):
        torch_mock = SimpleNamespace(
            cuda=SimpleNamespace(is_available=Mock(return_value=False), empty_cache=Mock()),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=Mock(return_value=True))),
            mps=SimpleNamespace(empty_cache=Mock()),
        )

        with patch("eole.utils.misc.torch", torch_mock):
            clear_gpu_cache()

        torch_mock.mps.empty_cache.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
