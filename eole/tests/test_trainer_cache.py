import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from eole.trainer import Trainer, TrainerConfig


class TestTrainerCacheClearing(unittest.TestCase):
    def test_empty_cache_steps_disabled_does_not_clear_cache(self):
        trainer = object.__new__(Trainer)
        trainer.config = TrainerConfig(empty_cache_steps=0)

        with patch("torch.cuda.empty_cache") as cuda_empty_cache, patch("torch.mps.empty_cache") as mps_empty_cache:
            trainer._maybe_clear_device_cache(step=1)

        cuda_empty_cache.assert_not_called()
        mps_empty_cache.assert_not_called()

    def test_empty_cache_steps_interval_mismatch_does_not_clear_cache(self):
        trainer = object.__new__(Trainer)
        trainer.config = TrainerConfig(empty_cache_steps=10)

        with patch("torch.cuda.empty_cache") as cuda_empty_cache, patch("torch.mps.empty_cache") as mps_empty_cache:
            trainer._maybe_clear_device_cache(step=9)

        cuda_empty_cache.assert_not_called()
        mps_empty_cache.assert_not_called()

    def test_empty_cache_steps_clears_cuda_cache_when_available(self):
        trainer = object.__new__(Trainer)
        trainer.config = TrainerConfig(empty_cache_steps=10)

        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.backends.mps.is_available", return_value=False
        ), patch("torch.cuda.empty_cache") as cuda_empty_cache, patch("torch.mps.empty_cache") as mps_empty_cache:
            trainer._maybe_clear_device_cache(step=10)

        cuda_empty_cache.assert_called_once_with()
        mps_empty_cache.assert_not_called()

    def test_empty_cache_steps_clears_mps_cache_and_graph_cache_when_available(self):
        trainer = object.__new__(Trainer)
        trainer.config = TrainerConfig(empty_cache_steps=10)

        with patch("torch.cuda.is_available", return_value=False), patch(
            "torch.backends.mps.is_available", return_value=True
        ), patch("torch.mps.empty_cache") as mps_empty_cache, patch(
            "torch.mps.clear_graph_cache", create=True
        ) as clear_graph_cache:
            trainer._maybe_clear_device_cache(step=10)

        mps_empty_cache.assert_called_once_with()
        clear_graph_cache.assert_called_once_with()

    def test_empty_cache_steps_clears_mps_cache_without_graph_cache_when_unavailable(self):
        trainer = object.__new__(Trainer)
        trainer.config = TrainerConfig(empty_cache_steps=10)
        mps = SimpleNamespace(empty_cache=Mock())

        with patch("torch.cuda.is_available", return_value=False), patch(
            "torch.backends.mps.is_available", return_value=True
        ), patch("torch.mps", mps):
            trainer._maybe_clear_device_cache(step=10)

        mps.empty_cache.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
