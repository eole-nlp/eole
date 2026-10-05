import unittest

import torch

from eole.config.training import TrainingConfig
from eole.utils.rl_loss import REINFORCELossCompute, build_rl_loss_compute


class DummyVocab(dict):
    def __getitem__(self, key):
        if isinstance(key, str) and not dict.__contains__(self, key):
            return 1  # pad id
        return dict.__getitem__(self, key)


class TestREINFORCELossCompute(unittest.TestCase):
    def _loss_compute(self, baseline="batch_mean", kl_coef=0.0):
        return REINFORCELossCompute(padding_idx=1, baseline=baseline, kl_coef=kl_coef)

    def test_invalid_baseline_raises(self):
        with self.assertRaises(ValueError):
            REINFORCELossCompute(padding_idx=1, baseline="not-a-baseline")

    def test_reward_magnitude_scales_the_loss(self):
        # With a 2-sample batch and the "batch_mean" baseline, doubling the
        # spread between rewards should scale the resulting (non-zero) loss.
        token_log_probs = torch.tensor([[-0.1, -0.2, -0.3], [-0.5, -0.4, 0.0]])
        padding_mask = torch.tensor([[True, True, True], [True, True, False]])

        lc = self._loss_compute()
        loss_r1, _ = lc.compute_policy_loss(token_log_probs.clone(), padding_mask, torch.tensor([1.0, 0.0]))
        loss_r2, _ = lc.compute_policy_loss(token_log_probs.clone(), padding_mask, torch.tensor([2.0, 0.0]))
        self.assertAlmostEqual(loss_r2.item(), 2 * loss_r1.item(), places=5)

    def test_batch_mean_baseline_zero_reward_signal_when_equal(self):
        token_log_probs = torch.tensor([[-0.1, -0.2], [-0.3, -0.1]], requires_grad=True)
        padding_mask = torch.tensor([[True, True], [True, True]])
        rewards = torch.tensor([1.0, 1.0])

        lc = self._loss_compute(baseline="batch_mean")
        loss, stats = lc.compute_policy_loss(token_log_probs, padding_mask, rewards)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)
        self.assertAlmostEqual(stats["reward_mean"], 1.0)

    def test_padding_positions_are_ignored(self):
        # 2-sample batch (matching `rewards`); the huge negative log-prob at
        # the padded position (index 2) must not blow up the loss.
        token_log_probs = torch.tensor([[-0.1, -0.2, -100.0], [-0.1, -0.2, -100.0]], requires_grad=True)
        padding_mask = torch.tensor([[True, True, False], [True, True, False]])
        rewards = torch.tensor([1.0, 0.0])

        lc = self._loss_compute()
        loss, _ = lc.compute_policy_loss(token_log_probs, padding_mask, rewards)
        self.assertTrue(torch.isfinite(loss))

    def test_kl_penalty_increases_loss_when_policies_diverge(self):
        token_log_probs = torch.tensor([[-0.1, -0.2]], requires_grad=True)
        ref_log_probs = torch.tensor([[-1.1, -1.2]])
        padding_mask = torch.tensor([[True, True]])
        rewards = torch.tensor([1.0])

        lc_no_kl = self._loss_compute(kl_coef=0.0)
        lc_with_kl = self._loss_compute(kl_coef=1.0)

        loss_no_kl, stats_no_kl = lc_no_kl.compute_policy_loss(
            token_log_probs, padding_mask, rewards, ref_token_log_probs=ref_log_probs
        )
        loss_with_kl, stats_with_kl = lc_with_kl.compute_policy_loss(
            token_log_probs, padding_mask, rewards, ref_token_log_probs=ref_log_probs
        )
        self.assertNotIn("kl", stats_no_kl)
        self.assertIn("kl", stats_with_kl)
        self.assertGreater(loss_with_kl.item(), loss_no_kl.item())

    def test_backward_populates_gradients(self):
        token_log_probs = torch.tensor([[-0.1, -0.2, -0.3], [-0.5, -0.4, 0.0]], requires_grad=True)
        padding_mask = torch.tensor([[True, True, True], [True, True, False]])
        rewards = torch.tensor([1.0, 0.0])

        lc = self._loss_compute()
        loss, _ = lc.compute_policy_loss(token_log_probs, padding_mask, rewards)
        loss.backward()
        self.assertIsNotNone(token_log_probs.grad)
        self.assertTrue(torch.any(token_log_probs.grad != 0))


class TestBuildRLLossCompute(unittest.TestCase):
    def test_reinforce_algorithm_builds_reinforce_loss(self):
        vocabs = {"specials": {}, "tgt": DummyVocab()}
        config = _FakeConfig(rl_algorithm="reinforce", rl_baseline="none", rl_kl_coef=0.0)
        loss_compute = build_rl_loss_compute(config, vocabs)
        self.assertIsInstance(loss_compute, REINFORCELossCompute)
        self.assertEqual(loss_compute.baseline, "none")

    def test_dpo_algorithm_raises_not_implemented(self):
        vocabs = {"specials": {}, "tgt": DummyVocab()}
        config = _FakeConfig(rl_algorithm="dpo", rl_baseline="batch_mean", rl_kl_coef=0.0)
        with self.assertRaises(NotImplementedError):
            build_rl_loss_compute(config, vocabs)


class _FakeTraining:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeConfig:
    def __init__(self, **kwargs):
        self.training = _FakeTraining(**kwargs)


class TestTrainingConfigRLValidation(unittest.TestCase):
    def test_default_disables_rl(self):
        config = TrainingConfig()
        self.assertIsNone(config.rl_algorithm)

    def test_unimplemented_algorithm_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            TrainingConfig(rl_algorithm="dpo")

    def test_reinforce_requires_reward_metric(self):
        with self.assertRaises(ValueError):
            TrainingConfig(rl_algorithm="reinforce")

    def test_reinforce_with_kl_requires_reference_model(self):
        with self.assertRaises(ValueError):
            TrainingConfig(rl_algorithm="reinforce", rl_reward_metric="BLEU", rl_kl_coef=0.1)

    def test_valid_reinforce_config(self):
        config = TrainingConfig(
            rl_algorithm="reinforce",
            rl_reward_metric="BLEU",
            rl_kl_coef=0.1,
            rl_reference_model="/tmp/ref_model.pt",
        )
        self.assertEqual(config.rl_algorithm, "reinforce")


if __name__ == "__main__":
    unittest.main()
