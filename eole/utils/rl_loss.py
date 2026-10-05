"""
Loss computations for RL fine-tuning methods (REINFORCE and, in the future,
GRPO/PPO). These build on top of the RL-readiness refactoring
(``ModelOutput``, ``BaseModel.compute_log_probs()``, ``eole.utils.generate_utils``)
to turn on-policy rollouts + reward signals into a policy-gradient loss.

DPO is intentionally *not* implemented here: it is an offline, pairwise
preference method (chosen/rejected completions for the same prompt) and
needs its own data pipeline rather than the on-policy generation used by
REINFORCE/GRPO/PPO. See ``eole.utils.loss`` / ``eole.trainer_rl`` docstrings
for the planned extension points.
"""

import torch.nn as nn

from eole.constants import DefaultTokens


class RLLossCompute(nn.Module):
    """Base class for RL loss computations that turn rollouts + rewards into
    a policy-gradient loss.

    Subclasses (currently only :class:`REINFORCELossCompute`; future
    ``GRPOLossCompute``/``PPOLossCompute``) must implement
    :meth:`compute_policy_loss`.

    Args:
        padding_idx (int): padding index of the target vocab, used to mask
            out padded positions when summing per-token log-probs.
        kl_coef (float): weight of the KL-divergence penalty against a frozen
            reference model's log-probs (0 disables it).
    """

    def __init__(self, padding_idx, kl_coef=0.0):
        super().__init__()
        self.padding_idx = padding_idx
        self.kl_coef = kl_coef

    def sequence_log_probs(self, token_log_probs, padding_mask):
        """Sum per-token log-probs over the sequence (ignoring padding).

        Args:
            token_log_probs: ``(batch, tgt_len)`` per-token log-probs.
            padding_mask: ``(batch, tgt_len)`` boolean mask (True = real token).

        Returns:
            ``(batch,)`` summed log-probs per sequence.
        """
        return (token_log_probs * padding_mask.to(token_log_probs.dtype)).sum(dim=1)

    def compute_kl_penalty(self, token_log_probs, ref_token_log_probs, padding_mask):
        """Per-sequence KL(policy || reference) approximated from log-probs of
        the sampled tokens (i.e. ``log pi(a) - log ref(a)``, averaged over
        non-padding positions).
        """
        diff = (token_log_probs - ref_token_log_probs) * padding_mask.to(token_log_probs.dtype)
        n_tokens = padding_mask.to(token_log_probs.dtype).sum(dim=1).clamp(min=1)
        return diff.sum(dim=1) / n_tokens

    def compute_policy_loss(self, token_log_probs, padding_mask, rewards, ref_token_log_probs=None):
        """Compute the policy-gradient loss.

        Args:
            token_log_probs: ``(batch, tgt_len)`` per-token log-probs of the
                sampled/generated tokens under the current policy.
            padding_mask: ``(batch, tgt_len)`` boolean mask (True = real token).
            rewards: ``(batch,)`` scalar reward per sampled sequence.
            ref_token_log_probs: optional ``(batch, tgt_len)`` per-token
                log-probs of the same tokens under a frozen reference model
                (used for the KL penalty).

        Returns:
            Tuple of ``(loss, stats_dict)``.
        """
        raise NotImplementedError


class REINFORCELossCompute(RLLossCompute):
    """Vanilla REINFORCE (score-function policy-gradient) loss.

    ``loss = -mean((reward - baseline) * sum_t log pi(a_t | s_t))``

    with an optional batch-mean baseline (variance reduction) and an optional
    KL penalty against a frozen reference model's log-probs.

    Args:
        padding_idx (int): padding index of the target vocab.
        baseline (str): ``"batch_mean"`` subtracts the mean reward of the
            batch from each reward (simple variance-reduction baseline
            commonly used with REINFORCE); ``"none"`` uses raw rewards.
        kl_coef (float): weight of the KL penalty (0 disables it).
    """

    def __init__(self, padding_idx, baseline="batch_mean", kl_coef=0.0):
        super().__init__(padding_idx, kl_coef=kl_coef)
        if baseline not in ("batch_mean", "none"):
            raise ValueError(f"Unsupported REINFORCE baseline: {baseline!r} (expected 'batch_mean' or 'none')")
        self.baseline = baseline

    def compute_policy_loss(self, token_log_probs, padding_mask, rewards, ref_token_log_probs=None):
        seq_log_probs = self.sequence_log_probs(token_log_probs, padding_mask)

        rewards = rewards.to(seq_log_probs.dtype)
        if self.baseline == "batch_mean" and rewards.numel() > 1:
            advantage = rewards - rewards.mean()
        else:
            advantage = rewards

        # advantage is a (detached) scalar weight on the log-prob gradient:
        # standard score-function/REINFORCE estimator.
        pg_loss = -(advantage.detach() * seq_log_probs).mean()

        stats = {
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std(unbiased=False).item() if rewards.numel() > 1 else 0.0,
            "pg_loss": pg_loss.item(),
        }

        loss = pg_loss
        if self.kl_coef != 0.0 and ref_token_log_probs is not None:
            kl = self.compute_kl_penalty(token_log_probs, ref_token_log_probs, padding_mask)
            kl_loss = kl.mean()
            loss = loss + self.kl_coef * kl_loss
            stats["kl"] = kl_loss.item()

        return loss, stats

    @classmethod
    def from_config(cls, config, vocabs):
        """Build a REINFORCELossCompute from the training config."""
        pad_token = vocabs["specials"].get("pad_token", DefaultTokens.PAD)
        padding_idx = vocabs["tgt"][pad_token]
        return cls(
            padding_idx=padding_idx,
            baseline=config.training.rl_baseline,
            kl_coef=config.training.rl_kl_coef,
        )


def build_rl_loss_compute(config, vocabs):
    """Factory building the appropriate RL loss compute for ``config.training.rl_algorithm``."""
    algorithm = config.training.rl_algorithm
    if algorithm == "reinforce":
        return REINFORCELossCompute.from_config(config, vocabs)
    elif algorithm in ("dpo", "grpo", "ppo"):
        raise NotImplementedError(
            f"rl_algorithm={algorithm!r} is planned but not implemented yet. "
            "Only 'reinforce' is currently supported."
        )
    else:
        raise ValueError(f"Unknown rl_algorithm={algorithm!r}")
