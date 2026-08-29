"""
RL trainer for on-policy RL fine-tuning methods (currently REINFORCE).

This subclasses :class:`eole.trainer.Trainer` and only overrides
``_train_step()`` (the extension point added by the RL-readiness
refactoring), reusing the outer loop for scheduling, validation,
checkpointing and reporting.

The per-step logic is:
  1. sample a rollout (generation) from the current policy for each prompt
     in the batch (:func:`eole.utils.generate_utils.generate_and_score`);
  2. score the rollout with the configured reward scorer(s);
  3. recompute the log-probs of the sampled tokens under the current policy
     (:meth:`eole.models.model.BaseModel.compute_log_probs`) — this is what
     is differentiated through, as opposed to the (non-differentiable)
     generation step itself;
  4. optionally recompute the same log-probs under a frozen reference model
     for a KL penalty;
  5. turn (log-probs, reward) into a policy-gradient loss
     (:class:`eole.utils.rl_loss.REINFORCELossCompute`) and backward it.

Only REINFORCE is implemented. GRPO/PPO would subclass :class:`RLTrainer`
(e.g. to add a group-relative baseline or a clipped surrogate objective) and
reuse the same rollout/scoring plumbing; DPO would not use this trainer at
all since it trains on offline preference pairs rather than on-policy
rollouts.
"""

from typing import Any, Dict, List

import torch

import eole.utils
from eole.trainer import Trainer
from eole.utils.logging import logger
from eole.utils.misc import clear_gpu_cache, get_autocast
from eole.utils.generate_utils import GenerationConfig, build_predictor_from_model, generate_and_score


class RLTrainer(Trainer):
    """Trainer for on-policy RL fine-tuning (REINFORCE)."""

    def __init__(
        self,
        model,
        train_loss,
        valid_loss,
        scoring_preparator,
        valid_scorers,
        optim,
        config,
        rl_loss_compute,
        reward_scorers: Dict[str, Any],
        rl_reward_metric: str,
        vocabs,
        model_config,
        device_id: int = -1,
        rl_gen_config: GenerationConfig = None,
        reference_model=None,
        report_manager=None,
        model_saver=None,
        earlystopper=None,
    ):
        super().__init__(
            model=model,
            train_loss=train_loss,
            valid_loss=valid_loss,
            scoring_preparator=scoring_preparator,
            valid_scorers=valid_scorers,
            optim=optim,
            config=config,
            report_manager=report_manager,
            model_saver=model_saver,
            earlystopper=earlystopper,
        )
        self.rl_loss_compute = rl_loss_compute
        self.reward_scorers = reward_scorers
        self.rl_reward_metric = rl_reward_metric
        self.vocabs = vocabs
        self.model_config = model_config
        self.device_id = device_id
        self.rl_gen_config = rl_gen_config or GenerationConfig()
        self.reference_model = reference_model
        if self.reference_model is not None:
            self.reference_model.eval()
        self._predictor = None

    def _get_predictor(self):
        """Lazily build (and cache) the in-memory predictor used for rollouts.

        Built lazily so the model is fully initialized/placed on its device
        before the predictor wraps it.
        """
        if self._predictor is None:
            self._predictor = build_predictor_from_model(
                self.model,
                self.vocabs,
                self.model_config,
                device_id=self.device_id,
                gen_config=self.rl_gen_config,
            )
        return self._predictor

    def _train_step(
        self,
        batches: List[Dict[str, Any]],
        normalization: int,
        total_stats: eole.utils.Statistics,
        report_stats: eole.utils.Statistics,
        step: int = 0,
    ):
        """Execute one REINFORCE training step on accumulated batches."""
        self.optim.zero_grad(set_to_none=True)

        for batch in batches:
            batch_stats = self._process_rl_batch(batch, normalization, step)
            if batch_stats is not None:
                total_stats.update(batch_stats)
                report_stats.update(batch_stats)

        self.gradient_synchronizer.synchronize()
        self.optim.step()

    def _process_rl_batch(self, batch, normalization, step):
        """Sample a rollout, score it, and backward the policy-gradient loss
        for a single batch. Returns the batch's Statistics (or None on OOM)."""
        predictor = self._get_predictor()

        try:
            gen_results = generate_and_score(
                predictor,
                batch,
                self.reward_scorers,
                return_token_ids=True,
            )
        except RuntimeError as exc:
            if "CUDA out of memory" in str(exc):
                logger.warning(
                    "Step %d, CUDA OOM during rollout generation - batch skipped "
                    "(consider reducing batch_size, rl_gen_max_length, or accum_count)",
                    step,
                )
                clear_gpu_cache()
                return None
            raise

        rewards = torch.tensor(
            gen_results["rewards"][self.rl_reward_metric],
            device=batch["src"].device,
        )

        # Rebuild a (batch, tgt_len) tensor of the sampled tokens (best/first
        # rollout per prompt) to recompute differentiable log-probs.
        # Use the model's actual pad index so compute_log_probs() builds the
        # correct padding mask (fallback of 1 matches the EOLE convention and
        # mirrors the fallback in compute_log_probs itself); for encoder-decoder
        # models (tgt_shift=1) prepend a BOS token so the first real token
        # survives the shift.
        pad_idx = getattr(self.model, "tgt_pad_idx", getattr(self.model, "pad_idx", 1))
        bos_id = None
        if getattr(self.model, "tgt_shift", 0) == 1:
            specials = self.vocabs.get("specials", {})
            bos_token = specials.get("bos_token", "<s>")
            if bos_token not in self.vocabs["tgt"]:
                raise RuntimeError(
                    f"RL training with an encoder-decoder model requires a BOS token in the target vocabulary, "
                    f"but '{bos_token}' was not found. Check the 'bos_token' entry in vocabs['specials']."
                )
            bos_id = self.vocabs["tgt"].lookup_token(bos_token)
        sampled_tgt, sampled_len = self._build_sampled_tgt(
            gen_results["token_ids"], batch["src"].device, pad_idx=pad_idx, bos_id=bos_id
        )

        src = batch["src"]
        src_len = batch["srclen"]
        kwargs = {"images": batch.get("images"), "prefix_len": batch.get("prefix_len")}

        with get_autocast(enabled=self.optim.amp):
            if self.model.is_decoder_only:
                out = self.model.compute_log_probs(sampled_tgt, sampled_tgt, sampled_len, **kwargs)
            else:
                out = self.model.compute_log_probs(src, sampled_tgt, src_len, **kwargs)

            ref_token_log_probs = None
            if self.reference_model is not None:
                with torch.no_grad():
                    if self.model.is_decoder_only:
                        ref_out = self.reference_model.compute_log_probs(
                            sampled_tgt, sampled_tgt, sampled_len, **kwargs
                        )
                    else:
                        ref_out = self.reference_model.compute_log_probs(src, sampled_tgt, src_len, **kwargs)
                ref_token_log_probs = ref_out["token_log_probs"]

            loss, loss_stats = self.rl_loss_compute.compute_policy_loss(
                out["token_log_probs"],
                out["padding_mask"],
                rewards,
                ref_token_log_probs=ref_token_log_probs,
            )

        if loss is not None:
            self.optim.backward(loss / normalization)

        n_sents = len(batch["srclen"])
        n_tokens = int(out["padding_mask"].sum().item())
        stats = eole.utils.Statistics(
            loss=loss.item() * n_sents,
            auxloss=0,
            n_batchs=1,
            n_sents=n_sents,
            n_tokens=n_tokens,
            n_correct=0,
            computed_metrics={"reward": loss_stats["reward_mean"]},
        )
        return stats

    @staticmethod
    def _build_sampled_tgt(token_ids, device, pad_idx=1, bos_id=None):
        """Pad the (best) sampled token-id sequences into a ``(batch, tgt_len)``
        LongTensor plus their lengths, suitable for ``compute_log_probs``.

        Args:
            token_ids: list (batch) of list (n_best) of 1D LongTensors, as
                returned by ``generate_from_batch(..., return_token_ids=True)``.
            device: target device for the returned tensors.
            pad_idx (int): vocabulary index of the padding token (defaults to 1).
                Must match the model's pad index so ``compute_log_probs`` builds
                the correct padding mask.
            bos_id (int or None): if not None, prepend this token to every
                sequence (required for encoder-decoder models where
                ``tgt_shift=1`` drops the first token before computing
                log-probabilities, so the BOS must be present to keep the
                sampled tokens aligned).

        Returns:
            Tuple ``(tgt, tgt_len)`` where ``tgt`` is ``(batch, max_len)``
            and ``tgt_len`` is ``(batch,)``.
        """
        sequences = [ids[0].to(device) for ids in token_ids]
        if bos_id is not None:
            bos = torch.tensor([bos_id], dtype=torch.long, device=device)
            sequences = [torch.cat([bos, seq]) for seq in sequences]
        lengths = torch.tensor([seq.size(0) for seq in sequences], device=device)
        max_len = int(lengths.max().item())
        padded = torch.full((len(sequences), max_len), pad_idx, dtype=torch.long, device=device)
        for i, seq in enumerate(sequences):
            padded[i, : seq.size(0)] = seq
        return padded, lengths
