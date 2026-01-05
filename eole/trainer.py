"""
trainer library.
"""

import time
import sys
import torch
import traceback
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import eole.utils
from eole.utils.distributed import all_gather_list, all_reduce_and_rescale_tensors
from eole.utils.loss import LossCompute
from eole.utils.logging import logger
from eole.utils.misc import clear_gpu_cache, get_autocast
from eole.utils.scoring_utils import ScoringPreparator
from eole.scorers import get_scorers_cls, build_scorers


@dataclass
class TrainerConfig:
    """Configuration for trainer initialization."""

    norm_method: str = "sents"
    accum_count: List[int] = None
    accum_steps: List[int] = None
    n_gpu: int = 1
    gpu_rank: int = 1
    parallel_mode: str = "data_parallel"
    with_align: bool = False
    average_decay: float = 0
    average_every: int = 1
    dropout: List[float] = None
    attention_dropout: List[float] = None
    dropout_steps: List[int] = None
    zero_out_prompt_loss: bool = False
    estim_loss_lambda: List[float] = None
    estim_loss_lambda_steps: List[int] = None

    def __post_init__(self):
        self.accum_count = self.accum_count or [1]
        self.accum_steps = self.accum_steps or [0]
        self.dropout = self.dropout or [0.3]
        self.attention_dropout = self.attention_dropout or [0.1]
        self.dropout_steps = self.dropout_steps or [0]
        self.estim_loss_lambda = self.estim_loss_lambda or [1.0]
        self.estim_loss_lambda_steps = self.estim_loss_lambda_steps or [0]


class ScheduledParameter:
    """Manages parameters that change on a schedule."""

    def __init__(self, values: List[float], steps: List[int], name: str):
        self.values = values
        self.steps = steps
        self.name = name
        self.current_value = values[0]

    def update(self, step: int) -> bool:
        """Update parameter if step matches schedule. Returns True if updated."""
        for i, threshold in enumerate(self.steps):
            if step > 1 and step == threshold + 1:
                self.current_value = self.values[i]
                logger.info(f"Updated {self.name} to {self.current_value} at step {step}")
                return True
        return False


class BatchProcessor:
    """Handles batch processing and gradient accumulation."""

    def __init__(
        self,
        model,
        train_loss,
        optim,
        n_gpu: int,
        parallel_mode: str,
        with_align: bool,
        zero_out_prompt_loss: bool,
        estim_loss_lambda: float,
        accum_count: int,
    ):
        self.model = model
        self.train_loss = train_loss
        self.optim = optim
        self.n_gpu = n_gpu
        self.parallel_mode = parallel_mode
        self.with_align = with_align
        self.zero_out_prompt_loss = zero_out_prompt_loss
        self.estim_loss_lambda = estim_loss_lambda
        self.accum_count = accum_count

    def process_batch(
        self,
        batch: Dict[str, Any],
        normalization: int,
        accum_count: int,
        total_stats: eole.utils.Statistics,
        report_stats: eole.utils.Statistics,
    ) -> bool:
        """Process a single batch. Returns False if batch should be skipped."""
        src = batch["src"]
        src_len = batch["srclen"]
        tgt = batch["tgt"]

        # Update token counts
        n_tokens = src_len.sum().item()
        report_stats.n_src_tokens += n_tokens
        total_stats.n_src_tokens += n_tokens

        kwargs = {"images": batch["images"], "prefix_len": batch["prefix_len"]}

        try:
            loss, batch_stats = self._compute_loss(src, tgt, src_len, batch, normalization, accum_count, kwargs)

            if loss is not None:
                self.optim.backward(loss)

            total_stats.update(batch_stats)
            report_stats.update(batch_stats)
            return True

        except Exception as exc:
            # <-- Inline error handling instead of calling method
            trace_content = traceback.format_exc()

            if "CUDA out of memory" in trace_content:
                logger.warning(
                    "Step %d, CUDA OOM - batch removed (src shape: %s, tgt shape: %s)",
                    self.optim.training_step,
                    src.shape if hasattr(src, "shape") else "unknown",
                    tgt.shape if hasattr(tgt, "shape") else "unknown",
                )
                clear_gpu_cache()

                if self.n_gpu > 1 and self.parallel_mode == "tensor_parallel":
                    logger.error("OOM in tensor parallel mode - terminating all processes")
                    torch.distributed.destroy_process_group()
                    sys.exit(1)

                return False  # OOM occurred
            else:
                logger.error("Training error at step %d: %s", self.optim.training_step, str(exc))
                traceback.print_exc()
                raise exc

    def _compute_loss(
        self, src, tgt, src_len, batch: Dict[str, Any], normalization: int, accum_count: int, kwargs: Dict[str, Any]
    ) -> Tuple[Optional[torch.Tensor], eole.utils.Statistics]:
        """Compute loss for a batch."""
        with get_autocast(enabled=self.optim.amp):
            model_out, attns, estim = self.model(src, tgt, src_len, with_align=self.with_align, **kwargs)

            if self.zero_out_prompt_loss:
                batch = self.train_loss.ignore_prompt(batch)

            loss, batch_stats, auxloss = self.train_loss(batch, model_out, attns, estim=estim)

        if loss is not None:
            loss = loss / normalization
            auxloss = auxloss / (accum_count * src_len.size(0))
            loss = loss + auxloss * self.estim_loss_lambda

        return loss, batch_stats


class GradientSynchronizer:
    """Handles gradient synchronization for distributed training."""

    def __init__(self, model, n_gpu: int, parallel_mode: str):
        self.model = model
        self.n_gpu = n_gpu
        self.parallel_mode = parallel_mode

    def synchronize(self):
        """Synchronize gradients across GPUs if needed."""
        if self.n_gpu > 1 and self.parallel_mode == "data_parallel":
            grads = [p.grad.data for p in self.model.parameters() if p.requires_grad and p.grad is not None]
            if grads:
                all_reduce_and_rescale_tensors(grads, float(self.n_gpu))


class Trainer:
    """Refactored trainer with improved separation of concerns."""

    def __init__(
        self,
        model,
        train_loss,
        valid_loss,
        scoring_preparator,
        valid_scorers,
        optim,
        config: TrainerConfig,
        report_manager=None,
        model_saver=None,
        earlystopper=None,
    ):
        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.scoring_preparator = scoring_preparator
        self.valid_scorers = valid_scorers
        self.optim = optim
        self.config = config
        self.report_manager = report_manager
        self.model_saver = model_saver
        self.earlystopper = earlystopper
        self.moving_average = None

        # Initialize scheduled parameters
        self.dropout_scheduler = ScheduledParameter(config.dropout, config.dropout_steps, "dropout/attention_dropout")
        self.estim_lambda_scheduler = ScheduledParameter(
            config.estim_loss_lambda, config.estim_loss_lambda_steps, "estimator lambda"
        )
        self.accum_count_scheduler = ScheduledParameter(config.accum_count, config.accum_steps, "accumulation count")

        # Initialize helper classes
        self.batch_processor = BatchProcessor(
            model=model,
            train_loss=train_loss,
            optim=optim,
            n_gpu=config.n_gpu,
            parallel_mode=config.parallel_mode,
            with_align=config.with_align,
            zero_out_prompt_loss=config.zero_out_prompt_loss,
            estim_loss_lambda=self.estim_lambda_scheduler.current_value,
            accum_count=int(self.accum_count_scheduler.current_value),
        )

        self.gradient_synchronizer = GradientSynchronizer(
            model=model, n_gpu=config.n_gpu, parallel_mode=config.parallel_mode
        )

        self.model.train()

    def train(
        self,
        train_iter,
        train_steps: int,
        save_checkpoint_steps: int = 5000,
        valid_iter=None,
        valid_steps: int = 10000,
    ):
        """Main training loop."""
        if valid_iter is None:
            logger.info("Start training loop without validation...")
        else:
            logger.info("Start training loop and validate every %d steps...", valid_steps)

        logger.info(f"Scoring with: {self.scoring_preparator.transforms}")

        total_stats = eole.utils.Statistics()
        report_stats = eole.utils.Statistics()
        self._start_report_manager(start_time=total_stats.start_time)
        clear_gpu_cache()

        for i, (batches, normalization) in enumerate(self._accum_batches(train_iter)):
            step = self.optim.training_step

            # Update scheduled parameters
            self._update_scheduled_params(step)

            # Synchronize normalization across GPUs
            if self.config.n_gpu > 1 and self.config.parallel_mode == "data_parallel":
                normalization = sum(all_gather_list(normalization))

            # Process batches
            self._process_accumulated_batches(batches, normalization, total_stats, report_stats)

            # Update moving average
            if self.config.average_decay > 0 and i % self.config.average_every == 0:
                self._update_average(step)

            # Report training progress
            report_stats = self._maybe_report_training(
                step, train_steps, self.optim.learning_rate(step=step), report_stats
            )

            # Validation
            valid_stats = None
            if valid_iter is not None and step % valid_steps == 0:
                if self.config.parallel_mode == "tensor_parallel" or self.config.gpu_rank <= 0:
                    valid_stats = self.validate(valid_iter, moving_average=self.moving_average)

            # Report step
            if step % valid_steps == 0 and self.config.gpu_rank <= 0:
                self._report_step(
                    self.optim.learning_rate(),
                    step,
                    valid_stats=valid_stats,
                    train_stats=total_stats,
                )

                # Check early stopping
                if self._should_stop_early(valid_stats, step):
                    break

            # Save checkpoint
            if self.model_saver is not None and save_checkpoint_steps != 0 and step % save_checkpoint_steps == 0:
                self.model_saver.save(step, moving_average=self.moving_average)

            # Check if training is complete
            if train_steps > 0 and step >= train_steps:
                break

        # Final checkpoint
        if self.model_saver is not None:
            self.model_saver.save(step, moving_average=self.moving_average)

        return total_stats

    def _update_scheduled_params(self, step: int):
        """Update all scheduled parameters."""
        if self.dropout_scheduler.update(step):
            self.model.update_dropout(
                self.dropout_scheduler.current_value,
                self.config.attention_dropout[self.config.dropout_steps.index(step - 1)],
            )

        if self.estim_lambda_scheduler.update(step):
            self.batch_processor.estim_loss_lambda = self.estim_lambda_scheduler.current_value

        if self.accum_count_scheduler.update(step):
            self.batch_processor.accum_count = int(self.accum_count_scheduler.current_value)

    def _process_accumulated_batches(
        self,
        batches: List[Dict[str, Any]],
        normalization: int,
        total_stats: eole.utils.Statistics,
        report_stats: eole.utils.Statistics,
    ):
        """Process accumulated batches and update gradients."""
        self.optim.zero_grad(set_to_none=True)

        accum_count = int(self.accum_count_scheduler.current_value)

        oom_occurred = False

        for k, batch in enumerate(batches):
            success = self.batch_processor.process_batch(batch, normalization, accum_count, total_stats, report_stats)
            if not success:
                oom_occurred = True

        # In multi-GPU data parallel, synchronize OOM status across ranks
        if self.config.n_gpu > 1 and self.config.parallel_mode == "data_parallel":
            import torch.distributed as dist

            oom_tensor = torch.tensor([1 if oom_occurred else 0], dtype=torch.int32, device=torch.cuda.current_device())
            dist.all_reduce(oom_tensor, op=dist.ReduceOp.MAX)
            any_rank_oom = oom_tensor.item() > 0

            if any_rank_oom:
                logger.warning("OOM detected on at least one rank - skipping gradient update for this step")
                self.optim.zero_grad(set_to_none=True)
                return
        elif oom_occurred:
            logger.warning("OOM occurred - skipping gradient update")
            return

        # CRITICAL: Always synchronize and step if no OOM
        self.gradient_synchronizer.synchronize()
        self.optim.step()

    def _should_stop_early(self, valid_stats: Optional[eole.utils.Statistics], step: int) -> bool:
        """Check if training should stop early."""
        if self.earlystopper is None:
            return False

        self.earlystopper(valid_stats, step)
        if self.earlystopper.has_stopped():
            logger.info("earlystopper has_stopped!")
            return True
        return False

    def _accum_batches(self, iterator):
        """Accumulate batches for gradient accumulation."""
        batches = []
        normalization = 0
        accum_count = int(self.accum_count_scheduler.current_value)

        for batch, bucket_idx in iterator:
            batches.append(batch)

            if self.config.norm_method == "tokens":
                num_tokens = batch["tgt"][:, 1:].ne(self.train_loss.padding_idx).sum()
                normalization += num_tokens.item()
                normalization -= len(batch["tgt"])
            else:
                normalization += len(batch["tgt"])

            if len(batches) == accum_count:
                yield batches, normalization
                accum_count = int(self.accum_count_scheduler.current_value)
                batches = []
                normalization = 0

        if batches:
            yield batches, normalization

    def _update_average(self, step: int):
        """Update exponential moving average of parameters."""
        if self.moving_average is None:
            self.moving_average = [params.detach() for params in self.model.parameters()]
        else:
            average_decay = max(self.config.average_decay, 1 - (step + 1) / (step + 10))
            for i, (avg, param) in enumerate(zip(self.moving_average, self.model.parameters())):
                self.moving_average[i] = (1 - average_decay) * avg + param.detach() * average_decay

    def validate(self, valid_iter, moving_average=None):
        """Validate model."""
        valid_model = self.model
        if moving_average:
            # swap model params w/ moving average
            # (and keep the original parameters)
            model_params_data = []
            for avg, param in zip(self.moving_average, valid_model.parameters()):
                model_params_data.append(param.data)
                param.data = avg.data.half() if param.dtype == torch.float16 else avg.data

        # Set model in validating mode.
        valid_model.eval()

        with torch.no_grad():
            stats = eole.utils.Statistics()
            start = time.time()
            for batch, bucket_idx in valid_iter:
                src = batch["src"]
                src_len = batch["srclen"]
                tgt = batch["tgt"]

                with get_autocast(enabled=self.optim.amp):
                    # F-prop through the model.
                    model_out, attns, estim = valid_model(src, tgt, src_len, with_align=self.config.with_align)

                    # Compute loss.
                    if self.config.zero_out_prompt_loss:
                        batch = self.valid_loss.ignore_prompt(batch)
                    _, batch_stats, _ = self.valid_loss(batch, model_out, attns, estim=estim)

                    stats.update(batch_stats)
            logger.info(
                """valid stats calculation
                           took: {} s.""".format(
                    time.time() - start
                )
            )

            # Compute validation metrics (at batch.dataset level)
            if len(self.valid_scorers) > 0:
                computed_metrics = {}
                start = time.time()
                with get_autocast(enabled=self.optim.amp):
                    preds, texts_ref = self.scoring_preparator.translate(
                        model=self.model,
                        gpu_rank=self.config.gpu_rank,
                        step=self.optim.training_step,
                    )
                logger.info(
                    """The translation of the valid dataset for dynamic scoring
                               took : {} s.""".format(
                        time.time() - start
                    )
                )
                for i, metric in enumerate(self.valid_scorers):
                    logger.info("UPDATING VALIDATION {}".format(metric))
                    self.valid_scorers[metric]["value"] = self._eval_handler(
                        scorer=self.valid_scorers[metric]["scorer"],
                        preds=preds,
                        texts_ref=texts_ref,
                    )
                    computed_metrics[metric] = self.valid_scorers[metric]["value"]
                    logger.info("validation {}: {}".format(metric, self.valid_scorers[metric]["value"]))
                    # Compute stats
                    metric_stats = eole.utils.Statistics(0, 0, 0, 0, 0, 0, computed_metrics)

                # Update statistics.
                stats.update(metric_stats)
                valid_model.decoder._disable_cache()

        if moving_average:
            for param_data, param in zip(model_params_data, self.model.parameters()):
                param.data = param_data

        # Set model back to training mode.
        valid_model.train()

        return stats

    def _eval_handler(self, scorer, preds, texts_ref):
        """Trigger metrics calculations

        Args:
            scorer (:obj:``eole.scorer.Scorer``): scorer.
            preds, texts_ref: outputs of the scorer's `translate` method.

        Returns:
            The metric calculated by the scorer."""

        return scorer.compute_score(preds, texts_ref)

    def _start_report_manager(self, start_time=None):
        """Start report manager."""
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_report_training(self, step, num_steps, learning_rate, report_stats):
        """Report training stats if manager is set."""
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step,
                num_steps,
                learning_rate,
                (None if self.earlystopper is None else self.earlystopper.current_tolerance),
                report_stats,
                multigpu=(self.config.n_gpu > 1 and self.config.parallel_mode == "data_parallel"),
            )
        return report_stats  # Return unchanged if no manager

    def _report_step(self, learning_rate, step, valid_stats=None, train_stats=None):
        """Report step stats."""
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate,
                (None if self.earlystopper is None else self.earlystopper.current_tolerance),
                step,
                valid_stats=valid_stats,
                train_stats=train_stats,
            )


def build_trainer(config, device_id, model, vocabs, optim, model_saver=None):
    """Build trainer from configuration."""
    train_loss = LossCompute.from_config(config, model, vocabs)
    valid_loss = LossCompute.from_config(config, model, vocabs, train=False)

    scoring_preparator = ScoringPreparator(vocabs, config)
    validset_transforms = getattr(config.data.get("valid", None), "transforms", None)
    if validset_transforms:
        scoring_preparator.warm_up(validset_transforms)

    scorers_cls = get_scorers_cls(config.valid_metrics)
    valid_scorers = build_scorers(config, scorers_cls)

    running_config = config.training

    if device_id >= 0:
        gpu_rank = running_config.gpu_ranks[device_id]
        n_gpu = running_config.world_size
    else:
        gpu_rank = -1
        n_gpu = 0

    earlystopper = (
        eole.utils.EarlyStopping(
            running_config.early_stopping,
            scorers=eole.utils.scorers_from_config(running_config),
        )
        if running_config.early_stopping > 0
        else None
    )

    report_manager = eole.utils.build_report_manager(config, gpu_rank)

    trainer_config = TrainerConfig(
        norm_method=running_config.normalization,
        accum_count=running_config.accum_count,
        accum_steps=running_config.accum_steps,
        n_gpu=n_gpu,
        gpu_rank=gpu_rank,
        parallel_mode=running_config.parallel_mode,
        with_align=(True if getattr(config.model.decoder, "lambda_align", 0.0) > 0 else False),
        average_decay=running_config.average_decay,
        average_every=running_config.average_every,
        dropout=running_config.dropout,
        attention_dropout=running_config.attention_dropout,
        dropout_steps=running_config.dropout_steps,
        zero_out_prompt_loss=running_config.zero_out_prompt_loss,
        estim_loss_lambda=running_config.estim_loss_lambda,
        estim_loss_lambda_steps=running_config.estim_loss_lambda_steps,
    )

    return Trainer(
        model=model,
        train_loss=train_loss,
        valid_loss=valid_loss,
        scoring_preparator=scoring_preparator,
        valid_scorers=valid_scorers,
        optim=optim,
        config=trainer_config,
        report_manager=report_manager,
        model_saver=model_saver,
        earlystopper=earlystopper,
    )
