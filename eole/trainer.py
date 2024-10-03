"""
    This is the loadable seq2seq trainer library that is
    in charge of training details, loss compute, and statistics.
    See train.py for a use case of this library.

    Note: To make this a general library, we implement *only*
          mechanism things here(i.e. what to do), and leave the strategy
          things to users(i.e. how to do it). Also see train.py(one of the
          users of this library) for the strategy things we do.
"""

import time
import sys
import torch
import traceback
import eole.utils
from eole.utils.loss import LossCompute
from eole.utils.logging import logger
from eole.utils.misc import clear_gpu_cache, get_autocast
from eole.utils.scoring_utils import ScoringPreparator
from eole.scorers import get_scorers_cls, build_scorers


def build_trainer(config, device_id, model, vocabs, optim, model_saver=None):
    """
    Simplify `Trainer` creation based on user `opt`s*

    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`eole.models.BaseModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`eole.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text"
        model_saver(:obj:`eole.models.ModelSaverBase`): the utility object
            used to save the model
    """

    train_loss = LossCompute.from_config(config, model, vocabs)
    valid_loss = LossCompute.from_config(config, model, vocabs, train=False)
    estim_loss_lambda = config.training.estim_loss_lambda
    estim_loss_lambda_steps = config.training.estim_loss_lambda_steps

    scoring_preparator = ScoringPreparator(vocabs, config)
    validset_transforms = getattr(config.data.get("valid", None), "transforms", None)
    if validset_transforms:
        scoring_preparator.warm_up(validset_transforms)
    scorers_cls = get_scorers_cls(config.valid_metrics)
    valid_scorers = build_scorers(config, scorers_cls)

    running_config = config.training
    trunc_size = running_config.truncated_decoder  # Badly named...
    norm_method = running_config.normalization
    accum_count = running_config.accum_count
    accum_steps = running_config.accum_steps
    n_gpu = running_config.world_size
    parallel_mode = running_config.parallel_mode
    average_decay = running_config.average_decay
    average_every = running_config.average_every
    dropout = running_config.dropout
    attention_dropout = running_config.attention_dropout
    dropout_steps = running_config.dropout_steps
    zero_out_prompt_loss = running_config.zero_out_prompt_loss
    if device_id >= 0:
        gpu_rank = running_config.gpu_ranks[device_id]
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

    trainer = Trainer(
        model,
        train_loss,
        valid_loss,
        scoring_preparator,
        valid_scorers,
        optim,
        trunc_size,
        norm_method,
        accum_count,
        accum_steps,
        n_gpu,
        gpu_rank,
        parallel_mode,
        report_manager,
        with_align=True
        if getattr(config.model.decoder, "lambda_align", 0.0) > 0
        else False,  # patch to support non transformer configs
        model_saver=model_saver,
        average_decay=average_decay,
        average_every=average_every,
        earlystopper=earlystopper,
        dropout=dropout,
        attention_dropout=attention_dropout,
        dropout_steps=dropout_steps,
        zero_out_prompt_loss=zero_out_prompt_loss,
        estim_loss_lambda=estim_loss_lambda,
        estim_loss_lambda_steps=estim_loss_lambda_steps,
    )
    return trainer


class Trainer(object):
    """Class that controls the training process.

    Args:
        model(:py:class:`eole.models.model.BaseModel`): model to train
        train_loss(:obj:`eole.utils.loss.LossComputeBase`):
          training loss computation
        valid_loss(:obj:`eole.utils.loss.LossComputeBase`):
          training loss computation
        scoring_preparator(:obj:`eole.predict.utils.ScoringPreparator`):
          preparator for the calculation of metrics via the
          _eval_handler method
        valid_scorers (dict): keeps in memory the current values
          of the validation metrics
        optim(:obj:`eole.utils.optimizers.Optimizer`):
          the optimizer responsible for update
        trunc_size(int): length of truncated back propagation
          through time
        accum_count(list): accumulate gradients this many times.
        accum_steps(list): steps for accum gradients changes.
        n_gpu (int): number of gpu.
        gpu_rank (int): ordinal rank of the gpu in the list.
        report_manager(:obj:`eole.utils.ReportMgrBase`):
          the object that creates reports, or None
        with_align (bool): whether to jointly lear alignment
          (Transformer)
        model_saver(:obj:`eole.models.ModelSaverBase`): the saver is
          used to save a checkpoint.
          Thus nothing will be saved if this parameter is None.
        average_decay (float): cf opt.average_decay
        average_every (int): average model every x steps.
        earlystopper (:obj:`eole.utils.EarlyStopping`): add early
          stopping mecanism
        dropout (float): dropout value in RNN or FF layers.
        attention_dropout (float): dropaout in attention layers.
        dropout_steps (list): dropout values scheduling in steps.
        zero_out_prompt_loss (bool): whether to zero-out the prompt loss
            (mostly for LLM finetuning).
        estim_loss_lambda (List[float]): weight applied to estimator loss
        estim_loss_lambda_steps (List[int]): steps to apply to estimator values"""

    def __init__(
        self,
        model,
        train_loss,
        valid_loss,
        scoring_preparator,
        valid_scorers,
        optim,
        trunc_size=0,
        norm_method="sents",
        accum_count=[1],
        accum_steps=[0],
        n_gpu=1,
        gpu_rank=1,
        parallel_mode="data_parallel",
        report_manager=None,
        with_align=False,
        model_saver=None,
        average_decay=0,
        average_every=1,
        earlystopper=None,
        dropout=[0.3],
        attention_dropout=[0.1],
        dropout_steps=[0],
        zero_out_prompt_loss=False,
        estim_loss_lambda=[1.0],
        estim_loss_lambda_steps=[0],
    ):
        # Basic attributes.

        self.model = model
        self.train_loss = train_loss
        self.estim_loss_lambda_l = estim_loss_lambda
        self.estim_loss_lambda = estim_loss_lambda[0]
        self.estim_loss_lambda_steps = estim_loss_lambda_steps
        self.valid_loss = valid_loss

        self.scoring_preparator = scoring_preparator
        self.valid_scorers = valid_scorers
        self.optim = optim
        self.trunc_size = trunc_size
        self.norm_method = norm_method
        self.accum_count_l = accum_count
        self.accum_count = accum_count[0]
        self.accum_steps = accum_steps
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.parallel_mode = parallel_mode
        self.report_manager = report_manager
        self.with_align = with_align
        self.model_saver = model_saver
        self.average_decay = average_decay
        self.moving_average = None
        self.average_every = average_every
        self.earlystopper = earlystopper
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.dropout_steps = dropout_steps
        self.zero_out_prompt_loss = zero_out_prompt_loss

        for i in range(len(self.accum_count_l)):
            assert self.accum_count_l[i] > 0

        # Set model in training mode.
        self.model.train()

    def _eval_handler(self, scorer, preds, texts_ref):
        """Trigger metrics calculations

        Args:
            scorer (:obj:``eole.scorer.Scorer``): scorer.
            preds, texts_ref: outputs of the scorer's `translate` method.

        Returns:
            The metric calculated by the scorer."""

        return scorer.compute_score(preds, texts_ref)

    def _accum_count(self, step):
        for i in range(len(self.accum_steps)):
            if step > self.accum_steps[i]:
                _accum = self.accum_count_l[i]
        return _accum

    def _maybe_update_dropout(self, step):
        for i in range(len(self.dropout_steps)):
            if step > 1 and step == self.dropout_steps[i] + 1:
                self.model.update_dropout(self.dropout[i], self.attention_dropout[i])
                logger.info(
                    "Updated dropout/attn dropout to %f %f at step %d"
                    % (self.dropout[i], self.attention_dropout[i], step)
                )

    def _maybe_update_estim_lambda(self, step):
        for i in range(len(self.estim_loss_lambda_steps)):
            if step > 1 and step == self.estim_loss_lambda_steps[i] + 1:
                self.estim_loss_lambda = self.estim_loss_lambda_l[i]
                logger.info(
                    "Updated estimator lambda to %f at step %d"
                    % (self.estim_loss_lambda_l[i], step)
                )

    def _accum_batches(self, iterator):
        batches = []
        normalization = 0
        self.accum_count = self._accum_count(self.optim.training_step)
        for batch, bucket_idx in iterator:
            batches.append(batch)
            if self.norm_method == "tokens":
                num_tokens = batch["tgt"][:, 1:].ne(self.train_loss.padding_idx).sum()
                normalization += num_tokens.item()
                normalization -= len(batch["tgt"])  # don't count for EOS
            else:
                normalization += len(batch["tgt"])
            if len(batches) == self.accum_count:
                yield batches, normalization
                self.accum_count = self._accum_count(self.optim.training_step)
                batches = []
                normalization = 0
        if batches:
            yield batches, normalization

    def _update_average(self, step):
        if self.moving_average is None:
            copy_params = [
                params.detach().float() for params in self.model.parameters()
            ]
            self.moving_average = copy_params
        else:
            average_decay = max(self.average_decay, 1 - (step + 1) / (step + 10))
            for (i, avg), cpt in zip(
                enumerate(self.moving_average), self.model.parameters()
            ):
                self.moving_average[i] = (
                    1 - average_decay
                ) * avg + cpt.detach().float() * average_decay

    def train(
        self,
        train_iter,
        train_steps,
        save_checkpoint_steps=5000,
        valid_iter=None,
        valid_steps=10000,
    ):
        """The main training loop by iterating over ``train_iter`` and possibly
        running validation on ``valid_iter``.

        Args:
            train_iter: An iterator that returns the next training batch.
            train_steps: Run training for this many iterations.
            save_checkpoint_steps: Save a checkpoint every this many
              iterations.
            valid_iter: A generator that returns the next validation batch.
            valid_steps: Run evaluation every this many iterations.

        Returns:
            :obj:``nmt.Statistics``: training loss statistics"""

        if valid_iter is None:
            logger.info("Start training loop without validation...")
            valid_stats = None
        else:
            logger.info(
                "Start training loop and validate every %d steps...", valid_steps
            )
        logger.info("Scoring with: {}".format(self.scoring_preparator.transforms))

        total_stats = eole.utils.Statistics()
        report_stats = eole.utils.Statistics()
        self._start_report_manager(start_time=total_stats.start_time)
        # Let's clean the GPUs before training loop
        clear_gpu_cache()

        for i, (batches, normalization) in enumerate(self._accum_batches(train_iter)):

            step = self.optim.training_step
            # UPDATE DROPOUT
            self._maybe_update_dropout(step)
            self._maybe_update_estim_lambda(step)

            if self.n_gpu > 1 and self.parallel_mode == "data_parallel":
                normalization = sum(
                    eole.utils.distributed.all_gather_list(normalization)
                )

            self._gradient_accumulation(
                batches, normalization, total_stats, report_stats
            )

            if self.average_decay > 0 and i % self.average_every == 0:
                self._update_average(step)

            report_stats = self._maybe_report_training(
                step, train_steps, self.optim.learning_rate(), report_stats
            )

            if valid_iter is not None and step % valid_steps == 0:
                if self.parallel_mode == "tensor_parallel" or self.gpu_rank <= 0:
                    valid_stats = self.validate(
                        valid_iter, moving_average=self.moving_average
                    )

            if step % valid_steps == 0 and self.gpu_rank <= 0:
                self._report_step(
                    self.optim.learning_rate(),
                    step,
                    valid_stats=valid_stats,
                    train_stats=total_stats,
                )

                # Run patience mechanism
                if self.earlystopper is not None:
                    self.earlystopper(valid_stats, step)
                    # If the patience has reached the limit, stop training
                    if self.earlystopper.has_stopped():
                        logger.info("earlystopper has_stopped!")
                        break

            if self.model_saver is not None and (
                save_checkpoint_steps != 0 and step % save_checkpoint_steps == 0
            ):
                self.model_saver.save(step, moving_average=self.moving_average)

            if train_steps > 0 and step >= train_steps:
                break

        if self.model_saver is not None:
            self.model_saver.save(step, moving_average=self.moving_average)
        return total_stats

    def validate(self, valid_iter, moving_average=None):
        """Validate model.

        Args:
            valid_iter: validate data iterator

        Returns:
            :obj:``nmt.Statistics``: validation loss statistics"""

        valid_model = self.model
        if moving_average:
            # swap model params w/ moving average
            # (and keep the original parameters)
            model_params_data = []
            for avg, param in zip(self.moving_average, valid_model.parameters()):
                model_params_data.append(param.data)
                param.data = (
                    avg.data.half() if param.dtype == torch.float16 else avg.data
                )

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
                    model_out, attns, estim = valid_model(
                        src, tgt, src_len, with_align=self.with_align
                    )

                    # Compute loss.
                    if self.zero_out_prompt_loss:
                        batch = self.valid_loss.ignore_prompt(batch)
                    _, batch_stats, _ = self.valid_loss(
                        batch, model_out, attns, estim=estim
                    )

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
                preds, texts_ref = self.scoring_preparator.translate(
                    model=self.model,
                    gpu_rank=self.gpu_rank,
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
                    logger.info(
                        "validation {}: {}".format(
                            metric, self.valid_scorers[metric]["value"]
                        )
                    )
                    # Compute stats
                    metric_stats = eole.utils.Statistics(
                        0, 0, 0, 0, 0, 0, computed_metrics
                    )

                # Update statistics.
                stats.update(metric_stats)

        if moving_average:
            for param_data, param in zip(model_params_data, self.model.parameters()):
                param.data = param_data

        # Set model back to training mode.
        valid_model.train()

        return stats

    def _gradient_accumulation(
        self, true_batches, normalization, total_stats, report_stats
    ):
        """Function that iterates over big batches = ``true_batches``

        Perform a backward on the loss of each sub_batch and
        finally update the params at the end of the big batch."""

        if self.accum_count > 1:
            self.optim.zero_grad(set_to_none=True)

        for k, batch in enumerate(true_batches):
            target_size = batch["tgt"].size(1)
            # Truncated BPTT: reminder not compatible with accum > 1
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                trunc_size = target_size

            src = batch["src"]
            src_len = batch["srclen"]
            if src_len is not None:
                report_stats.n_src_words += src_len.sum().item()
                total_stats.n_src_words += src_len.sum().item()

            tgt_outer = batch["tgt"]

            bptt = False
            for j in range(0, target_size - 1, trunc_size):
                # 1. Create truncated target.

                tgt = tgt_outer[:, j : j + trunc_size]

                # 2. F-prop all but generator.
                if self.accum_count == 1:
                    self.optim.zero_grad(set_to_none=True)
                try:
                    with get_autocast(enabled=self.optim.amp):
                        model_out, attns, estim = self.model(
                            src, tgt, src_len, bptt=bptt, with_align=self.with_align
                        )
                        bptt = True

                        # 3. Compute loss.
                        if self.zero_out_prompt_loss:
                            # The loss of the prompt will be set to zero.
                            batch = self.train_loss.ignore_prompt(batch)
                        loss, batch_stats, auxloss = self.train_loss(
                            batch,
                            model_out,
                            attns,
                            trunc_start=j,
                            trunc_size=trunc_size,
                            estim=estim,
                        )
                    if loss is not None:
                        loss /= normalization
                        auxloss /= self.accum_count * src_len.size(0)
                        loss = loss + auxloss * self.estim_loss_lambda
                        self.optim.backward(loss)

                    total_stats.update(batch_stats)
                    report_stats.update(batch_stats)

                except Exception as exc:
                    trace_content = traceback.format_exc()
                    if "CUDA out of memory" in trace_content:
                        logger.info(
                            "Step %d, cuda OOM - batch removed",
                            self.optim.training_step,
                        )
                        clear_gpu_cache()
                        if self.n_gpu > 1 and self.parallel_mode == "tensor_parallel":
                            torch.distributed.destroy_process_group()
                            sys.exit()
                    else:
                        traceback.print_exc()
                        raise exc

                # If truncated, don't backprop fully.
                if self.model.decoder is not None and self.model.decoder.state != {}:
                    self.model.decoder.detach_state()

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.n_gpu > 1 and self.parallel_mode == "data_parallel":
            grads = [
                p.grad.data
                for p in self.model.parameters()
                if p.requires_grad and p.grad is not None
            ]
            eole.utils.distributed.all_reduce_and_rescale_tensors(
                grads, float(self.n_gpu)
            )

        self.optim.step()

    def _start_report_manager(self, start_time=None):
        """Simple function to start report manager (if any)"""

        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_report_training(self, step, num_steps, learning_rate, report_stats):
        """Simple function to report training stats (if report_manager is set)
        see ``eole.utils.ReportManagerBase.report_training`` for doc"""

        if self.report_manager is not None:
            return self.report_manager.report_training(
                step,
                num_steps,
                learning_rate,
                None
                if self.earlystopper is None
                else self.earlystopper.current_tolerance,
                report_stats,
                multigpu=self.n_gpu > 1 and self.parallel_mode == "data_parallel",
            )

    def _report_step(self, learning_rate, step, valid_stats=None, train_stats=None):
        """Simple function to report stats (if report_manager is set)
        see ``eole.utils.ReportManagerBase.report_step`` for doc"""

        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate,
                None
                if self.earlystopper is None
                else self.earlystopper.current_tolerance,
                step,
                valid_stats=valid_stats,
                train_stats=train_stats,
            )
