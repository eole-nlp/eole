""" Report manager utility """

import time
from datetime import datetime

import eole

from eole.utils.logging import logger


def build_report_manager(config, gpu_rank):
    managers = []

    # Stdout is always enabled
    managers.append(StdoutReportMgr(config.report_every))

    # TensorBoard
    if config.tensorboard and gpu_rank <= 0:
        from torch.utils.tensorboard import SummaryWriter

        if getattr(config, "tensorboard_log_dir_dated", None) is None:
            config.tensorboard_log_dir_dated = config.tensorboard_log_dir + datetime.now().strftime("/%b-%d_%H-%M-%S")
        writer = SummaryWriter(config.tensorboard_log_dir_dated, comment="Unmt")
        managers.append(TensorboardReportMgr(config.report_every, writer))

    # Trackio
    if config.trackio and gpu_rank <= 0:
        try:
            import trackio

            trackio.init(
                project=config.trackio_project,
                name=config.trackio_run_name,
                space_id=config.trackio_space_id,
                bucket_id=config.trackio_bucket_id,
            )
            # Log model and training config as hyperparameters
            trackio_config = {}
            if hasattr(config, "model") and config.model is not None:
                trackio_config["model"] = config.model.model_dump()
            if hasattr(config, "training") and config.training is not None:
                trackio_config["training"] = config.training.model_dump()
            if trackio_config:
                trackio.config.update(trackio_config)
            managers.append(TrackioReportMgr(config.report_every))
        except ImportError:
            logger.warning("trackio not installed; skipping trackio logging.")
        except Exception:
            logger.warning("Failed to initialize trackio; skipping trackio logging.", exc_info=True)

    return CompositeReportMgr(config.report_every, managers)


class ReportMgrBase(object):
    """
    Report Manager Base class
    Inherited classes should override:
        * `_report_training`
        * `_report_step`
    """

    def __init__(self, report_every, start_time=-1.0):
        """
        Args:
            report_every(int): Report status every this many sentences
            start_time(float): manually set report start time. Negative values
                means that you will need to set it later or use `start()`
        """
        self.report_every = report_every
        self.start_time = start_time

    def start(self):
        self.start_time = time.time()

    def log(self, *args, **kwargs):
        logger.info(*args, **kwargs)

    def report_training(self, step, num_steps, learning_rate, patience, report_stats, multigpu=False):
        """
        This is the user-defined batch-level training progress
        report function.
        Args:
            step(int): current step count.
            num_steps(int): total number of batches.
            learning_rate(float): current learning rate.
            report_stats(Statistics): old Statistics instance.
        Returns:
            report_stats(Statistics): updated Statistics instance.
        """
        if self.start_time < 0:
            raise ValueError(
                """ReportMgr needs to be started
                                (set 'start_time' or use 'start()'"""
            )

        if step % self.report_every == 0:
            if multigpu:
                report_stats = eole.utils.Statistics.all_gather_stats(report_stats)
            self._report_training(step, num_steps, learning_rate, patience, report_stats)
            return eole.utils.Statistics()
        else:
            return report_stats

    def _report_training(self, *args, **kwargs):
        """To be overridden"""
        raise NotImplementedError()

    def report_step(self, lr, patience, step, train_stats=None, valid_stats=None):
        """
        Report stats of a step

        Args:
            lr(float): current learning rate
            patience(int): current patience
            step(int): current step
            train_stats(Statistics): training stats
            valid_stats(Statistics): validation stats
        """
        self._report_step(lr, patience, step, valid_stats=valid_stats, train_stats=train_stats)

    def _report_step(self, *args, **kwargs):
        raise NotImplementedError()

    def close(self):
        """Clean up resources. Override in subclasses."""
        pass


class StdoutReportMgr(ReportMgrBase):
    """Report manager that writes statistics to standard output."""

    def _report_training(self, step, num_steps, learning_rate, patience, report_stats):
        report_stats.output(step, num_steps, learning_rate, self.start_time)

    def _report_step(self, lr, patience, step, valid_stats=None, train_stats=None):
        if train_stats is not None:
            self.log("Train perplexity: %g" % train_stats.ppl())
            self.log("Train accuracy: %g" % train_stats.accuracy())
            self.log("Sentences processed: %g" % train_stats.n_sents)
            self.log(train_stats.data_stats)
            self.log(
                "Average bsz: %4.0f/%4.0f/%2.0f"
                % (
                    train_stats.n_src_tokens / train_stats.n_batchs,
                    train_stats.n_tokens / train_stats.n_batchs,
                    train_stats.n_sents / train_stats.n_batchs,
                )
            )
        if valid_stats is not None:
            self.log("Validation perplexity: %g" % valid_stats.ppl())
            self.log("Validation accuracy: %g" % valid_stats.accuracy())


class TensorboardReportMgr(ReportMgrBase):
    """Report manager that logs statistics to TensorBoard."""

    def __init__(self, report_every, writer, start_time=-1.0):
        super().__init__(report_every, start_time)
        self.writer = writer

    def _report_training(self, step, num_steps, learning_rate, patience, report_stats):
        report_stats.log_tensorboard("progress", self.writer, learning_rate, patience, step)

    def _report_step(self, lr, patience, step, valid_stats=None, train_stats=None):
        if train_stats is not None:
            train_stats.log_tensorboard("train", self.writer, lr, patience, step)
        if valid_stats is not None:
            valid_stats.log_tensorboard("valid", self.writer, lr, patience, step)

    def close(self):
        self.writer.close()


class TrackioReportMgr(ReportMgrBase):
    """Report manager that logs statistics to trackio.

    Fail-safe: if trackio raises at runtime (e.g. network error when logging
    to a remote HF Space or self-hosted server), the error is logged once and
    trackio is disabled for the rest of the run.
    """

    def __init__(self, report_every, start_time=-1.0):
        super().__init__(report_every, start_time)
        self._disabled = False

    def _report_training(self, step, num_steps, learning_rate, patience, report_stats):
        if self._disabled:
            return
        try:
            report_stats.log_trackio("progress", learning_rate, patience, step)
        except Exception:
            logger.warning("trackio logging failed; disabling for the rest of this run.", exc_info=True)
            self._disabled = True

    def _report_step(self, lr, patience, step, valid_stats=None, train_stats=None):
        if self._disabled:
            return
        try:
            if train_stats is not None:
                train_stats.log_trackio("train", lr, patience, step)
            if valid_stats is not None:
                valid_stats.log_trackio("valid", lr, patience, step)
        except Exception:
            logger.warning("trackio logging failed; disabling for the rest of this run.", exc_info=True)
            self._disabled = True

    def close(self):
        if self._disabled:
            return
        try:
            import trackio

            trackio.finish()
        except Exception:
            logger.warning("trackio finish failed.", exc_info=True)


class CompositeReportMgr(ReportMgrBase):
    """Report manager that dispatches to multiple backend managers."""

    def __init__(self, report_every, managers, start_time=-1.0):
        self.managers = managers  # set before super() so the start_time setter can propagate
        super().__init__(report_every, start_time)

    @property
    def start_time(self):
        return self._start_time

    @start_time.setter
    def start_time(self, value):
        self._start_time = value
        for mgr in self.managers:
            mgr.start_time = value

    def start(self):
        self.start_time = time.time()

    def _report_training(self, step, num_steps, learning_rate, patience, report_stats):
        # Call _report_training directly on children rather than report_training,
        # because the step-check and multigpu gather in the base class should
        # only happen once (in this composite), not once per child.
        for mgr in self.managers:
            mgr._report_training(step, num_steps, learning_rate, patience, report_stats)

    def _report_step(self, lr, patience, step, valid_stats=None, train_stats=None):
        for mgr in self.managers:
            mgr._report_step(lr, patience, step, valid_stats=valid_stats, train_stats=train_stats)

    def close(self):
        for mgr in self.managers:
            try:
                mgr.close()
            except Exception:
                logger.warning("Failed to close %s.", mgr.__class__.__name__, exc_info=True)
