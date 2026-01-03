#!/usr/bin/env python
"""Train models with dynamic data."""
import torch
from functools import partial
from typing import List
from enum import Enum

from eole.utils.distributed import ErrorHandler
from eole.utils.distributed_workers import spawned_train
from eole.utils.misc import set_random_seed
from eole.utils.logging import init_logger, logger
from argparse import ArgumentParser
from eole.train_single import main as single_main
from eole.config.run import TrainConfig
from eole.bin import register_bin
from eole.bin.run import RunBin


class TrainingMode(Enum):
    """Training execution modes."""

    MULTIPROCESS = "multiprocess"
    SINGLE_GPU = "single_gpu"
    CPU = "cpu"


class TrainingOrchestrator:
    """Orchestrates training process selection and execution."""

    def __init__(self, config: TrainConfig):
        self.config = config
        self.running_config = config.training
        self.nb_gpu = len(self.running_config.gpu_ranks)

    def determine_mode(self) -> TrainingMode:
        """Determine which training mode to use."""
        if self.running_config.world_size > 1:
            return TrainingMode.MULTIPROCESS
        elif self.nb_gpu == 1:
            return TrainingMode.SINGLE_GPU
        else:
            return TrainingMode.CPU

    def execute(self):
        """Execute training in the appropriate mode."""
        mode = self.determine_mode()

        logger.info(f"Starting training in {mode.value} mode")

        if mode == TrainingMode.MULTIPROCESS:
            self._train_multiprocess()
        elif mode == TrainingMode.SINGLE_GPU:
            self._train_single_gpu()
        else:
            self._train_cpu()

    def _train_multiprocess(self):
        """Execute multiprocess training across multiple GPUs."""
        mp = torch.multiprocessing.get_context("spawn")
        error_queue = mp.SimpleQueue()
        error_handler = ErrorHandler(error_queue)

        processes = self._spawn_training_processes(mp, error_queue, error_handler)
        self._wait_for_processes(processes)

    def _spawn_training_processes(
        self, mp, error_queue, error_handler: ErrorHandler
    ) -> List[torch.multiprocessing.Process]:
        """Spawn training processes for each GPU."""
        train_process = partial(single_main)
        processes = []

        for device_id in range(self.nb_gpu):
            process = mp.Process(
                target=spawned_train,
                args=(train_process, self.config, device_id, error_queue),
                daemon=False,
            )
            process.start()
            processes.append(process)

            logger.info(f"Started training process on GPU {device_id} (pid: {process.pid})")
            error_handler.add_child(process.pid)

        return processes

    def _wait_for_processes(self, processes: List[torch.multiprocessing.Process]):
        """Wait for all training processes to complete."""
        logger.info(f"Waiting for {len(processes)} training processes to complete...")
        for process in processes:
            process.join()
        logger.info("All training processes completed")

    def _train_single_gpu(self):
        """Execute training on a single GPU."""
        logger.info("Training on single GPU (device_id=0)")
        single_main(self.config, device_id=0)

    def _train_cpu(self):
        """Execute training on CPU."""
        logger.warning("Training on CPU - this may be very slow")
        single_main(self.config, device_id=-1)


class TrainingSetup:
    """Handles training initialization and setup."""

    def __init__(self, config: TrainConfig):
        self.config = config

    def initialize(self):
        """Initialize all training prerequisites."""
        self._validate_config()
        self._setup_logging()
        self._setup_random_seed()

    def _validate_config(self):
        """Validate configuration before training."""
        logger.info("Validating vocabulary configuration...")
        self.config._validate_vocab_config()

    def _setup_logging(self):
        """Initialize logging system."""
        init_logger(self.config.log_file)
        logger.info(f"Logging initialized (log_file: {self.config.log_file})")

    def _setup_random_seed(self):
        """Set random seed for reproducibility."""
        set_random_seed(self.config.seed, False)
        logger.info(f"Random seed set to: {self.config.seed}")


def train(config: TrainConfig):
    """
    Main training entry point.

    Args:
        config: Full TrainConfig namespace with all training parameters.
    """
    # Initialize training setup
    setup = TrainingSetup(config)
    setup.initialize()

    # Log training configuration summary
    _log_training_summary(config)

    # Execute training
    orchestrator = TrainingOrchestrator(config)
    orchestrator.execute()


def _log_training_summary(config: TrainConfig):
    """Log a summary of the training configuration."""
    running_config = config.training

    logger.info("=" * 60)
    logger.info("Training Configuration Summary")
    logger.info("=" * 60)
    logger.info(f"World size: {running_config.world_size}")
    logger.info(f"GPU ranks: {running_config.gpu_ranks}")
    logger.info(f"Parallel mode: {running_config.parallel_mode}")
    logger.info(f"Train steps: {running_config.train_steps}")
    logger.info(f"Batch size: {running_config.batch_size}")
    logger.info(f"Accumulation count: {running_config.accum_count}")
    logger.info("=" * 60)


@register_bin(name="train")
class Train(RunBin):
    """Training command-line interface."""

    config_class = TrainConfig
    require_config = True

    @classmethod
    def run(cls, args):
        """
        Execute training from command-line arguments.

        Args:
            args: Parsed command-line arguments.
        """
        config = cls.build_config(args)

        try:
            train(config)
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise


def _get_parser():
    """
    Get argument parser for sphinx-argparse documentation.

    Returns:
        ArgumentParser with all training arguments.
    """
    parser = ArgumentParser(description="Train neural machine translation models")
    Train.add_args(parser)
    return parser


if __name__ == "__main__":
    # Allow direct execution for testing
    parser = _get_parser()
    args = parser.parse_args()
    Train.run(args)
