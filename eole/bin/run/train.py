#!/usr/bin/env python
"""Train models with dynamic data."""
import torch
from functools import partial
from eole.utils.distributed import ErrorHandler, spawned_train
from eole.utils.misc import set_random_seed
from eole.utils.logging import init_logger, logger
from argparse import ArgumentParser

from eole.train_single import main as single_main

from eole.config.run import TrainConfig
from eole.bin import register_bin
from eole.bin.run import RunBin


# Set sharing strategy manually instead of default based on the OS.
# torch.multiprocessing.set_sharing_strategy('file_system')


def train(config):
    # config is the full TrainConfig namespace here
    config._validate_vocab_config()
    init_logger(config.log_file)
    set_random_seed(config.seed, False)

    train_process = partial(single_main)

    # a running config can either be training or inference, here we are training
    running_config = config.training

    nb_gpu = len(running_config.gpu_ranks)

    if running_config.world_size > 1:
        mp = torch.multiprocessing.get_context("spawn")
        # Create a thread to listen for errors in the child processes.
        error_queue = mp.SimpleQueue()
        error_handler = ErrorHandler(error_queue)
        # Train with multiprocessing.
        procs = []
        for device_id in range(nb_gpu):
            procs.append(
                mp.Process(
                    target=spawned_train,
                    args=(train_process, config, device_id, error_queue),
                    daemon=False,
                )
            )
            procs[device_id].start()
            logger.info(" Starting process pid: %d  " % procs[device_id].pid)
            error_handler.add_child(procs[device_id].pid)
        for p in procs:
            p.join()

    elif nb_gpu == 1:  # case 1 GPU only
        train_process(config, device_id=0)
    else:  # case only CPU
        train_process(config, device_id=-1)


@register_bin(name="train")
class Train(RunBin):
    config_class = TrainConfig
    require_config = True

    @classmethod
    def run(cls, args):
        config = cls.build_config(args)
        train(config)


def _get_parser():
    # Patch to allow autodoc with sphinx-argparse
    parser = ArgumentParser()
    Train.add_args(parser)
    return parser
