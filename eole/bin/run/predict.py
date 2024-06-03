#!/usr/bin/env python
# -*- coding: utf-8 -*-
from eole.inference_engine import InferenceEnginePY

from argparse import ArgumentParser
from eole.utils.misc import use_gpu, set_random_seed
from torch.profiler import profile, record_function, ProfilerActivity
from eole.config.run import PredictConfig
from eole.bin import register_bin
from eole.bin.run import RunBin
from time import time


def predict(config):
    set_random_seed(config.seed, use_gpu(config))

    engine = InferenceEnginePY(config)
    _, _, _ = engine.infer_file()
    engine.terminate()


@register_bin(name="predict")
class Predict(RunBin):
    config_class = PredictConfig
    require_config = False

    @classmethod
    def run(cls, args):
        config = cls.build_config(args)

        if config.profile:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                profile_memory=True,
                with_stack=True,
            ) as prof:
                with record_function("Predict"):
                    predict(config)
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40))
        else:
            init_time = time()
            predict(config)
            print("Time w/o python interpreter load/terminate: ", time() - init_time)


def _get_parser():
    # Patch to allow autodoc with sphinx-argparse
    parser = ArgumentParser()
    Predict.add_args(parser)
    return parser
