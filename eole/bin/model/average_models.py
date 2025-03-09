#!/usr/bin/env python
import torch
from eole.bin import BaseBin, register_bin
from eole.models import model_saver
from eole.config import recursive_model_fields_set
from safetensors.torch import load_file, save_file
import os
import json


def average_models(model_paths, fp32=False):
    vocab = None
    config = None
    avg_model = None

    for i, model_path in enumerate(model_paths):
        m = model_saver.load_checkpoint(model_path)
        model_weights = load_file(os.path.join(model_path, "model.00.safetensors"))

        if fp32:
            for k, v in model_weights.items():
                model_weights[k] = v.float()

        if i == 0:
            vocab, config, optim = m["vocab"], m["config"], m["optim"]
            avg_model = model_weights
        else:
            for k, v in avg_model.items():
                avg_model[k].mul_(i).add_(model_weights[k]).div_(i + 1)

    final = {
        "vocab": vocab,
        "config": config,
        "optim": optim,
        "model": avg_model,
    }
    return final


@register_bin(name="average")
class AverageModels(BaseBin):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument("-models", "-m", nargs="+", required=True, help="List of models")
        parser.add_argument("-output", "-o", required=True, help="Output file")
        parser.add_argument("-fp32", "-f", action="store_true", help="Cast params to float32")

    @classmethod
    def run(cls, args):
        final = average_models(args.models, args.fp32)

        if not os.path.isdir(args.output):
            os.makedirs(args.output, exist_ok=True)

        # this maybe better implemented using model_saver classes
        # config
        with open(os.path.join(args.output, "config.json"), "w", encoding="utf-8") as f:
            json.dump(
                recursive_model_fields_set(final["config"]),
                f,
                indent=2,
                ensure_ascii=False,
            )
        # vocab
        with open(os.path.join(args.output, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(final["vocab"], f, indent=2, ensure_ascii=False)
        # optimizer
        torch.save(final["optim"], os.path.join(args.output, "optimizer.pt"))
        # model weights
        save_file(final["model"], os.path.join(args.output, "model.00.safetensors"))
