"""
    Inspired from https://github.com/karpathy/llm.c/blob/master/dev/data/hellaswag.py
"""

import json
import yaml
import os
import requests
from tqdm import tqdm
from argparse import ArgumentParser
from eole.inference_engine import InferenceEnginePY

from eole.bin import BaseBin, register_bin
from eole.config.cli import add_model
from eole.config import get_non_default_values
from eole.config.run import PredictConfig

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")

hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}


def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def download(split):
    """Downloads HellaSwag DATA_CACHE_DIR"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")


def iterate_examples(split):
    # there are 10,042 examples in total in val
    download(split)
    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") as f:
        for line in f:
            example = json.loads(line)
            yield example


def evaluate(config):
    total = 0
    correct = 0
    correct_norm = 0
    engine = InferenceEnginePY(config)
    left_pad = config.model.left_pad
    for data in iterate_examples("val"):
        total += 1
        # retrieve length of tokenized context
        transformed_context = engine.transform_pipe.apply(
            {"src": data["ctx"].split(" "), "tgt": None}
        )
        context_length = len(transformed_context["src"])
        label = data["label"]
        examples = [f'{data["ctx"]} {ending}' for ending in data["endings"]]
        scores = engine.score_list(examples)
        # do not consider context log probs (note: padding ones are 0)
        norm_scores = []
        max_len = max([len(log_probs) for _, log_probs, _ in scores])
        for _, log_probs, length in scores:
            if left_pad:
                offset = max_len - length + context_length
            else:
                offset = context_length
            norm_score = sum(log_probs[offset:]) / (length - context_length)
            norm_scores.append(norm_score)
        _scores = [item[0] for item in scores]
        pred_label = _scores.index(max(_scores))
        pred_label_norm = norm_scores.index(max(norm_scores))
        correct += int(pred_label == label)
        correct_norm += int(pred_label_norm == label)
        print(
            f"{total} acc: {correct/total:.4f} "
            f"acc_norm: {correct_norm}/{total}={correct_norm/total:.4f}"
        )


@register_bin(name="eval_hellaswag")
class EvalHellaswag(BaseBin):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "-config",
            "--config",
            "-c",
            required=False,
            help="Path of main YAML config file.",
        )
        add_model(parser, PredictConfig)

    def run(args):
        # same as in eole.bin.translate, should be improved/factorized
        if args.config is not None:
            with open(args.config) as f:
                config = yaml.safe_load(os.path.expandvars(f.read()))
        else:
            config = {}
        _parser = ArgumentParser()
        add_model(_parser, PredictConfig)
        defaults = vars(_parser.parse_args([]))
        stuff_to_update = get_non_default_values(args, defaults)
        config.update(stuff_to_update)

        # pop extra fields added by argparse, not supported in pydantic configs
        if "bin" in config.keys():
            config.pop("bin")
        if "sub_bin" in config.keys():
            config.pop("sub_bin")
        if "config" in config.keys():
            config.pop("config")

        config = PredictConfig(**config)

        evaluate(config)
