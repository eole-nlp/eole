"""
    Code taken and adapted from https://github.com/FranxYao/chain-of-thought-hub
"""

import json
import yaml
import os
import time
from argparse import ArgumentParser
from eole.utils.logging import init_logger
from eole.inference_engine import InferenceEnginePY
from eole.utils.misc import use_gpu, set_random_seed

from eole.bin import BaseBin, register_bin
from eole.config.cli import add_model
from eole.config import get_non_default_values
from eole.config.run import PredictConfig

TASKS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

choices = ["A", "B", "C", "D"]


def compute_metric(output_filename):
    with open(output_filename, "r") as f:
        run_results = json.load(f)
    total_acc = 0
    total_num = 0
    for task in run_results:
        acc = 0
        pred_answers = run_results[task]["pred_answers"]
        gold_answers = run_results[task]["gold_answers"]
        for pred, gold in zip(pred_answers, gold_answers):
            if pred == gold:
                acc += 1
        print("ACC-%s: %.4f" % (task, acc / len(gold_answers)))
        total_acc += acc
        total_num += len(gold_answers)
    print("ACC-all: %.4f" % (total_acc / total_num))


def format_subject(subject):
    subl = subject.split("_")
    s = ""
    for entry in subl:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


# def custom_stopping_criteria(input_ids, score, **kwargs):
#     stop_ids = [29871, 13, 13] # \n\n
#     return input_ids[-len(stop_ids)]


def evaluate(opt):
    import pandas as pd

    logger = init_logger(opt.log_file)
    set_random_seed(opt.seed, use_gpu(opt))

    run_results = {}
    dir_name = os.path.dirname(opt.model_path[0])

    output_filename = os.path.join(dir_name, "mmlu_results.json")

    # Build the translator (along with the model)
    engine = InferenceEnginePY(opt)

    # this considers that we are always in the recipes/mmlu folder, should be improved
    data_dir = "recipes/mmlu/data/"
    ntrain = 5  # nshots from dev

    start_time = time.time()
    for task in TASKS:
        logger.info("Testing %s ..." % task)
        records = []
        src = []
        dev_df = pd.read_csv(
            os.path.join(data_dir, "dev", task + "_dev.csv"), header=None
        )[:ntrain]
        test_df = pd.read_csv(
            os.path.join(data_dir, "test", task + "_test.csv"), header=None
        )
        for i in range(test_df.shape[0]):
            # get prompt and make sure it fits
            k = ntrain
            prompt_end = format_example(test_df, i, include_answer=False)
            train_prompt = gen_prompt(dev_df, task, k)
            prompt = train_prompt + prompt_end

            while len(prompt.split(" ")) > 768:
                prompt_split = prompt.split("\n\n")
                prompt_split.pop(1)
                prompt = "\n\n".join(prompt_split)

            label = test_df.iloc[i, test_df.shape[1] - 1]
            records.append({"prompt": prompt, "answer": label})
            src.append(prompt.replace("\n", "｟newline｠"))

        scores, _, preds = engine.infer_list(src)

        pred_answers = [
            x.lstrip() for sublist in preds for x in sublist
        ]  # flatten the list of list

        gold_answers = [record["answer"] for record in records]
        run_results[task] = {"pred_answers": pred_answers, "gold_answers": gold_answers}

    engine.terminate()

    with open(output_filename, "w") as f:
        json.dump(run_results, f, ensure_ascii=False, indent=2)

    compute_metric(output_filename)
    end_time = time.time()
    logger.info("total run time %.2f" % (end_time - start_time))


@register_bin(name="run_mmlu")
class RunMMLU(BaseBin):
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
