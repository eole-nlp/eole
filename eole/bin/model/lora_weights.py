import torch
from eole.utils.logging import init_logger, logger
from eole.models.model_saver import load_checkpoint
from eole.inputters.inputter import dict_to_vocabs, vocabs_to_dict
from eole.config import recursive_model_fields_set
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import glob
import json
import os

from eole.bin import BaseBin, register_bin

"""
    This script merges or concat LoRa weights into the main model
    * merge
        the weights of LoRa are merged and we save the merged model without
        the optimizer hence in the same format as the original base model.
    * concat
        the weights of LoRa are added to the base model in the way the model
        can be further trained as it was stopped. The Optimizer is also
        restored from the LoRa checkpoint
"""


@register_bin(name="lora")
class LoraWeights(BaseBin):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--action",
            type=str,
            required=False,
            default="merge",
            choices=["merge", "concat"],
            help="""Path to the model directory""",
        )
        parser.add_argument("--base_model", type=str, required=True, help="""Path to the base model""")
        parser.add_argument(
            "--lora_weights",
            type=str,
            required=True,
            help="""Path to the lora checkpoint""",
        )
        parser.add_argument("--output", type=str, required=True, help="""Path to the output model""")

    @classmethod
    def run(cls, args):
        init_logger()
        config_path = os.path.join(args.base_model, "config.json")
        with open(config_path) as f:
            config = json.load(f)
            inference_config = config.get("inference", None)
        # base_checkpoint = load_checkpoint(args.base_model)
        lora_checkpoint = load_checkpoint(args.lora_weights)
        vocabs = dict_to_vocabs(lora_checkpoint["vocab"])
        config = lora_checkpoint["config"]
        running_config = config.training
        running_config.quant_layers = []  # we need to remove any quantization to merge weights
        # running_config.lora_layers = []
        running_config.parallel_mode = "data_parallel"

        logger.info("Load state_dict from lora weights")
        lora_dict = safe_open(args.lora_weights + "/model.00.safetensors", framework="pt", device="cpu")
        lora_keys = []
        non_lora_keys = []
        for key in lora_dict.keys():
            if "lora" in key:
                lora_keys.append(".".join(key.split(".")[:-1]))
            else:  # for example estimator keys added with lora training
                non_lora_keys.append(key)
        lora_keys = list(set(lora_keys))
        non_lora_keys = non_lora_keys

        scaling = config.training.lora_alpha / config.training.lora_rank

        shards = sorted(glob.glob(os.path.join(args.base_model, "model.*.safetensors")))

        os.makedirs(args.output, exist_ok=True)
        for i, shard in enumerate(shards):
            shard_dict = load_file(shard, device="cpu")
            extra_dict = {}
            for key in shard_dict.keys():
                module_name = ".".join(key.split(".")[:-1])
                if module_name in lora_keys:
                    if args.action == "merge":
                        lora_A = lora_dict.get_tensor(module_name + ".lora_A").transpose(0, 1)
                        lora_B = lora_dict.get_tensor(module_name + ".lora_B").transpose(0, 1)
                        shard_dict[key] += (lora_A @ lora_B).transpose(0, 1) * scaling
                    elif args.action == "concat":
                        extra_dict[module_name + ".lora_A"] = lora_dict.get_tensor(module_name + ".lora_A")
                        extra_dict[module_name + ".lora_B"] = lora_dict.get_tensor(module_name + ".lora_B")
                    else:
                        raise ValueError("action not supported, please choose merge or concat")
                elif key in non_lora_keys:
                    logger.warning("same key: %s found in base and lora_weight but not a Lora parameter" % key)
            shard_dict |= extra_dict
            if i == 0:
                for key in non_lora_keys:
                    shard_dict[key] = lora_dict.get_tensor(key)
            logger.info("saving shard" + args.output + "/model.{:02d}.safetensors".format(i))
            save_file(
                shard_dict,
                os.path.join(args.output, "model.{:02d}.safetensors".format(i)),
            )

        if args.action == "merge":
            optim = None
            new_config = config  # base_checkpoint["config"]
            # use compute_dtype from lora finetuning
            new_config.training.compute_dtype = config.training.compute_dtype
        elif args.action == "concat":
            optim = lora_checkpoint["optim"]
            new_config = config
        else:
            raise ValueError("action not supported, please choose merge or concat")

        # save optimizer
        optim_path = os.path.join(args.output, "optimizer.pt")
        torch.save(optim, optim_path)
        # save vocab
        vocab_dict = vocabs_to_dict(vocabs)
        with open(os.path.join(args.output, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(vocab_dict, f, indent=2, ensure_ascii=False)
        # save config
        config_dict = recursive_model_fields_set(new_config)
        if inference_config is not None:
            config_dict["inference"] = inference_config
        with open(os.path.join(args.output, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
