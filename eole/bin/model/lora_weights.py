import torch
from eole.utils.logging import init_logger, logger
from eole.models.model_saver import load_checkpoint
from eole.models.model import get_model_class
from eole.inputters.inputter import dict_to_vocabs, vocabs_to_dict
from eole.config import recursive_model_fields_set
from safetensors import safe_open
from safetensors.torch import save_file
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
        base_checkpoint = load_checkpoint(args.base_model)
        lora_checkpoint = load_checkpoint(args.lora_weights)
        vocabs = dict_to_vocabs(lora_checkpoint["vocab"])
        config = lora_checkpoint["config"]
        running_config = config.training
        running_config.quant_layers = []  # we need to remove any quantization to merge weights
        running_config.parallel_mode = "data_parallel"
        model_config = config.model
        model = get_model_class(model_config).build_base_model(model_config, vocabs, running_config=running_config)
        logger.info("Load state_dict from base_model")
        model.load_safe_state_dict(
            args.base_model,
            device=torch.device("cpu"),
            strict=False,
        )
        logger.info("Load state_dict from lora_weights")
        model.load_safe_state_dict(
            args.lora_weights,
            device=torch.device("cpu"),
            strict=False,
        )

        if args.action == "merge":
            model.eval()  # this merges automatically LoRa weights in main
            optim = None
            model_state_dict = model.state_dict()
            new_config = base_checkpoint["config"]
            # use compute_dtype from lora finetuning
            new_config.training.compute_dtype = config.training.compute_dtype
        elif args.action == "concat":
            optim = lora_checkpoint["optim"]
            model_state_dict = model.state_dict()
            new_config = config
        else:
            raise ValueError("action not supported, please choose merge or concat")
        os.makedirs(args.output, exist_ok=True)
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
        shards = glob.glob(os.path.join(args.base_model, "model.*.safetensors"))
        f = []
        for i, shard in enumerate(shards):
            shard_dict = {}
            f.append(safe_open(shard, framework="pt", device="cpu"))
            for key in f[i].keys():
                shard_dict[key] = model_state_dict[key].to(running_config.storage_dtype)
            if i == 0:
                for key in model_state_dict.keys():
                    if "estimator" in key:
                        shard_dict[key] = model_state_dict[key].to(running_config.storage_dtype)
            logger.info("saving shard" + args.output + "/model.{:02d}.safetensors".format(i))
            save_file(
                shard_dict,
                os.path.join(args.output, "model.{:02d}.safetensors".format(i)),
            )
