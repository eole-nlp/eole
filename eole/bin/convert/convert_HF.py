#!/usr/bin/env python
# Standard Library Imports
import configparser
import json
import logging
import math
import os
import re
from dataclasses import dataclass, field, fields
from typing import Optional

# Third-Party Library Imports
import huggingface_hub
import pyonmttok
import safetensors
import torch
from huggingface_hub import hf_hub_download, utils
from safetensors.torch import save_file
from sentencepiece import SentencePieceProcessor

# Eole Imports
from eole.bin import BaseBin, register_bin
from eole.config import recursive_model_fields_set
from eole.config.run import TrainConfig
from eole.config.training import TrainingConfig
from eole.constants import DefaultTokens, TORCH_DTYPES, PositionEncodingType
from eole.inputters.inputter import vocabs_to_dict

from eole.bin.convert.HF_mappings import (
    KEY_MAPS,
    ACT_TABLE,
    LN_TABLE,
    ARCH_TABLE,
    TOK_TABLE,
)


def get_sentencepiece_vocab(model_path):
    """Get the vocabulary from a SentencePiece model.

    Args:
        model_path (str): Path to the SentencePiece model file

    Returns:
        list: The vocabulary as a list of strings
    """
    sp_model = SentencePieceProcessor(model_file=model_path)
    n_words = sp_model.vocab_size()
    return [sp_model.id_to_piece(i) for i in range(n_words)]


@dataclass
class HuggingfaceFiles:
    """
    Helper class to centralize HF files and config processing.
    """

    model_dir: str
    token: str
    # files retrieved from the hub
    config_path: str
    tokenizer_config_json: str
    generation_config_json: Optional[str] = None
    tokenizer_model: Optional[str] = None
    tokenizer_json: Optional[str] = None
    wmap_path: Optional[str] = None
    model_path: Optional[str] = None
    special_tokens_json: Optional[str] = None

    # Unified dictionary to cache loaded files
    _loaded_files: dict = field(default_factory=dict, init=False)

    @classmethod
    def fetch(cls, args, mode="local"):
        """
        Factory method to fetch files either locally or from the HuggingFace Hub.

        Args:
            args: Namespace containing model_dir, token, output, etc.
            mode: Either 'local' or 'hub', determines the source of the files.

        Returns:
            HuggingfaceFiles instance with all paths populated.
        """

        if os.path.exists(args.model_dir):
            mode = "local"
        else:
            mode = "hub"

        def get_local_path(file_name, required=False):
            """Helper to check if a file exists in the model directory."""
            path = os.path.join(args.model_dir, file_name)
            if os.path.exists(path):
                return path
            if required:
                raise FileNotFoundError(f"Required file '{file_name}' is missing locally.")
            return None

        def download_file_from_hub(file_name, required=True):
            """Helper to download a file from the HuggingFace Hub."""
            try:
                return hf_hub_download(
                    repo_id=args.model_dir,
                    filename=file_name,
                    token=args.token,
                    local_dir=args.output,
                )
            except utils.EntryNotFoundError:
                if required:
                    raise utils.EntryNotFoundError(f"Required file '{file_name}' is missing from the repository.")
                return None

        get_file_fn = get_local_path if mode == "local" else download_file_from_hub

        os.makedirs(args.output, exist_ok=True)  # Ensure output directory exists

        # Fetch required and optional files
        paths = {
            "config_path": get_file_fn("config.json", required=True),
            "tokenizer_config_json": get_file_fn("tokenizer_config.json", required=True),
            "generation_config_json": get_file_fn("generation_config.json", required=False),
            "tokenizer_model": get_file_fn("tokenizer.model", required=False)
            or get_file_fn("sentencepiece.bpe.model", required=False),
            "tokenizer_json": get_file_fn("tokenizer.json", required=False),
            "wmap_path": get_file_fn("model.safetensors.index.json", required=False)
            or get_file_fn("pytorch_model.bin.index.json", required=False),
            "model_path": get_file_fn("model.safetensors", required=False)
            or get_file_fn("pytorch_model.bin", required=False),
            "special_tokens_json": get_file_fn("special_tokens_map.json", required=False),
        }

        return cls(**paths, model_dir=args.model_dir, token=args.token)

    def _get_file_content(self, file_attr):
        """Combined method to load and cache file contents."""
        file_path = getattr(self, file_attr, None)
        if not file_path or not os.path.exists(file_path):
            return None

        if file_path in self._loaded_files:
            return self._loaded_files[file_path]

        with open(file_path, encoding="utf-8") as f:
            content = json.load(f) if file_path.endswith(".json") else f.read()

        self._loaded_files[file_path] = content
        return content

    def __getattr__(self, name):
        """Generic attribute getter for file contents."""
        file_fields = {field.name for field in fields(self)}
        if f"{name}_json" in file_fields:
            return self._get_file_content(f"{name}_json")
        elif f"{name}_path" in file_fields:
            return self._get_file_content(f"{name}_path")
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    @property
    def arch(self):
        return self.config["architectures"][0]

    @property
    def vocab_size(self):
        config = self.config.get("text_config", self.config)
        if "vocab_size" in config:
            return config["vocab_size"]
        tokenizer = self.tokenizer
        if tokenizer:
            max_added_tokens = max([token["id"] for token in tokenizer["added_tokens"]])
            max_tokens = max([token_id for token_id in tokenizer["model"]["vocab"].values()])
            return max(max_added_tokens, max_tokens)
        raise NotImplementedError("Vocabulary size could not be determined.")

    @property
    def encoder_layer_prefix(self):
        return KEY_MAPS[self.arch].get("encoder_layer_prefix", None)

    @property
    def encoder_sam_layer_prefix(self):
        return KEY_MAPS[self.arch].get("encoder_sam_layer_prefix", None)

    @property
    def decoder_layer_prefix(self):
        return KEY_MAPS[self.arch].get("decoder_layer_prefix", None)

    @property
    def base_dir(self):
        return os.path.split(self.wmap_path)[0]

    def get_load_ckpt(self, dir_path, file_path):
        if os.path.exists(os.path.join(dir_path, file_path)):
            ckpt_path = os.path.join(dir_path, file_path)
        else:
            try:
                ckpt_path = huggingface_hub.hf_hub_download(
                    repo_id=self.model_dir,
                    filename=file_path,
                    token=self.token,
                )
            except huggingface_hub.utils.EntryNotFoundError:
                raise huggingface_hub.utils.EntryNotFoundError("Checkpoint not found on the hub")
            except PermissionError:
                ckpt_path = os.path.join(dir_path, file_path)
        if ckpt_path[-4:] == ".bin":
            checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
        else:
            checkpoint = ckpt_path

        return checkpoint

    def checkpoint(self, ckpt):
        if self.wmap_path:
            checkpoint = self.get_load_ckpt(self.base_dir, ckpt)
        else:
            checkpoint = self.get_load_ckpt(*os.path.split(self.model_path))
        return checkpoint


def get_git_remote_url(model_dir):
    """Gets the git remote URL from a model directory.

    Args:
        model_dir (str): Path to model directory containing .git folder

    Returns:
        str: Repository name in format "owner/repo", or None if not found
    """
    config_path = os.path.join(model_dir, ".git", "config")
    config = configparser.ConfigParser()
    config.read(config_path)
    # Get the URL from the 'remote "origin"' section
    try:
        url = config['remote "origin"']["url"]
        return "/".join(url.split("/")[-2:])
    except KeyError:
        return None


def get_huggingface_model(args):
    """Gets HuggingFace model identifier from directory path or direct model name.

    Args:
        args: Arguments containing model_dir attribute with path to model directory or model identifier

    Returns:
        str: HuggingFace model identifier
    """
    if os.path.exists(args.model_dir):
        huggingface_model = get_git_remote_url(args.model_dir)
        if huggingface_model is None:
            logging.warning("huggingface_model could not be retrieved from model_dir git config")
    else:
        huggingface_model = args.model_dir
    return huggingface_model


def build_config_dict(hf):
    """Convert Hugging Face model configuration to eole config format.

    This function takes a HuggingfaceFiles dataclass object and extracts configuration parameters
    to create dictionaries for model architecture, training settings and various hyper-parameters.

    Args:
        hf: HuggingfaceFiles dataclass object containing config and architecture information

    Returns:
        tuple: Contains:
            - model_config (dict): Model architecture configuration
            - training_config (dict): Training and quantization settings
            - params (list): List of parameter names to extract
    """
    config = hf.config
    arch = hf.arch
    print("Architecture: ", arch)

    vision_config = config.get("vision_config", None)
    other_config = config  # save what is not text/vision for later use
    config = config.get("text_config", config.get("language_config", config))

    model_config = {}
    training_config = {}

    # Initialize model_config with defaults and fallbacks
    model_config = {
        "layers": config.get("num_hidden_layers", config.get("n_layer", config.get("n_layers"))),
        "hidden_size": config.get("hidden_size", config.get("n_embd", config.get("hidden_dim", config.get("d_model")))),
        "heads": config.get(
            "num_attention_heads",
            config.get("n_head", config.get("n_heads", config.get("decoder_attention_heads", 32))),
        ),  # default 32 patch for mistral-community/pixtral-12b
        "transformer_ff": config.get("intermediate_size", config.get("decoder_ffn_dim", None)),
        "mlp_activation_fn": ACT_TABLE[arch],
        "layer_norm": LN_TABLE[arch],
        "heads_kv": config.get("multi_query", False)
        or config.get(
            "num_key_value_heads",
            config.get(
                "num_kv_heads",
                config.get("n_head_kv", config.get("num_attention_heads", config.get("n_head"))),
            ),
        ),
        "head_dim": config.get("head_dim"),
        "attn_scaling": config.get("query_pre_attn_scalar", None),
        "parallel_residual": config.get("parallel_attn", False),
        "norm_eps": config.get(
            "rms_norm_eps",
            config.get("layer_norm_epsilon", config.get("layer_norm_eps", 1e-5)),
        ),
        "sliding_window": config.get("sliding_window", 0) or 4096,
        "num_experts": config.get("num_local_experts", config.get("n_routed_experts", config.get("num_experts", 0))),
        "num_shared_experts": config.get("n_shared_experts", 0),
        "first_k_dense_replace": config.get("first_k_dense_replace", 0),
        "num_experts_per_tok": config.get("num_experts_per_tok", 0),
        "moe_transformer_ff": config.get("moe_intermediate_size", None),
        "add_qkvbias": False,
        "add_final_linear_bias": False,
        "add_ffnbias": False,
        "shared_layer_norm": False,
        "left_pad": True,
        "generator_bias": False,
        "adapter_bias": False,
        "rope_config": {
            "rotary_interleave": False,
            "rotary_theta": config.get("rope_theta", config.get("rope_parameters", {}).get("rope_theta", 10000)),
        },
        "embeddings": {},  # Populated later
    }
    if model_config["num_experts"] == 1:
        model_config["num_experts"] = 0

    # Vision encoder
    if vision_config is not None:
        model_config["encoder"] = {
            "layers": vision_config.get("num_hidden_layers", 24),  # hard-coded for mistral-community/pixtral-12b
            "hidden_size": vision_config.get("hidden_size", vision_config.get("image_size", 0)),
            "transformer_ff": vision_config.get("intermediate_size", vision_config.get("image_size", 0) * 4),
            "image_size": vision_config.get("image_size", 0),
            "patch_size": vision_config.get("patch_size", 0),
            "heads": vision_config.get(
                "num_attention_heads", vision_config.get("image_size", 0) / vision_config.get("head_dim", 1)
            ),
            "heads_kv": vision_config.get(
                "num_attention_heads", vision_config.get("image_size", 0) / vision_config.get("head_dim", 1)
            ),
            "head_dim": vision_config.get(
                "head_dim", vision_config.get("hidden_size", 0) // vision_config.get("num_attention_heads", 1)
            ),
        }
    if arch in ["LlavaForConditionalGeneration", "Mistral3ForConditionalGeneration"]:
        model_config["encoder"].update(
            {
                "mlp_activation_fn": model_config["mlp_activation_fn"],
                "layer_norm": model_config["layer_norm"],
                "norm_eps": model_config["norm_eps"],
                "rope_config": {
                    "rotary_theta": vision_config["rope_theta"],
                    "rotary_interleave": False,
                },
            }
        )
        model_config["adapter_bias"] = other_config.get("multimodal_projector_bias", False)
        model_config["projector_activation_fn"] = other_config.get("projector_hidden_act", "gelu")
        model_config["spatial_merge_size"] = other_config.get("spatial_merge_size", None)

    if arch == "Gemma3ForConditionalGeneration":
        if model_config.get("head_dim", None) is None:
            model_config["head_dim"] = 256  # src/transformers/models/gemma3/configuration_gemma3.py#L61
        if model_config.get("heads_kv", None) is None:
            model_config["heads_kv"] = 4
        if model_config.get("heads", None) is None:
            model_config["heads"] = 8
        model_config["encoder"].update(
            {
                "n_positions": (vision_config["image_size"] // vision_config["patch_size"]) ** 2,
                # head related stuff patched to match 1152 dim of siglip
                # https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py#L399-L402
                "mm_tokens_per_image": hf.config["mm_tokens_per_image"],
                "image_token_id": hf.config["image_token_index"],
            }
        )

    if arch in ["DeepseekOCRForCausalLM"]:
        model_config["encoder"].update(
            {
                "n_positions": (
                    vision_config["width"]["clip-l-14-224"]["image_size"]
                    // vision_config["width"]["clip-l-14-224"]["patch_size"]
                )
                ** 2
                + 1,
                "patch_size": vision_config["width"]["clip-l-14-224"].get("patch_size", 14),
                "heads": vision_config["width"]["clip-l-14-224"].get("heads"),
                "heads_kv": vision_config["width"]["clip-l-14-224"].get("heads"),
            }
        )

    if arch in ["HunYuanVLForConditionalGeneration"]:
        model_config["encoder"].update({})
        model_config["spatial_merge_size"] = vision_config.get("spatial_merge_size", None)

    # patch transformer_ff
    if model_config["transformer_ff"] is None:
        model_config["transformer_ff"] = model_config["hidden_size"] * 4

    # patch sliding window
    if model_config["sliding_window"] is None:
        model_config["sliding_window"] = 4096

    # Populate embeddings
    model_config["embeddings"] = {
        "src_word_vec_size": model_config["hidden_size"],
        "tgt_word_vec_size": model_config["hidden_size"],
    }

    # Default position encoding configuration
    model_config["embeddings"].update(
        {
            "position_encoding_type": PositionEncodingType.Rotary,
            "n_positions": 0,
        }
    )

    # patch rotary dim
    if "rotary_dim" in config.keys():
        model_config["rope_config"]["rotary_dim"] = config["rotary_dim"]
    elif "partial_rotary_factor" in config.keys():
        model_config["rope_config"]["rotary_dim"] = int(
            config["partial_rotary_factor"] * (model_config["hidden_size"] // model_config["heads"])
        )
    elif model_config.get("head_dim", None) is not None:
        model_config["rope_config"]["rotary_dim"] = model_config["head_dim"]
    else:
        model_config["rope_config"]["rotary_dim"] = model_config["hidden_size"] // model_config["heads"]

    # Update rope scaling related settings
    if config.get("rope_scaling", None) is not None:
        model_config["rope_config"].update(
            {
                "scaling_type": config["rope_scaling"].get("rope_type", config["rope_scaling"].get("type", None)),
                "scaling_factor": config["rope_scaling"].get("factor", 8.0),
                "alpha": config["rope_scaling"].get("alpha", None),
                "xdrope_section": config["rope_scaling"].get("xdrope_section", None),
                "low_freq_factor": config["rope_scaling"].get("low_freq_factor", 1.0),
                "high_freq_factor": config["rope_scaling"].get("high_freq_factor", 4.0),
                "original_max_position_embeddings": config["rope_scaling"].get(
                    "original_max_position_embeddings", config.get("max_position_embeddings", 8192)
                ),
            }
        )

    # Validate required fields
    required_fields = {
        "layers": "Can't find the number of layers in the config.json file",
        "hidden_size": "Can't find the model hidden size in the config.json file",
        "heads": "Can't find the number of heads in the config.json file",
    }

    for key, error_msg in required_fields.items():
        if model_config[key] is None:
            raise ValueError(error_msg)

    # Handle quantization
    quant_config = config.get("quantization_config", {})
    if quant_config:
        if quant_config.get("quant_method") == "awq":
            backend = quant_config.get("backend", "autoawq")
            quant_type_mapping = {
                "llm-awq": "awq_gemv",
                "autoawq": {"gemm": "awq_gemm", "gemv": "awq_gemv"},
            }
            quant_type = quant_type_mapping.get(backend, "").get(quant_config.get("version").lower(), "")
            if not quant_type:
                raise ValueError("Unknown quantization config")
        else:
            raise ValueError("Can convert only awq models for now")
        training_config["quant_type"] = quant_type
        training_config["w_bit"] = quant_config.get("bits", quant_config.get("w_bit", 0))
        training_config["group_size"] = quant_config.get("group_size", quant_config.get("q_group_size", 0))
        training_config["quant_layers"] = [
            "gate_up_proj",
            "down_proj",
            "up_proj",
            "linear_values",
            "linear_query",
            "linear_keys",
            "final_linear",
        ]
        params = ["qweight", "qzeros", "scales"] + ["weight", "bias"]
    else:
        training_config["quant_type"] = ""
        training_config["w_bit"] = 0
        training_config["group_size"] = 0
        training_config["quant_layers"] = []
        params = ["weight", "bias"]

    model_config["share_decoder_embeddings"] = config.get("tie_word_embeddings", False)

    # Update model_config based on architecture
    if arch in KEY_MAPS:
        for key, value in KEY_MAPS[arch].get("config", {}).items():
            if isinstance(value, dict):
                # Update nested dictionaries
                model_config[key] = {**model_config.get(key, {}), **value}
            else:
                # Update regular keys
                model_config[key] = value

    return model_config, training_config, params


def get_weight(checkpoint, tensor_name):
    if isinstance(checkpoint, dict):
        if tensor_name in checkpoint.keys():
            return checkpoint[tensor_name].contiguous()
    else:
        with safetensors.safe_open(checkpoint, framework="pt", device="cpu") as f:
            if tensor_name in f.keys():
                return f.get_tensor(tensor_name).contiguous()
    return None


def check_tokenizer_config(hf):
    config = hf.config
    add_bos_token = hf.tokenizer_config.get("add_bos_token", hf.tokenizer_config.get("bos_token", None) is not None)
    chat_template = {"chat_template": hf.tokenizer_config.get("chat_template", None)}
    eos_token_id = config.get("eos_token_id", None)
    optional_eos = []
    if isinstance(eos_token_id, list) and "added_tokens_decoder" in hf.tokenizer_config:
        eos_tokens = [hf.tokenizer_config["added_tokens_decoder"][str(index)]["content"] for index in eos_token_id]
        optional_eos = eos_tokens[1:]
    # Automatically convert added_tokens into mapped_tokens
    mapped_tokens = [
        (
            token["content"],
            re.sub(r"<\|([^|]*)\|>", "\uff5f\\1\uff60", token["content"]),
        )
        for token in hf.tokenizer_config.get("added_tokens_decoder", {}).values()
    ]
    return (add_bos_token, chat_template, optional_eos, mapped_tokens)


def check_special_tokens(hf):
    vocabs = {"specials": {}}
    if hf.special_tokens is not None:
        special_tokens_map = hf.special_tokens
        for token_name in ["bos_token", "unk_token", "eos_token", "pad_token"]:
            token = special_tokens_map.get(token_name, None)
            if isinstance(token, list):
                vocabs["specials"][token_name] = token[0]
            elif isinstance(token, str):
                vocabs["specials"][token_name] = token
            elif isinstance(token, dict):
                vocabs["specials"][token_name] = token["content"]
    elif hf.tokenizer_config is not None:
        for token_name in ["bos_token", "unk_token", "eos_token", "pad_token"]:
            token = hf.tokenizer_config.get(token_name, None)
            if isinstance(token, list):
                vocabs["specials"][token_name] = token[0]
            elif isinstance(token, str):
                vocabs["specials"][token_name] = token
            elif isinstance(token, dict):
                vocabs["specials"][token_name] = token["content"]
    return vocabs


def check_generation_config(hf):
    if hf.generation_config is None:
        return {}
    generation_config_dict = {}
    # we probably need a better mapping at some point
    keys = ["top_k", "top_p", "temperature", "max_length"]
    generation_config_dict = {key: hf.generation_config[key] for key in keys if key in hf.generation_config}
    return generation_config_dict


def get_shards_map(model_config, hf, nshards):
    """
    Distributes model checkpoints across shards based on layer ranges.

    Args:
        model_config (dict): Configuration dictionary containing model parameters,
            including number of layers.
        hf (object): HuggingfaceFiles dataclass object containing model path and weight mapping.
        nshards (int): Number of shards to distribute the model across.

    Returns:
        tuple: A tuple containing:
            - list of lists: Checkpoint paths for each shard
            - list of ranges: Layer ranges assigned to each shard
    """
    # Compute layer range for each shard
    if "encoder" in model_config.keys():
        n_layers = max(model_config["layers"], model_config["encoder"]["layers"])
    else:
        n_layers = model_config["layers"]
    layers_per_shard = math.ceil(n_layers / nshards)
    shard_layer_ranges = [
        range(
            layers_per_shard * shard,
            min(layers_per_shard * (shard + 1), n_layers),
        )
        for shard in range(nshards)
    ]
    if hf.wmap_path is None:
        return [[hf.model_path]] * nshards, shard_layer_ranges

    weightmap = hf.wmap["weight_map"]

    # Initialize a list of checkpoint lists, one for each shard
    shard_checkpoints = [set() for _ in range(nshards)]

    # Check if a layer key belongs to the current shard
    def is_layer_in_range(key, prefix, layer_range):
        return prefix and key.startswith(prefix) and int(key.split(".")[len(prefix.split(".")) - 1]) in layer_range

    # Loop over the weightmap and distribute checkpoints to the appropriate shards
    for key, ckpt in weightmap.items():
        for shard, layer_range in enumerate(shard_layer_ranges):
            if is_layer_in_range(key, hf.decoder_layer_prefix, layer_range) or is_layer_in_range(
                key, hf.encoder_layer_prefix, layer_range
            ):
                shard_checkpoints[shard].add(ckpt)

    return shard_checkpoints, shard_layer_ranges


def build_shards(model_config, hf, args, params):
    """
    Build sharded model files from HuggingFace checkpoint.

    Args:
        model_config (dict): Configuration dictionary containing model architecture settings
        hf (HuggingfaceFiles): dataclass containing model files and configuration
        args (Namespace): Command line arguments containing nshards and output path
        params (list): List of parameter names to convert

    The function splits the model into shards and saves them as safetensor files.
    Layer parameters are distributed across shards based on the sharding configuration.
    The first shard contains embeddings and model-level parameters on top of its layer split.
    """
    shard_checkpoints, shard_layer_ranges = get_shards_map(model_config, hf, args.nshards)

    for shard in range(args.nshards):

        print("starting output shard: %d/%d" % (shard + 1, args.nshards))
        eole_safetensor = {}

        def build_first_shard(hf, eole_safetensor):
            for target in KEY_MAPS[hf.arch].keys():
                if model_config["share_decoder_embeddings"] and target == "generator.weight":
                    continue
                source = KEY_MAPS[hf.arch][target]
                srckey, srcmap = source if isinstance(source, tuple) else (source, None)
                if isinstance(srckey, str):
                    if hf.wmap_path:
                        if srckey in hf.wmap["weight_map"]:
                            checkpoint = hf.get_load_ckpt(
                                hf.base_dir,
                                hf.wmap["weight_map"][srckey],
                            )
                        else:
                            checkpoint = None
                    else:
                        checkpoint = hf.get_load_ckpt(*os.path.split(hf.model_path))
                else:
                    checkpoint = None
                if checkpoint is None:
                    continue
                w = get_weight(checkpoint, srckey)
                if w is not None:
                    if srcmap is not None:
                        w = eval(
                            "w" + srcmap,
                            {
                                "w": w,
                                "hidden_size": model_config["hidden_size"],
                                "transformer_ff": model_config["transformer_ff"],
                            },
                        ).contiguous()
                    if target == "encoder.class_embedding.weight":
                        eole_safetensor[target] = w.unsqueeze(0)
                    else:
                        eole_safetensor[target] = w
            return eole_safetensor

        if shard == 0:
            eole_safetensor = build_first_shard(hf, eole_safetensor)

        # TODO: could we reverse the mapping and loop on params instead? (would reduce conditions)
        for ckpt in shard_checkpoints[shard]:
            print("Loading %s" % ckpt)
            checkpoint = hf.checkpoint(ckpt)
            for i in shard_layer_ranges[shard]:
                prefix_mapping = (
                    ("encoder", hf.encoder_layer_prefix, "encoder.transformer_layers."),
                    ("encoder.sam", hf.encoder_sam_layer_prefix, "encoder.sam.blocks."),
                    ("decoder", hf.decoder_layer_prefix, "decoder.transformer_layers."),
                )
                for section, hf_prefix, eole_prefix in prefix_mapping:
                    if hf_prefix is None:
                        continue
                    key_map = KEY_MAPS[hf.arch].get(section, {})
                    for target, source in key_map.items():
                        for param in params:
                            # TODO: this should be cleaned up when rationalizing encoder/decoder mappings
                            if not (isinstance(source, str) or isinstance(source, tuple)):
                                continue
                            srckey, srcmap = source if isinstance(source, tuple) else (source, None)

                            if srckey.endswith("."):
                                srckey = srckey + param
                            w = get_weight(
                                checkpoint,
                                hf_prefix + str(i) + srckey,
                            )

                            if w is not None:
                                if srcmap is not None:
                                    hidden_size = (
                                        model_config["hidden_size"]
                                        if section.startswith("decoder")
                                        else model_config["encoder"]["hidden_size"]
                                    )
                                    w = eval(
                                        "w" + srcmap,
                                        {
                                            "w": w,
                                            "hidden_size": hidden_size,
                                            "transformer_ff": model_config["transformer_ff"],
                                        },
                                    ).contiguous()
                                target1 = target
                                if target.endswith("."):
                                    target1 = target + param
                                eole_safetensor[eole_prefix + str(i) + target1] = w

        # Convert to another dtype if specified
        if args.dtype is not None:
            for key in eole_safetensor.keys():
                eole_safetensor[key] = eole_safetensor[key].to(TORCH_DTYPES[args.dtype])
        print("Saving output model shard: %d" % shard)
        save_file(
            eole_safetensor,
            os.path.join(args.output, "model.{:02d}.safetensors".format(shard)),
        )


def check_sentencepiece_tokenizer(hf):
    tokenizer_basename = os.path.basename(hf.tokenizer_model)
    if hf.tokenizer_json is not None:
        tokenizer_vocab = hf.tokenizer["model"]["vocab"]
        if isinstance(tokenizer_vocab, dict):
            vocab = list(hf.tokenizer["model"]["vocab"].keys())
        elif isinstance(tokenizer_vocab, list):
            vocab = [token for token, freq in tokenizer_vocab]
        else:
            raise NotImplementedError(f"Type {type(tokenizer_vocab)} is not supported for SentencePiece vocab.")
    else:
        vocab = get_sentencepiece_vocab(hf.tokenizer_model)
        if hf.tokenizer_json is not None:
            # We need to add 'added_tokens' that are not in the SP model
            newtokens = [tok["content"] for tok in hf.tokenizer["added_tokens"] if tok["content"] not in vocab]
            vocab.extend(newtokens)
            for tok in hf.tokenizer["added_tokens"]:
                vocab[tok["id"]] = tok["content"]
    src_vocab = pyonmttok.build_vocab_from_tokens(
        vocab,
    )
    return src_vocab, tokenizer_basename


def check_bpe_tokenizer(hf, vocabs, directory_path):
    config = hf.config.get("text_config", hf.config)
    vocab_size = hf.vocab_size
    # gpt2_pretok
    pretokenizers = hf.tokenizer.get("pre_tokenizer", {}).get("pretokenizers", [{}])
    pre_tokenizer = hf.tokenizer.get("pre_tokenizer", None)
    pretokenizers = pre_tokenizer.get("pretokenizers", None)
    if pretokenizers is None:
        pretokenizers = [pre_tokenizer]
    for pretokenizer in pretokenizers:
        if pretokenizer.get("type", None) == "ByteLevel":
            gpt2_pretok = True
    if "pad_token" in vocabs["specials"]:
        vocab = [tok for tok in hf.tokenizer["model"]["vocab"]]
    else:
        vocab = [
            tok if tok != "Ā" else DefaultTokens.PAD
            # "Ā" is '\x00' in unicode (cf tokenize.py gpt2 mapping)
            for tok in hf.tokenizer["model"]["vocab"]
        ]
        if DefaultTokens.PAD in vocab:
            vocabs["specials"]["pad_token"] = DefaultTokens.PAD
    voc_size = len(vocab)
    if vocab_size > voc_size:
        for i in range(vocab_size - voc_size):
            vocab.append(DefaultTokens.VOCAB_PAD + str(i))
    for tok in hf.tokenizer["added_tokens"]:
        vocab[tok["id"]] = tok["content"]
    src_vocab = pyonmttok.build_vocab_from_tokens(vocab)
    # TODO: not sure for which model(s) this is needed
    for token_name in ["bos_token", "unk_token", "eos_token", "pad_token"]:
        if f"{token_name}_id" in config.keys():
            token = config[f"{token_name}_id"]
            if isinstance(token, list):
                vocabs["specials"][token_name] = vocab[token[0]]
            elif isinstance(token, str):
                vocabs["specials"][token_name] = vocab[token]
            # elif isinstance(token, dict):
            #     vocabs["specials"][token_name] = token["content"]
    tokenizer_basename = "bpe.model"
    with open(os.path.join(directory_path, tokenizer_basename), "w", encoding="utf-8") as bpemodel:
        bpemodel.write("v3;false;false;false;Ġ;Ġ\n")
        for merge in hf.tokenizer["model"]["merges"]:
            if isinstance(merge, str):
                bpemodel.write(merge + "\n")
            elif isinstance(merge, list):
                bpemodel.write(" ".join(merge) + "\n")
            else:
                raise NotImplementedError(f"Type {type(merge)} is not supported for BPE merges.")
    return src_vocab, tokenizer_basename, vocabs, gpt2_pretok


def save_vocab(vocabs, src_vocab, directory_path):
    vocabs["src"] = src_vocab
    vocabs["tgt"] = src_vocab
    vocab_dict = vocabs_to_dict(vocabs)
    with open(os.path.join(directory_path, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, indent=2, ensure_ascii=False)

    with open(os.path.join(directory_path, "vocab.txt"), "w", encoding="utf-8") as vocabfile:
        for tok in vocab_dict["src"]:
            vocabfile.write(tok + "\n")


@register_bin(name="HF")
class LlamaHFConverter(BaseBin):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--model_dir",
            type=str,
            required=True,
            help="""Path to the input (to convert) model directory""",
        )

        parser.add_argument(
            "--output",
            type=str,
            required=True,
            help="""Path to the output (converted) model directory""",
        )
        parser.add_argument(
            "--nshards",
            type=int,
            default=1,
            help="""Number of safetensors shards weights will be spread across.""",
        )
        parser.add_argument(
            "--token",
            type=str,
            default="",
            help="""HF token""",
        )
        parser.add_argument(
            "--dtype",
            type=str,
            default=None,
            choices=TORCH_DTYPES.keys(),
            help="Specify which dtype to save model parameters into, " "default will keep the same as the input.",
        )
        parser.add_argument(
            "--tokenizer",
            type=str,
            default="hf",
            choices=["hf", "onmt"],
            help="Specify which tokenizer should be used by default.",
        )

    @classmethod
    def run(cls, args):
        hf = HuggingfaceFiles.fetch(args)

        # Build eole compatible configs from HF config
        model_config, training_config, params = build_config_dict(hf)
        gpt2_pretok = False

        # Deduce dtype from args or config, or default to fp16
        compute_dtype = args.dtype or hf.config.get("torch_dtype") or "fp16"

        # Check tokenizer and vocab related configuration
        (
            add_bos_token,
            chat_template,
            optional_eos,
            mapped_tokens,
        ) = check_tokenizer_config(hf)
        vocabs = check_special_tokens(hf)
        # TODO: could we have a better condition to check for sentencepiece/bpe?
        if hf.tokenizer_model is not None:  # sentencepiece mode
            src_subword_type = "sentencepiece"
            src_vocab, tokenizer_basename = check_sentencepiece_tokenizer(hf)
        else:  # BPE mode - we leverage the HF tokenizer.json info
            src_subword_type = "bpe"
            src_vocab, tokenizer_basename, vocabs, gpt2_pretok = check_bpe_tokenizer(hf, vocabs, args.output)

        # Configure transforms
        match TOK_TABLE[hf.arch], args.tokenizer:
            case "huggingface_tokenize", "hf":
                transforms = ["huggingface_tokenize"]
                transforms_configs = {
                    TOK_TABLE[hf.arch]: {"max_length": 512},
                }
            case _:
                transforms = ["onmt_tokenize", "filtertoolong"]
                transforms_configs = {
                    "filtertoolong": {"src_seq_length": 512, "tgt_seq_length": 512},
                    "onmt_tokenize": {
                        "src_subword_type": src_subword_type,
                        "src_subword_model": os.path.join("${MODEL_PATH}", tokenizer_basename),
                        "gpt2_pretok": gpt2_pretok,
                        "mapped_tokens": mapped_tokens,
                    },
                }

        if hf.config.get("decoder_start_token_id", None) is not None:
            vocabs["decoder_start_token"] = src_vocab.ids_to_tokens[hf.config["decoder_start_token_id"]]
        elif add_bos_token:
            vocabs["decoder_start_token"] = vocabs["specials"]["bos_token"]
        else:
            vocabs["decoder_start_token"] = ""

        # Check HF generation config for default settings
        generation_config_dict = check_generation_config(hf)

        # Save vocab files to output model directory
        save_vocab(vocabs, src_vocab, args.output)

        # Build shards
        build_shards(model_config, hf, args, params)

        # Build eole config and save to output model directory
        config = TrainConfig(
            data=None,
            skip_empty_level="silent",
            save_data=None,
            n_sample=0,
            src_vocab=None,
            tgt_vocab=None,
            share_vocab=True,
            src_vocab_size=hf.vocab_size,
            tgt_vocab_size=hf.vocab_size,
            vocab_size_multiple=8,
            decoder_start_token=vocabs["decoder_start_token"],
            **vocabs["specials"],
            transforms=transforms,
            transforms_configs=transforms_configs,
            model=ARCH_TABLE[hf.arch](
                **model_config,
                huggingface_model=get_huggingface_model(args),
            ),
            training=TrainingConfig(
                compute_dtype=compute_dtype,
                batch_size=896,
                batch_size_multiple=1,
                batch_type="tokens",
                normalization="tokens",
                accum_count=[32],
                accum_steps=[0],
                valid_batch_size=256,
                **training_config,
            ),
        )
        # Grab only explicitly set fields to save in the model's json config
        config_dict = recursive_model_fields_set(config)

        # Add inference related settings for transparent default behaviour
        inference_dict = {
            "optional_eos": optional_eos,
            # TODO: map other settings from HF decoding_config.json
            **generation_config_dict,
            **chat_template,
        }
        config_dict["inference"] = inference_dict

        with open(os.path.join(args.output, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
