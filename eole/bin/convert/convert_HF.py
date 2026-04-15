#!/usr/bin/env python
# Standard Library Imports
import configparser
import ctypes
import json
import logging
import math
import os
import re
import shutil
import struct
from dataclasses import dataclass, field, fields
from typing import Optional

# Third-Party Library Imports
import huggingface_hub
import pyonmttok
import safetensors
import torch
from huggingface_hub import hf_hub_download, utils

# from safetensors.torch import save_file
from sentencepiece import SentencePieceProcessor
from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing

# Eole Imports
from eole.bin import BaseBin, register_bin
from eole.config import recursive_model_fields_set, recursive_update_dict
from eole.constants import DefaultTokens, TORCH_DTYPES, PositionEncodingType
from eole.inputters.inputter import vocabs_to_dict
from eole.bin.convert.HF_mappings import (
    KEY_MAPS,
    MODEL_OVERRIDES,
    ACT_TABLE,
    LN_TABLE,
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
    chat_template_jinja_path: Optional[str] = None

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
            "chat_template_jinja_path": get_file_fn("chat_template.jinja", required=False),
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
        "sliding_window": config.get("sliding_window") or 0,
        "max_position_embeddings": config.get("max_position_embeddings", 8192),
        "num_experts": config.get("num_local_experts", config.get("n_routed_experts", config.get("num_experts", 0))),
        "num_shared_experts": config.get("n_shared_experts", 1 if config.get("shared_expert_intermediate_size") else 0),
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
        "embeddings": {
            "position_encoding_type": PositionEncodingType.Rotary,
            "n_positions": 0,
        },
    }
    if model_config["num_experts"] == 1 or model_config["num_experts"] is None:
        model_config["num_experts"] = 0

    # Vision encoder base fields (common to all VLMs with a vision_config)
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

    # Apply arch-specific config overrides via the unified config_from_hf callable.
    # Each arch in MODEL_OVERRIDES may provide a config_from_hf(top, text, vis) -> dict
    # function that returns a partial config dict to be deep-merged here.
    config_fn = MODEL_OVERRIDES.get(arch, {}).get("config_from_hf", None)
    if config_fn is not None:
        arch_config = config_fn(other_config, config, vision_config)
        if arch_config:
            model_config = recursive_update_dict(model_config, arch_config, {})

    # Whisper: remove rope_config (not used) and extract generation settings
    if arch == "WhisperForConditionalGeneration":
        model_config.pop("rope_config", None)
        if hf.generation_config is not None:
            audio_keys = ["suppress_tokens", "begin_suppress_tokens", "no_timestamps_token_id"]
            for key in audio_keys:
                if key in hf.generation_config:
                    model_config[key] = hf.generation_config[key]
            # Rename to avoid collision with decoder's alignment_heads (int)
            if "alignment_heads" in hf.generation_config:
                model_config["word_timestamp_heads"] = hf.generation_config["alignment_heads"]

    # patch transformer_ff
    if model_config["transformer_ff"] is None:
        model_config["transformer_ff"] = model_config["hidden_size"] * 4

    # patch sliding window
    if model_config["sliding_window"] is None:
        model_config["sliding_window"] = 0

    # Populate embeddings
    model_config["embeddings"].update(
        {
            "src_word_vec_size": model_config["hidden_size"],
            "tgt_word_vec_size": model_config["hidden_size"],
        }
    )

    # patch rotary dim (skip for models using learned positional embeddings)
    # partial_rotary_factor may live at the top level or inside rope_parameters (newer HF format)
    if "rope_config" in model_config:
        _partial_rotary = config.get(
            "partial_rotary_factor", config.get("rope_parameters", {}).get("partial_rotary_factor")
        )
        if "rotary_dim" in config.keys():
            model_config["rope_config"]["rotary_dim"] = config["rotary_dim"]
        elif _partial_rotary is not None:
            _head_dim = model_config.get("head_dim") or (model_config["hidden_size"] // model_config["heads"])
            model_config["rope_config"]["rotary_dim"] = int(_partial_rotary * _head_dim)
        elif model_config.get("head_dim", None) is not None:
            model_config["rope_config"]["rotary_dim"] = model_config["head_dim"]
        else:
            model_config["rope_config"]["rotary_dim"] = model_config["hidden_size"] // model_config["heads"]

    # Update rope scaling related settings
    if config.get("rope_scaling", None) is not None and "rope_config" in model_config:
        model_config["rope_config"].update(
            {
                "scaling_type": config["rope_scaling"].get("rope_type", config["rope_scaling"].get("type", None)),
                "scaling_factor": config["rope_scaling"].get("factor", 8.0),
                "alpha": config["rope_scaling"].get("alpha", None),
                "xdrope_section": config["rope_scaling"].get(
                    "xdrope_section", config["rope_scaling"].get("mrope_section", None)
                ),
                "low_freq_factor": config["rope_scaling"].get("low_freq_factor", 1.0),
                "high_freq_factor": config["rope_scaling"].get("high_freq_factor", 4.0),
                "original_max_position_embeddings": config["rope_scaling"].get(
                    "original_max_position_embeddings", None
                ),
            }
        )

    # Handle rope_parameters (newer HF format, e.g. Qwen3.5 VL).
    # Only apply mrope_section/mrope_interleaved when present; this naturally
    # skips Gemma4's per-layer-type format (which lacks these keys and is
    # handled by _build_gemma4_decoder_patch inside config_from_hf).
    if config.get("rope_parameters", None) is not None:
        rope_params = config["rope_parameters"]
        mrope_section = rope_params.get("mrope_section", None)
        if mrope_section is not None:
            model_config["rope_config"]["xdrope_section"] = mrope_section
        if rope_params.get("mrope_interleaved", False):
            model_config["rope_config"]["rotary_interleave"] = True

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
    # quantization_config may be at the top level (e.g. Qwen3_5MoeForConditionalGeneration)
    # or inside text_config; check top-level (other_config) first, then text_config fallback.
    quant_config = other_config.get("quantization_config", config.get("quantization_config", {}))
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
        elif quant_config.get("quant_method") in ["intel/auto-round", "autoround", "auto-round"]:
            training_config["quant_type"] = "autoround"
            training_config["w_bit"] = quant_config.get("bits", 4)
            training_config["group_size"] = quant_config.get("group_size", 128)
            training_config["autoround_packing_format"] = quant_config.get("packing_format", "auto_round:auto_gptq")
            training_config["autoround_sym"] = quant_config.get("sym", True)
            training_config["quant_layers"] = [
                "gate_up_proj",
                "down_proj",
                "up_proj",
                "linear_values",
                "linear_query",
                "linear_keys",
                "final_linear",
                # GatedDeltaNet linear-attention layers (Qwen3.5)
                "in_proj_qkv",
                "in_proj_z",
                "in_proj_b",
                "in_proj_a",
                "out_proj",
            ]
            # We search each path for a known eole module name so that
            # replace_autoround_linear can skip entire subtrees.
            extra_config = quant_config.get("extra_config", {})
            target_bits = quant_config.get("bits", 4)
            if extra_config:
                excluded_parent_modules: set[str] = set()

                # Build reverse map: HF module name → eole module name
                # e.g., "shared_expert" (HF) → "shared_experts" (eole)
                decoder_map = KEY_MAPS.get(arch, {}).get("decoder", {})
                hf_to_eole_map = {}

                for eole_key, hf_val in decoder_map.items():
                    if not isinstance(eole_key, str):
                        continue
                    hf_suffix = hf_val[0] if isinstance(hf_val, tuple) else hf_val
                    if not isinstance(hf_suffix, str):
                        continue
                    eole_parts = [p for p in eole_key.strip(".").split(".") if p and not p.isdigit()]
                    hf_parts = [p for p in hf_suffix.strip(".").split(".") if p and not p.isdigit()]
                    for hp, ep in zip(hf_parts, eole_parts):  # ← pairwise, not cartesian
                        hf_to_eole_map[hp] = ep

                # ── Fix 2: look one level deeper (sub_parts[1]) instead of sub_parts[0] ────
                for hf_path, layer_cfg in extra_config.items():
                    if ".*" in hf_path:
                        continue

                    bits_val = layer_cfg.get("bits", target_bits)
                    data_type = layer_cfg.get("data_type", "")

                    if not (bits_val < 16 and not data_type.lower().startswith(("float", "fp"))):
                        parts = hf_path.split(".")
                        if "layers" not in hf_path:
                            continue
                        try:
                            layer_idx_pos = [i for i, p in enumerate(parts) if p == "layers"][0]
                            # parts after "layers" and the layer index, e.g. ['mlp', 'shared_expert_gate']
                            sub_parts = [p for p in parts[layer_idx_pos + 2 :] if p and not p.isdigit()]
                            # ↑ +2 skips "layers" and the numeric index
                            # We want the submodule *inside* the top-level block (mlp, …),
                            # so take index 1 when available, otherwise fall back to index 0.
                            hf_module_name = sub_parts[1] if len(sub_parts) >= 2 else sub_parts[0]
                            eole_module_name = hf_to_eole_map.get(hf_module_name, hf_module_name)
                            excluded_parent_modules.add(eole_module_name)
                        except (IndexError, ValueError):
                            continue

                if excluded_parent_modules:
                    training_config["quant_exclude_modules"] = sorted(list(excluded_parent_modules))
            params = ["qweight", "qzeros", "scales"] + ["weight", "bias"]
        elif quant_config.get("quant_method") == "gptq":
            # GPTQ models use the same tensor format as AutoRound GPTQ-packing.
            # Reuse the autoround backend with auto_gptq packing convention.
            training_config["quant_type"] = "autoround"
            training_config["w_bit"] = quant_config.get("bits", 4)
            training_config["group_size"] = quant_config.get("group_size", 128)
            training_config["autoround_packing_format"] = "auto_round:auto_gptq"
            training_config["autoround_sym"] = quant_config.get("sym", True)
            training_config["quant_layers"] = [
                "gate_up_proj",
                "down_proj",
                "up_proj",
                "linear_values",
                "linear_query",
                "linear_keys",
                "final_linear",
                # GatedDeltaNet linear-attention layers (Qwen3.5)
                "in_proj_qkv",
                "in_proj_z",
                "in_proj_b",
                "in_proj_a",
                "out_proj",
            ]
            # Parse the GPTQ dynamic field.  Keys prefixed with "-:" are negation
            # patterns: regex expressions over the full HF module path that identify
            # layers which were NOT quantized.  Match them against known eole module
            # names so that replace_autoround_linear can skip those subtrees.
            dynamic = quant_config.get("dynamic", {})
            if dynamic:
                # Eole module names that can be direct children of a transformer layer.
                _CANDIDATE_EOLE_MODULES = ["self_attn", "linear_attn", "shared_experts", "experts", "mlp"]
                excluded_patterns = [k[2:] for k in dynamic if k.startswith("-:")]
                excluded_modules = []
                for pattern in excluded_patterns:
                    compiled = re.compile(pattern)
                    for mod in _CANDIDATE_EOLE_MODULES:
                        if compiled.search(mod) and mod not in excluded_modules:
                            excluded_modules.append(mod)
                if excluded_modules:
                    training_config["quant_exclude_modules"] = excluded_modules
            params = ["qweight", "qzeros", "scales"] + ["weight", "bias"]
        else:
            raise ValueError("Can convert only awq, autoround or gptq models for now")
    else:
        training_config["quant_type"] = ""
        training_config["w_bit"] = 0
        training_config["group_size"] = 0
        training_config["quant_layers"] = []
        params = ["weight", "bias"]

    model_config["share_decoder_embeddings"] = config.get("tie_word_embeddings", False)

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
    # Prefer the inline template from tokenizer_config.json; fall back to a
    # standalone chat_template.jinja file (used by some modern HF models).
    chat_template_value = hf.tokenizer_config.get("chat_template", None)
    if chat_template_value is None and hf.chat_template_jinja_path is not None:
        chat_template_value = hf.chat_template_jinja
    chat_template = {"chat_template": chat_template_value}
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

    return [sorted(s) for s in shard_checkpoints], shard_layer_ranges


class _StreamingTensorStore:
    """Write safetensors tensors to disk immediately to avoid RAM accumulation.

    Acts as a write-only dict-like store. Each tensor assigned via __setitem__
    is serialised to a temporary data file instantly, so peak RAM stays bounded
    by one layer's worth of tensors rather than the entire output shard.
    Call .save(path) to flush and assemble the final safetensors file.
    """

    # safetensors dtype string for each torch dtype we may encounter
    _DTYPE_MAP = {
        torch.float64: "F64",
        torch.float32: "F32",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.int64: "I64",
        torch.int32: "I32",
        torch.int16: "I16",
        torch.int8: "I8",
        torch.uint8: "U8",
    }

    def __init__(self, tmp_path: str):
        self._tmp_path = tmp_path
        self._fp = open(tmp_path, "wb")
        self._meta: dict = {}
        self._offset: int = 0

    def __setitem__(self, name: str, tensor):
        t = tensor.contiguous()
        dtype_str = self._DTYPE_MAP[t.dtype]
        nbytes = t.nbytes
        self._meta[name] = {
            "dtype": dtype_str,
            "shape": list(t.shape),
            "data_offsets": [self._offset, self._offset + nbytes],
        }
        self._write_tensor_bytes(t, nbytes)
        self._offset += nbytes

    _CHUNK_SIZE = 1 << 30  # 1 GiB — safely under INT32_MAX

    def _write_tensor_bytes(self, t, nbytes: int):
        ptr = t.data_ptr()
        written = 0
        while written < nbytes:
            chunk = min(self._CHUNK_SIZE, nbytes - written)
            self._fp.write(ctypes.string_at(ptr + written, chunk))
            written += chunk

    def keys(self):
        return self._meta.keys()

    def save(self, output_path: str) -> None:
        """Finalize: assemble header + raw data into a valid safetensors file."""
        self._fp.flush()
        self._fp.close()
        self._fp = None

        # safetensors requires a "__metadata__" key in the header.
        header = {"__metadata__": {}}
        header.update(self._meta)
        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        # Safetensors spec: header must be padded to a multiple of 8 bytes with spaces.
        pad = (-len(header_bytes)) % 8
        header_bytes += b" " * pad

        with open(output_path, "wb") as fout:
            fout.write(struct.pack("<Q", len(header_bytes)))
            fout.write(header_bytes)
            with open(self._tmp_path, "rb") as ftmp:
                shutil.copyfileobj(ftmp, fout, length=64 * 1024 * 1024)

        os.remove(self._tmp_path)


def _collect_all_source_keys(hf):
    """Collect all tensor keys present in the source HuggingFace checkpoint(s).

    For sharded models (with a weight-map index file) the key names are read
    directly from the already-loaded JSON index so no additional file I/O is
    required.  For single-file checkpoints the file is opened once to obtain
    its key listing.

    Args:
        hf (HuggingfaceFiles): Source model files dataclass.

    Returns:
        set: All tensor key names found across every source checkpoint file.
    """
    if hf.wmap_path is not None:
        # The weight-map already contains every tensor key as a dict key.
        return set(hf.wmap["weight_map"].keys())
    else:
        ckpt = hf.get_load_ckpt(*os.path.split(hf.model_path))
        if isinstance(ckpt, dict):
            return set(ckpt.keys())
        else:
            with safetensors.safe_open(ckpt, framework="pt", device="cpu") as f:
                return set(f.keys())


def check_conversion_completeness(all_src_keys, consumed_src_keys):
    """Check that every source checkpoint tensor was accounted for during conversion.

    Tensors that exist in the source checkpoints but were never read and stored
    in any output shard are printed as warnings.

    Args:
        all_src_keys (set): All tensor keys from the source checkpoint(s).
        consumed_src_keys (set): Source keys that contributed to at least one
            output tensor.

    Returns:
        set: Source keys that were not converted (empty when everything matched).
    """
    unconverted = all_src_keys - consumed_src_keys
    if unconverted:
        print("\nWARNING: %d source tensor(s) were not converted:" % len(unconverted))
        for key in sorted(unconverted):
            print("  - %s" % key)
    else:
        print("\nAll source tensors were successfully converted.")
    return unconverted


def check_conversion_equality(hf, conversion_details, target_dtype):
    """Verify that every converted tensor in the output matches its source.

    For each entry in *conversion_details* the corresponding source tensor is
    reloaded from disk, the same transformation that was applied during
    conversion is re-applied, and the result is compared element-wise with the
    tensor that was written to the output shard.  Any mismatch is reported.

    Args:
        hf (HuggingfaceFiles): Source model files dataclass (used to reload
            source tensors).
        conversion_details (list[dict]): One dict per written output tensor,
            each containing:
            - ``shard_path``  – path of the output safetensors shard file
            - ``eole_key``    – tensor key inside that shard
            - ``srckey``      – full tensor key in the source checkpoint
            - ``srcmap``      – optional eval-string transformation (or None)
            - ``context``     – variable bindings for the eval string
            - ``special``     – optional post-transform tag (e.g. "unsqueeze(0)")
        target_dtype: Target torch dtype applied during conversion, or None.

    Returns:
        list[tuple]: ``(eole_key, reason)`` pairs for every mismatch found.
            Empty when all tensors matched.
    """
    mismatches = []
    print("\nVerifying converted tensor equality (%d tensors)..." % len(conversion_details))
    for detail in conversion_details:
        shard_path = detail["shard_path"]
        eole_key = detail["eole_key"]
        srckey = detail["srckey"]
        srcmap = detail["srcmap"]
        context = detail["context"]
        special = detail["special"]

        # Reload the source tensor from the appropriate checkpoint file.
        if hf.wmap_path is not None:
            ckpt_name = hf.wmap["weight_map"].get(srckey)
            if ckpt_name is None:
                mismatches.append((eole_key, "source key '%s' not found in weight map" % srckey))
                continue
            ckpt = hf.get_load_ckpt(hf.base_dir, ckpt_name)
        else:
            ckpt = hf.get_load_ckpt(*os.path.split(hf.model_path))

        src = get_weight(ckpt, srckey)
        if src is None:
            mismatches.append((eole_key, "source tensor '%s' could not be loaded" % srckey))
            continue

        # Re-apply the same transformation used during conversion.
        if srcmap is not None:
            src = eval("w" + srcmap, {"w": src, **context}).contiguous()

        # Re-apply any special post-transform.
        if special == "unsqueeze(0)":
            src = src.unsqueeze(0)

        # Cast to the same dtype that was applied during conversion.
        if target_dtype is not None:
            src = src.to(target_dtype)

        # Load the corresponding tensor from the saved output shard.
        with safetensors.safe_open(shard_path, framework="pt", device="cpu") as f:
            if eole_key not in f.keys():
                mismatches.append((eole_key, "key not found in output shard '%s'" % shard_path))
                continue
            tgt = f.get_tensor(eole_key)

        # Element-wise comparison.
        if not torch.equal(src, tgt):
            mismatches.append((eole_key, "values differ between source (after transform) and output"))

    if mismatches:
        print("\nERROR: %d converted tensor(s) do not match their source:" % len(mismatches))
        for key, reason in mismatches:
            print("  - %s: %s" % (key, reason))
    else:
        print("All %d converted tensors match their source." % len(conversion_details))
    return mismatches


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
    target_dtype = TORCH_DTYPES[args.dtype] if args.dtype is not None else None

    # Collect every tensor key present in the source checkpoint(s) upfront.
    all_src_keys = _collect_all_source_keys(hf)
    consumed_src_keys = set()
    conversion_details = []

    for shard in range(args.nshards):

        print("starting output shard: %d/%d" % (shard + 1, args.nshards))
        output_path = os.path.join(args.output, "model.{:02d}.safetensors".format(shard))
        eole_safetensor = _StreamingTensorStore(output_path + ".part")

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
                    consumed_src_keys.add(srckey)
                    context = {
                        "hidden_size": model_config["hidden_size"],
                        "transformer_ff": model_config["transformer_ff"],
                    }
                    if srcmap is not None:
                        w = eval(
                            "w" + srcmap,
                            {"w": w, **context},
                        ).contiguous()
                    if target_dtype is not None:
                        w = w.to(target_dtype)
                    special = None
                    if target == "encoder.class_embedding.weight":
                        eole_safetensor[target] = w.unsqueeze(0)
                        special = "unsqueeze(0)"
                    else:
                        eole_safetensor[target] = w
                    conversion_details.append(
                        {
                            "shard_path": output_path,
                            "eole_key": target,
                            "srckey": srckey,
                            "srcmap": srcmap,
                            "context": context,
                            "special": special,
                        }
                    )
            return eole_safetensor

        if shard == 0:
            eole_safetensor = build_first_shard(hf, eole_safetensor)

            # Gemma4MultimodalEmbedder uses Gemma4RMSNorm(with_scale=False): no learnable
            # weight.  EOLE's Gemma4MultiModalProjector uses RMSNormNoScale (no parameters),
            # so there is no adapter.norm.weight to inject.  Previously we created a zeros
            # tensor which was then reported as "Extra key in checkpoint" at load time.

        # TODO: could we reverse the mapping and loop on params instead? (would reduce conditions)
        for ckpt in shard_checkpoints[shard]:
            print("Loading %s" % ckpt)
            checkpoint = hf.checkpoint(ckpt)
            # Pre-compute the set of keys available in this checkpoint once,
            # so we can skip lookups for missing keys without repeatedly opening the file.
            if isinstance(checkpoint, dict):
                ckpt_keys = set(checkpoint.keys())
            else:
                with safetensors.safe_open(checkpoint, framework="pt", device="cpu") as f:
                    ckpt_keys = set(f.keys())
            for i in shard_layer_ranges[shard]:
                # Cache raw tensors read from this checkpoint+layer to avoid
                # re-reading the same stacked tensor (e.g. mlp.experts.gate_up_proj)
                # hundreds of times when splitting MoE expert weights.
                _layer_tensor_cache = {}
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
                            full_srckey = hf_prefix + str(i) + srckey
                            if full_srckey not in _layer_tensor_cache:
                                if full_srckey not in ckpt_keys:
                                    _layer_tensor_cache[full_srckey] = None
                                else:
                                    _layer_tensor_cache[full_srckey] = get_weight(
                                        checkpoint,
                                        full_srckey,
                                    )
                            w = _layer_tensor_cache[full_srckey]

                            if w is not None:
                                consumed_src_keys.add(full_srckey)
                                if srcmap is not None:
                                    hidden_size = (
                                        model_config["hidden_size"]
                                        if section.startswith("decoder")
                                        else model_config["encoder"]["hidden_size"]
                                    )
                                    context = {
                                        "hidden_size": hidden_size,
                                        "transformer_ff": model_config["transformer_ff"],
                                        "moe_transformer_ff": model_config.get("decoder", {}).get(
                                            "moe_transformer_ff",
                                            model_config.get("moe_transformer_ff", 0),
                                        ),
                                    }
                                    w = eval(
                                        "w" + srcmap,
                                        {"w": w, **context},
                                    ).contiguous()
                                else:
                                    context = {}
                                if target_dtype is not None:
                                    w = w.to(target_dtype)
                                target1 = target
                                if target.endswith("."):
                                    target1 = target + param
                                eole_key = eole_prefix + str(i) + target1
                                if eole_key not in eole_safetensor.keys():
                                    eole_safetensor[eole_key] = w
                                    conversion_details.append(
                                        {
                                            "shard_path": output_path,
                                            "eole_key": eole_key,
                                            "srckey": full_srckey,
                                            "srcmap": srcmap,
                                            "context": context,
                                            "special": None,
                                        }
                                    )
                                    # For Gemma4 full_attention layers with attention_k_eq_v=True,
                                    # v_proj is None in HF (value reuses key): tie linear_values to linear_keys.
                                    if (
                                        ".self_attn.linear_keys." in eole_key
                                        and section == "decoder"
                                        and hf.arch in ("Gemma4ForCausalLM", "Gemma4ForConditionalGeneration")
                                        and hf.config.get("text_config", hf.config).get("attention_k_eq_v", False)
                                    ):
                                        layer_types = model_config.get("decoder", {}).get("layer_types", [])
                                        if i < len(layer_types) and layer_types[i] == "full_attention":
                                            v_key = eole_key.replace(
                                                ".self_attn.linear_keys.", ".self_attn.linear_values."
                                            )
                                            if v_key not in eole_safetensor.keys():
                                                eole_safetensor[v_key] = w
                                                conversion_details.append(
                                                    {
                                                        "shard_path": output_path,
                                                        "eole_key": v_key,
                                                        "srckey": full_srckey,
                                                        "srcmap": srcmap,
                                                        "context": context,
                                                        "special": "gemma4_kv_eq",
                                                    }
                                                )
                _layer_tensor_cache.clear()

        print("Saving output model shard: %d" % shard)
        eole_safetensor.save(output_path)

    return all_src_keys, consumed_src_keys, conversion_details


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
    pre_tokenizer = hf.tokenizer.get("pre_tokenizer", None) or {}
    pretokenizers = pre_tokenizer.get("pretokenizers", None)
    if pretokenizers is None:
        pretokenizers = [pre_tokenizer]
    gpt2_pretok = False
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
            vocabfile.write(tok.replace("\n", DefaultTokens.SEP) + "\n")


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
            default=None,
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
        parser.add_argument(
            "--check-tensors",
            action="store_true",
            default=False,
            help=(
                "After conversion, verify completeness (every source tensor "
                "was consumed) and equality (every output tensor matches its "
                "source after the same transform)."
            ),
        )
        parser.add_argument(
            "--config-only",
            action="store_true",
            default=False,
            help=("Skip weight conversion entirely and only write config.json " "to the output directory."),
        )

    @classmethod
    def run(cls, args):
        from eole.models.hf_loader import build_train_config

        hf = HuggingfaceFiles.fetch(args)

        # Deduce dtype from args or config, or default to fp16
        compute_dtype = args.dtype or hf.config.get("torch_dtype") or "fp16"

        # Build eole config and vocab using the shared helper
        train_config, vocabs, params, model_config, inference_dict = build_train_config(
            hf,
            args.output,
            tokenizer_pref=args.tokenizer,
            dtype=compute_dtype,
            use_env_path=True,
        )

        # Strip tokenizer post_processor for Whisper models.
        # This must happen after build_train_config() (which writes the BPE/tokenizer
        # artifacts) so we can save a cleaned version over the original.
        if hf.arch == "WhisperForConditionalGeneration" and hf.tokenizer_json:
            tokenizer = Tokenizer.from_file(hf.tokenizer_json)
            tokenizer.post_processor = TemplateProcessing(
                single="$A",
                pair="$A $B",
                special_tokens=[],
            )
            clean_tokenizer_path = os.path.join(args.output, "tokenizer.json")
            tokenizer.save(clean_tokenizer_path)
            print("Stripped tokenizer post_processor for Whisper model")

        # Save vocab files to output model directory
        save_vocab(vocabs, vocabs["src"], args.output)

        # Build shards (skipped when --config-only is set)
        if not args.config_only:
            all_src_keys, consumed_src_keys, conversion_details = build_shards(model_config, hf, args, params)

            # Post-conversion validation (opt-in via --check-tensors)
            if args.check_tensors:
                target_dtype = TORCH_DTYPES[args.dtype] if args.dtype is not None else None
                check_conversion_completeness(all_src_keys, consumed_src_keys)
                check_conversion_equality(hf, conversion_details, target_dtype)

        # Grab only explicitly set fields to save in the model's json config
        config_dict = recursive_model_fields_set(train_config)

        # Whisper-specific: update the tokenizer path in the serialised config to
        # point at the cleaned tokenizer.json (post_processor stripped above).
        if hf.arch == "WhisperForConditionalGeneration" and hf.tokenizer_json:
            (
                config_dict.setdefault("transforms_configs", {})
                .setdefault("huggingface_tokenize", {})
                .__setitem__("path", clean_tokenizer_path)
            )

        # Add inference related settings for transparent default behaviour
        config_dict["inference"] = inference_dict

        with open(os.path.join(args.output, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        # Copy chat_template.jinja to the output directory so the converted
        # model is self-contained (needed when the source is a local directory,
        # since hub downloads already land in args.output).
        if hf.chat_template_jinja_path is not None:
            dest = os.path.join(args.output, "chat_template.jinja")
            if os.path.normpath(hf.chat_template_jinja_path) != os.path.normpath(dest):
                shutil.copy2(hf.chat_template_jinja_path, dest)
