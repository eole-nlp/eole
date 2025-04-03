#!/usr/bin/env python
# Standard Library Imports
import configparser
import json
import logging
import math
import os
import re
from collections import defaultdict
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
from eole.config.models import (
    TransformerEncoderModelConfig,
    TransformerModelConfig,
    TransformerLMModelConfig,
    VisionTransformerLMModelConfig,
)
from eole.config.run import TrainConfig
from eole.config.training import TrainingConfig
from eole.constants import DefaultTokens, TORCH_DTYPES, PositionEncodingType
from eole.inputters.inputter import vocabs_to_dict


# Default tensor key mappings, based on Llama
BASE_KEY_MAP = {
    "decoder_layer_prefix": "model.layers.",
    "tgt_emb.embeddings.weight": "model.embed_tokens.weight",
    "decoder.layer_norm.weight": "model.norm.weight",
    "generator.weight": "lm_head.weight",
    ".self_attn.linear_query.": ".self_attn.q_proj.",
    ".self_attn.linear_keys.": ".self_attn.k_proj.",
    ".self_attn.linear_values.": ".self_attn.v_proj.",
    ".self_attn.final_linear.": ".self_attn.o_proj.",
    ".mlp.gate_up_proj.": ".mlp.gate_proj.",
    ".mlp.down_proj.": ".mlp.down_proj.",
    ".mlp.up_proj.": ".mlp.up_proj.",
    ".input_layernorm.weight": ".input_layernorm.weight",
    ".post_attention_layernorm.weight": ".post_attention_layernorm.weight",
}


MODEL_OVERRIDES = {
    "LlamaForCausalLM": {},  # default
    "MistralForCausalLM": {},
    "Qwen2ForCausalLM": {},
    "Gemma2ForCausalLM": {
        ".pre_feedforward_layernorm.weight": ".pre_feedforward_layernorm.weight",
        ".post_feedforward_layernorm.weight": ".post_feedforward_layernorm.weight",
    },
    "MixtralForCausalLM": {
        ".mlp.gate.weight": ".block_sparse_moe.gate.weight",
        **{
            f".mlp.experts.{i}.{attr}": f".block_sparse_moe.experts.{i}.w{j}."
            for i in range(8)
            for j, attr in enumerate(["gate_up_proj.", "down_proj.", "up_proj."])
        },
        **{f".mlp.experts.{i}.layer_norm.weight": ".post_attention_layernorm.weight" for i in range(8)},
    },
    "PhiForCausalLM": {
        "decoder.layer_norm.weight": "model.final_layernorm.weight",
        "decoder.layer_norm.bias": "model.final_layernorm.bias",
        "generator.bias": "lm_head.bias",
        ".self_attn.final_linear.": ".self_attn.dense.",
        ".mlp.gate_up_proj.": ".mlp.fc1.",
        ".mlp.down_proj.": ".mlp.fc2.",
        ".input_layernorm.bias": (".input_layernorm.bias", ""),
    },
    "Phi3ForCausalLM": {
        ".self_attn.linear_query.": (".self_attn.qkv_proj.", "[:hidden_size, :]"),
        ".self_attn.linear_keys.": (
            ".self_attn.qkv_proj.",
            "[hidden_size:2*hidden_size, :]",
        ),
        ".self_attn.linear_values.": (".self_attn.qkv_proj.", "[-hidden_size:, :]"),
        ".mlp.gate_up_proj.": (".mlp.gate_up_proj.", "[:transformer_ff, :]"),
        ".mlp.up_proj.": (".mlp.gate_up_proj.", "[transformer_ff:, :]"),
    },
    "GPT2LMHeadModel": {
        "decoder_layer_prefix": "h.",
        "tgt_emb.pe.weight": "wpe.weight",
        ".self_attn.linear_query.": (".attn.c_attn.", ".t()[:hidden_size, ...]"),
        ".self_attn.linear_keys.": (
            ".attn.c_attn.",
            ".t()[hidden_size:2*hidden_size, ...]",
        ),
        ".self_attn.linear_values.": (".attn.c_attn.", ".t()[-hidden_size:, ...]"),
        ".self_attn.final_linear.": (".attn.c_proj.", ".t()"),
        ".mlp.gate_up_proj.": (".mlp.c_fc.", ".t()"),
        ".mlp.down_proj.": (".mlp.c_proj.", ".t()"),
        ".input_layernorm.weight": ".ln_1.weight",
        ".input_layernorm.bias": ".ln_1.bias",
        ".post_attention_layernorm.weight": ".ln_2.weight",
        ".post_attention_layernorm.bias": ".ln_2.bias",
        "decoder.layer_norm.weight": "ln_f.weight",
        "decoder.layer_norm.bias": "ln_f.bias",
    },
    "XLMRobertaXLForMaskedLM": {
        "encoder_layer_prefix": "roberta.encoder.layer.",
        "src_emb.embeddings.weight": "roberta.embeddings.word_embeddings.weight",
        "src_emb.pe.weight": "roberta.embeddings.position_embeddings.weight",
        ".self_attn.linear_query.": ".attention.self.query.",
        ".self_attn.linear_keys.": ".attention.self.key.",
        ".self_attn.linear_values.": ".attention.self.value.",
        ".self_attn.final_linear.": ".attention.output.dense.",
        ".mlp.gate_up_proj.": ".intermediate.dense.",
        ".mlp.down_proj.": ".output.dense.",
        ".input_layernorm.weight": ".attention.self_attn_layer_norm.weight",
        ".input_layernorm.bias": ".attention.self_attn_layer_norm.bias",
        ".post_attention_layernorm.weight": ".LayerNorm.weight",
        ".post_attention_layernorm.bias": ".LayerNorm.bias",
        "encoder.layer_norm.weight": "roberta.encoder.LayerNorm.weight",
        "encoder.layer_norm.bias": "roberta.encoder.LayerNorm.bias",
    },
    "LlavaForConditionalGeneration": {
        "decoder_layer_prefix": "language_model.model.layers.",
        "tgt_emb.embeddings.weight": "language_model.model.embed_tokens.weight",
        "decoder.layer_norm.weight": "language_model.model.norm.weight",
        "generator.weight": "language_model.lm_head.weight",
        "encoder.patch_conv.weight": "vision_tower.patch_conv.weight",
        "encoder.ln_pre.weight": "vision_tower.ln_pre.weight",
        # vision_tower
        "encoder_layer_prefix": "vision_tower.transformer.layers.",
        "encoder": {
            "layers": 24,
            ".self_attn.linear_query.": ".attention.q_proj.",
            ".self_attn.linear_keys.": ".attention.k_proj.",
            ".self_attn.linear_values.": ".attention.v_proj.",
            ".self_attn.final_linear.": ".attention.o_proj.",
            ".mlp.gate_up_proj.": ".feed_forward.gate_proj.",
            ".mlp.down_proj.": ".feed_forward.down_proj.",
            ".mlp.up_proj.": ".feed_forward.up_proj.",
            ".input_layernorm.weight": ".attention_norm.weight",  # not sure about this one
            ".post_attention_layernorm.weight": ".ffn_norm.weight",
        },
        # vision_adapter
        "adapter.w_in.weight": "multi_modal_projector.linear_1.weight",
        "adapter.w_in.bias": "multi_modal_projector.linear_1.bias",
        "adapter.w_out.weight": "multi_modal_projector.linear_2.weight",
        "adapter.w_out.bias": "multi_modal_projector.linear_2.bias",
    },
    "Mistral3ForConditionalGeneration": {
        "decoder_layer_prefix": "language_model.model.layers.",
        "tgt_emb.embeddings.weight": "language_model.model.embed_tokens.weight",
        "decoder.layer_norm.weight": "language_model.model.norm.weight",
        "generator.weight": "language_model.lm_head.weight",
        "encoder.patch_conv.weight": "vision_tower.patch_conv.weight",
        "encoder.ln_pre.weight": "vision_tower.ln_pre.weight",
        # vision_tower
        "encoder_layer_prefix": "vision_tower.transformer.layers.",
        "encoder": {
            "layers": 24,
            ".self_attn.linear_query.": ".attention.q_proj.",
            ".self_attn.linear_keys.": ".attention.k_proj.",
            ".self_attn.linear_values.": ".attention.v_proj.",
            ".self_attn.final_linear.": ".attention.o_proj.",
            ".mlp.gate_up_proj.": ".feed_forward.gate_proj.",
            ".mlp.down_proj.": ".feed_forward.down_proj.",
            ".mlp.up_proj.": ".feed_forward.up_proj.",
            ".input_layernorm.weight": ".attention_norm.weight",  # not sure about this one
            ".post_attention_layernorm.weight": ".ffn_norm.weight",
        },
        # vision_adapter
        "adapter.w_in.weight": "multi_modal_projector.linear_1.weight",
        "adapter.w_out.weight": "multi_modal_projector.linear_2.weight",
        "adapter.layernorm.weight": "multi_modal_projector.norm.weight",
        "adapter.patch_merger.merging_layer.weight": "multi_modal_projector.patch_merger.merging_layer.weight",
    },
    "M2M100ForConditionalGeneration": {
        "encoder_layer_prefix": "model.encoder.layers.",
        "decoder_layer_prefix": "model.decoder.layers.",
        "src_emb.embeddings.weight": "model.encoder.embed_tokens.weight",
        "tgt_emb.embeddings.weight": "model.decoder.embed_tokens.weight",
        "decoder.layer_norm.weight": "model.decoder.layer_norm.weight",
        "decoder.layer_norm.bias": "model.decoder.layer_norm.bias",
        "encoder.layer_norm.weight": "model.encoder.layer_norm.weight",
        "encoder.layer_norm.bias": "model.encoder.layer_norm.bias",
        ".self_attn.linear_query.": ".self_attn.q_proj.",
        ".self_attn.linear_keys.": ".self_attn.k_proj.",
        ".self_attn.linear_values.": ".self_attn.v_proj.",
        ".self_attn.final_linear.": ".self_attn.out_proj.",
        ".precontext_layernorm.weight": ".encoder_attn_layer_norm.weight",
        ".precontext_layernorm.bias": ".encoder_attn_layer_norm.bias",
        ".context_attn.linear_query.": ".encoder_attn.q_proj.",
        ".context_attn.linear_keys.": ".encoder_attn.k_proj.",
        ".context_attn.linear_values.": ".encoder_attn.v_proj.",
        ".context_attn.final_linear.": ".encoder_attn.out_proj.",
        ".mlp.gate_up_proj.": ".fc1.",
        ".mlp.down_proj.": ".fc2.",
        ".input_layernorm.weight": ".self_attn_layer_norm.weight",
        ".input_layernorm.bias": ".self_attn_layer_norm.bias",
        ".post_attention_layernorm.weight": ".final_layer_norm.weight",
        ".post_attention_layernorm.bias": ".final_layer_norm.bias",
        "encoder": {
            ".self_attn.linear_query.": ".self_attn.q_proj.",
            ".self_attn.linear_keys.": ".self_attn.k_proj.",
            ".self_attn.linear_values.": ".self_attn.v_proj.",
            ".self_attn.final_linear.": ".self_attn.out_proj.",
            ".mlp.gate_up_proj.": ".fc1.",
            ".mlp.down_proj.": ".fc2.",
            ".input_layernorm.weight": ".self_attn_layer_norm.weight",
            ".input_layernorm.bias": ".self_attn_layer_norm.bias",
            ".post_attention_layernorm.weight": ".final_layer_norm.weight",
            ".post_attention_layernorm.bias": ".final_layer_norm.bias",
        },
    },
}

# Combine base mappings with overrides
KEY_MAPS = {model: {**BASE_KEY_MAP, **overrides} for model, overrides in MODEL_OVERRIDES.items()}

# Layer norm type
LN_TABLE = defaultdict(
    lambda: "rms",
    {
        "PhiForCausalLM": "standard",
        "GPT2LMHeadModel": "standard",
        "XLMRobertaXLForMaskedLM": "standard",
        "Gemma2ForCausalLM": "gemma-rms",
        "M2M100ForConditionalGeneration": "standard",
    },
)

# Activation type (gated-silu also enables the ffn gate)
ACT_TABLE = defaultdict(
    lambda: "gated-silu",
    {
        "PhiForCausalLM": "gelu",
        "GPT2LMHeadModel": "gelu",
        "XLMRobertaXLForMaskedLM": "gelu",
        "Gemma2ForCausalLM": "gated-gelu",
        "M2M100ForConditionalGeneration": "relu",
    },
)
VISION_ACT_TABLE = defaultdict(
    lambda: "gated-silu",
    {
        "Mistral3ForConditionalGeneration": "gated-gelu",
    },
)

# Eole config class
ARCH_TABLE = defaultdict(
    lambda: TransformerLMModelConfig,
    {
        "XLMRobertaXLForMaskedLM": TransformerEncoderModelConfig,
        "LlavaForConditionalGeneration": VisionTransformerLMModelConfig,
        "Mistral3ForConditionalGeneration": VisionTransformerLMModelConfig,
        "M2M100ForConditionalGeneration": TransformerModelConfig,
    },
)

# Default tokenization transform
TOK_TABLE = defaultdict(lambda: "huggingface_tokenize")


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
        return config["vocab_size"]

    @property
    def encoder_layer_prefix(self):
        return KEY_MAPS[self.arch].get("encoder_layer_prefix", None)

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

    vision_config = config.get("vision_config", None)
    other_config = config  # save what is not text/vision for later use
    config = config.get("text_config", config)

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
        "parallel_residual": config.get("parallel_attn", False),
        "norm_eps": config.get(
            "rms_norm_eps",
            config.get("layer_norm_epsilon", config.get("layer_norm_eps", 1e-5)),
        ),
        "sliding_window": config.get("sliding_window", 0) or 4096,
        "num_experts": config.get("num_local_experts", 0),
        "num_experts_per_tok": config.get("num_experts_per_tok", 0),
        "add_qkvbias": False,
        "add_final_linear_bias": False,
        "add_ffnbias": False,
        "shared_layer_norm": False,
        "left_pad": True,
        "generator_bias": False,
        "adapter_bias": False,
        "rope_config": {
            "rotary_interleave": False,
        },
        "embeddings": {},  # Populated later
    }

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

    # patch rotary theta
    if "rope_theta" in config.keys():
        model_config["rope_config"]["rotary_theta"] = config["rope_theta"]

    # Validate required fields
    required_fields = {
        "layers": "Can't find the number of layers in the config.json file",
        "hidden_size": "Can't find the model hidden size in the config.json file",
        "heads": "Can't find the number of heads in the config.json file",
    }

    for key, error_msg in required_fields.items():
        if model_config[key] is None:
            raise ValueError(error_msg)

    # Update rope scaling related settings
    if config.get("rope_scaling", None) is not None:
        model_config["rope_config"].update(
            {
                "scaling_type": config["rope_scaling"].get("rope_type"),
                "scaling_factor": config["rope_scaling"].get("factor", 8.0),
                "low_freq_factor": config["rope_scaling"].get("low_freq_factor", 1.0),
                "high_freq_factor": config["rope_scaling"].get("high_freq_factor", 4.0),
                "original_max_position_embeddings": config["rope_scaling"].get(
                    "original_max_position_embeddings", 8192
                ),
            }
        )

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
        params = ["qweight", "qzeros", "scales"]
    else:
        training_config["quant_type"] = ""
        training_config["w_bit"] = 0
        training_config["group_size"] = 0
        training_config["quant_layers"] = []
        params = ["weight", "bias"]

    model_config["share_decoder_embeddings"] = config.get("tie_word_embeddings", False)

    # Default position encoding configuration
    model_config["embeddings"].update(
        {
            "position_encoding_type": PositionEncodingType.Rotary,
            "n_positions": 0,
        }
    )

    # Define architecture-specific configurations
    arch_configs = {
        "PhiForCausalLM": {
            "parallel_residual": True,
            "shared_layer_norm": True,
            "add_qkvbias": True,
            "add_final_linear_bias": True,
            "add_ffnbias": True,
        },
        "GPT2LMHeadModel": {
            "parallel_residual": False,
            "shared_layer_norm": True,
            "add_qkvbias": True,
            "add_final_linear_bias": True,
            "add_ffnbias": True,
            "embeddings": {
                "position_encoding_type": "Learned",
                "n_positions": 1024,
            },
            "left_pad": False,
        },
        "XLMRobertaXLForMaskedLM": {
            "add_qkvbias": True,
            "add_final_linear_bias": True,
            "add_ffnbias": True,
            "embeddings": {
                "position_encoding_type": "Learned",
                "n_positions": 514,
                "position_shift": 2,
            },
            "left_pad": False,
        },
        "Qwen2ForCausalLM": {
            "add_qkvbias": True,
            "add_final_linear_bias": False,
        },
        "Gemma2ForCausalLM": {
            "share_decoder_embeddings": True,
            "ffn_layernorm": True,
            "embeddings": {
                "normalize": True,
            },
        },
        "M2M100ForConditionalGeneration": {
            "parallel_residual": False,
            "add_qkvbias": True,
            "add_final_linear_bias": True,
            "add_ffnbias": True,
            "embeddings": {
                "position_encoding_type": "SinusoidalConcat",
                "n_positions": 1024,
            },
            "left_pad": False,
            "share_decoder_embeddings": True,
        },
    }

    # Vision encoder
    if vision_config is not None:
        # TODO: extend to other Llava models (with CLIP vision encoder)
        model_config["encoder"] = {
            "mlp_activation_fn": VISION_ACT_TABLE[arch],
            "layer_norm": model_config["layer_norm"],
            "norm_eps": model_config["norm_eps"],
            "hidden_size": vision_config.get("hidden_size", vision_config["image_size"]),
            "transformer_ff": vision_config.get(
                "intermediate_size", vision_config["image_size"] * 4
            ),  # hard-coded for mistral-community/pixtral-12b
            "num_channels": 3,
            "image_size": vision_config["image_size"],
            "patch_size": vision_config["patch_size"],
            "rope_config": {
                "rotary_theta": vision_config["rope_theta"],
                "rotary_interleave": False,
            },
            "layers": vision_config.get("num_hidden_layers", 24),  # hard-coded for mistral-community/pixtral-12b
            "heads": vision_config.get("num_attention_heads", vision_config["image_size"] / vision_config["head_dim"]),
            "heads_kv": vision_config.get(
                "num_attention_heads", vision_config["image_size"] / vision_config["head_dim"]
            ),
            "head_dim": vision_config["head_dim"],
            "image_token_id": 10,
        }
        model_config["multimodal_projector_bias"] = other_config.get("multimodal_projector_bias", False)
        model_config["projector_activation_fn"] = other_config.get("projector_hidden_act", "gelu")
        model_config["spatial_merge_size"] = other_config.get("spatial_merge_size", None)

    # Update model_config based on architecture
    if arch in arch_configs:
        for key, value in arch_configs[arch].items():
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
    add_bos_token = hf.tokenizer_config.get("add_bos_token", hf.tokenizer_config.get("bos_token", False) is not None)
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
    layers_per_shard = math.ceil(model_config["layers"] / nshards)
    shard_layer_ranges = [
        range(
            layers_per_shard * shard,
            min(layers_per_shard * (shard + 1), model_config["layers"]),
        )
        for shard in range(nshards)
    ]
    if hf.wmap_path is None:
        return [[hf.model_path]] * nshards, shard_layer_ranges

    weightmap = hf.wmap["weight_map"]

    # Initialize a list of checkpoint lists, one for each shard
    ckpt_lists = [set() for _ in range(nshards)]

    # Check if a layer key belongs to the current shard
    def is_layer_in_range(key, prefix, layer_range):
        return prefix and key.startswith(prefix) and int(key.split(".")[len(prefix.split(".")) - 1]) in layer_range

    # Loop over the weightmap and distribute checkpoints to the appropriate shards
    for key, ckpt in weightmap.items():
        for shard, layer_range in enumerate(shard_layer_ranges):
            if is_layer_in_range(key, hf.decoder_layer_prefix, layer_range) or is_layer_in_range(
                key, hf.encoder_layer_prefix, layer_range
            ):
                ckpt_lists[shard].add(ckpt)

    return ckpt_lists, shard_layer_ranges


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
    ckpt_lists, shard_layer_ranges = get_shards_map(model_config, hf, args.nshards)

    for shard in range(args.nshards):

        print("starting output shard: %d/%d" % (shard + 1, args.nshards))
        eole_safetensor = {}
        first_shard_targets = [
            "tgt_emb.embeddings.weight",
            "tgt_emb.pe.weight",
            "decoder.layer_norm.weight",
            "decoder.layer_norm.bias",
            "src_emb.embeddings.weight",
            "src_emb.pe.weight",
            "encoder.layer_norm.weight",
            "encoder.layer_norm.bias",
            "generator.weight",
            "generator.bias",
            "encoder.patch_conv.weight",
            "encoder.ln_pre.weight",
            "adapter.w_in.weight",
            "adapter.w_in.bias",
            "adapter.w_out.weight",
            "adapter.w_out.bias",
            "adapter.layernorm.weight",
            "adapter.patch_merger.merging_layer.weight",
        ]

        def build_first_shard(hf, eole_safetensor):
            if model_config["share_decoder_embeddings"]:
                first_shard_targets.remove("generator.weight")
            for target in first_shard_targets:
                if target in KEY_MAPS[hf.arch].keys():
                    source = KEY_MAPS[hf.arch][target]
                    if hf.wmap_path:
                        checkpoint = hf.get_load_ckpt(
                            hf.base_dir,
                            hf.wmap["weight_map"][source],
                        )
                    else:
                        checkpoint = hf.get_load_ckpt(*os.path.split(hf.model_path))
                    w = get_weight(checkpoint, source)
                    if w is not None:
                        eole_safetensor[target] = w

                    if target == "generator.bias":
                        model_config["generator_bias"] = True
                    if target == "adapter.w_in.bias":
                        model_config["adapter_bias"] = True
            return eole_safetensor

        if shard == 0:
            eole_safetensor = build_first_shard(hf, eole_safetensor)

        for ckpt in ckpt_lists[shard]:
            print("Loading %s" % ckpt)
            checkpoint = hf.checkpoint(ckpt)
            for i in shard_layer_ranges[shard]:
                prefix_mapping = (
                    (hf.encoder_layer_prefix, "encoder.transformer_layers."),
                    (hf.decoder_layer_prefix, "decoder.transformer_layers."),
                )
                for hf_prefix, eole_prefix in prefix_mapping:
                    if hf_prefix is None:
                        continue
                    for param in params:
                        # TODO: factorize this better
                        for key_map in [KEY_MAPS[hf.arch], KEY_MAPS[hf.arch].get("encoder", {})]:
                            for target, source in key_map.items():
                                # TODO: this should be cleaned up when rationalizing encoder/decoder mappings
                                if not (isinstance(source, str) or isinstance(source, tuple)):
                                    continue
                                if target in first_shard_targets:
                                    continue
                                srckey, srcmap = source if isinstance(source, tuple) else (source, None)
                                w = get_weight(
                                    checkpoint,
                                    hf_prefix + str(i) + srckey + param,
                                )

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
                                    eole_safetensor[eole_prefix + str(i) + target + param] = w

                    if model_config["shared_layer_norm"]:
                        idx = 0
                    else:
                        idx = 1
                    # Not sure if we can handle these in the loop above
                    for p in ["weight", "bias"]:
                        for module in [
                            "input_layernorm",
                            "layer_norm_res",
                            "precontext_layernorm",
                            "post_attention_layernorm",
                            "pre_feedforward_layernorm",
                            "post_feedforward_layernorm",
                            "mlp.gate",
                        ]:
                            if hf_prefix == hf.encoder_layer_prefix:
                                source_map = KEY_MAPS[hf.arch]["encoder"]
                            else:
                                source_map = KEY_MAPS[hf.arch]
                            module_p = f".{module}.{p}"
                            if module_p in source_map.keys():
                                if isinstance(source_map[module_p], tuple):
                                    w = get_weight(
                                        checkpoint,
                                        hf_prefix + str(i) + source_map[module_p][idx],
                                    )
                                else:
                                    w = get_weight(
                                        checkpoint,
                                        hf_prefix + str(i) + source_map[module_p],
                                    )
                                if w is not None:
                                    eole_safetensor[eole_prefix + str(i) + module_p] = w

                        for j in range(model_config["num_experts"]):
                            if f".mlp.experts.{j}.layer_norm." + p in source_map.keys():
                                w = get_weight(
                                    checkpoint,
                                    hf_prefix + str(i) + source_map[f".mlp.experts.{j}.layer_norm." + p],
                                )
                                if w is not None:
                                    eole_safetensor[eole_prefix + str(i) + f".mlp.experts.{j}.layer_norm." + p] = w

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
        vocab = list(hf.tokenizer["model"]["vocab"].keys())
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
