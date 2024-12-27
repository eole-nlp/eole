#!/usr/bin/env python
import os
import re
import json
import torch
import pyonmttok

import huggingface_hub
import safetensors
from safetensors.torch import save_file
from sentencepiece import SentencePieceProcessor

from eole.inputters.inputter import vocabs_to_dict
from eole.constants import DefaultTokens, TORCH_DTYPES
from eole.config.models import (
    EmbeddingsConfig,
    TransformerEncoderModelConfig,
    TransformerLMModelConfig,
)
from eole.config.run import TrainConfig
from eole.config.training import TrainingConfig
from eole.config import recursive_model_fields_set
from eole.bin import BaseBin, register_bin


key_maps = {}
key_maps["LlamaForCausalLM"] = {
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

key_maps["Qwen2ForCausalLM"] = key_maps["LlamaForCausalLM"]

key_maps["MistralForCausalLM"] = key_maps["LlamaForCausalLM"]
key_maps["MixtralForCausalLM"] = {
    "decoder_layer_prefix": "model.layers.",
    "tgt_emb.embeddings.weight": "model.embed_tokens.weight",
    "decoder.layer_norm.weight": "model.norm.weight",
    "generator.weight": "lm_head.weight",
    ".self_attn.linear_query.": ".self_attn.q_proj.",
    ".self_attn.linear_keys.": ".self_attn.k_proj.",
    ".self_attn.linear_values.": ".self_attn.v_proj.",
    ".self_attn.final_linear.": ".self_attn.o_proj.",
    ".input_layernorm.weight": ".input_layernorm.weight",
    ".mlp.gate.weight": ".block_sparse_moe.gate.weight",
    ".mlp.experts.0.gate_up_proj.": ".block_sparse_moe.experts.0.w1.",
    ".mlp.experts.0.down_proj.": ".block_sparse_moe.experts.0.w2.",
    ".mlp.experts.0.up_proj.": ".block_sparse_moe.experts.0.w3.",
    ".mlp.experts.0.layer_norm.weight": ".post_attention_layernorm.weight",
    ".mlp.experts.1.gate_up_proj.": ".block_sparse_moe.experts.1.w1.",
    ".mlp.experts.1.down_proj.": ".block_sparse_moe.experts.1.w2.",
    ".mlp.experts.1.up_proj.": ".block_sparse_moe.experts.1.w3.",
    ".mlp.experts.1.layer_norm.weight": ".post_attention_layernorm.weight",
    ".mlp.experts.2.gate_up_proj.": ".block_sparse_moe.experts.2.w1.",
    ".mlp.experts.2.down_proj.": ".block_sparse_moe.experts.2.w2.",
    ".mlp.experts.2.up_proj.": ".block_sparse_moe.experts.2.w3.",
    ".mlp.experts.2.layer_norm.weight": ".post_attention_layernorm.weight",
    ".mlp.experts.3.gate_up_proj.": ".block_sparse_moe.experts.3.w1.",
    ".mlp.experts.3.down_proj.": ".block_sparse_moe.experts.3.w2.",
    ".mlp.experts.3.up_proj.": ".block_sparse_moe.experts.3.w3.",
    ".mlp.experts.3.layer_norm.weight": ".post_attention_layernorm.weight",
    ".mlp.experts.4.gate_up_proj.": ".block_sparse_moe.experts.4.w1.",
    ".mlp.experts.4.down_proj.": ".block_sparse_moe.experts.4.w2.",
    ".mlp.experts.4.up_proj.": ".block_sparse_moe.experts.4.w3.",
    ".mlp.experts.4.layer_norm.weight": ".post_attention_layernorm.weight",
    ".mlp.experts.5.gate_up_proj.": ".block_sparse_moe.experts.5.w1.",
    ".mlp.experts.5.down_proj.": ".block_sparse_moe.experts.5.w2.",
    ".mlp.experts.5.up_proj.": ".block_sparse_moe.experts.5.w3.",
    ".mlp.experts.5.layer_norm.weight": ".post_attention_layernorm.weight",
    ".mlp.experts.6.gate_up_proj.": ".block_sparse_moe.experts.6.w1.",
    ".mlp.experts.6.down_proj.": ".block_sparse_moe.experts.6.w2.",
    ".mlp.experts.6.up_proj.": ".block_sparse_moe.experts.6.w3.",
    ".mlp.experts.6.layer_norm.weight": ".post_attention_layernorm.weight",
    ".mlp.experts.7.gate_up_proj.": ".block_sparse_moe.experts.7.w1.",
    ".mlp.experts.7.down_proj.": ".block_sparse_moe.experts.7.w2.",
    ".mlp.experts.7.up_proj.": ".block_sparse_moe.experts.7.w3.",
    ".mlp.experts.7.layer_norm.weight": ".post_attention_layernorm.weight",
}
key_maps["PhiForCausalLM"] = {
    "decoder_layer_prefix": "model.layers.",
    "tgt_emb.embeddings.weight": "model.embed_tokens.weight",
    "decoder.layer_norm.weight": "model.final_layernorm.weight",
    "decoder.layer_norm.bias": "model.final_layernorm.bias",
    "generator.weight": "lm_head.weight",
    "generator.bias": "lm_head.bias",
    ".self_attn.linear_query.": ".self_attn.q_proj.",
    ".self_attn.linear_keys.": ".self_attn.k_proj.",
    ".self_attn.linear_values.": ".self_attn.v_proj.",
    ".self_attn.final_linear.": ".self_attn.dense.",
    ".mlp.gate_up_proj.": ".mlp.fc1.",
    ".mlp.down_proj.": ".mlp.fc2.",
    ".input_layernorm.weight": (".input_layernorm.weight", ""),
    ".input_layernorm.bias": (".input_layernorm.bias", ""),
}
key_maps["Phi3ForCausalLM"] = {
    "decoder_layer_prefix": "model.layers.",
    "tgt_emb.embeddings.weight": "model.embed_tokens.weight",
    "decoder.layer_norm.weight": "model.norm.weight",
    "generator.weight": "lm_head.weight",
    ".self_attn.linear_query.": (".self_attn.qkv_proj.", "[:hidden_size, :]"),
    ".self_attn.linear_keys.": (
        ".self_attn.qkv_proj.",
        "[hidden_size:2*hidden_size, :]",
    ),
    ".self_attn.linear_values.": (".self_attn.qkv_proj.", "[-hidden_size:, :]"),
    ".self_attn.final_linear.": ".self_attn.o_proj.",
    ".mlp.gate_up_proj.": (".mlp.gate_up_proj.", "[:transformer_ff, :]"),
    ".mlp.down_proj.": ".mlp.down_proj.",
    ".mlp.up_proj.": (".mlp.gate_up_proj.", "[transformer_ff:, :]"),
    ".input_layernorm.weight": ".input_layernorm.weight",
    ".post_attention_layernorm.weight": ".post_attention_layernorm.weight",
}

# note: weights are transposed in Linear, not in Conv1D, which is used in HF
key_maps["GPT2LMHeadModel"] = {
    "decoder_layer_prefix": "h.",
    "tgt_emb.embeddings.weight": "wte.weight",
    "generator.weight": "wte.weight",  # shared with embeddings
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
}

key_maps["XLMRobertaXLForMaskedLM"] = {
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
}

ln_table = {
    "LlamaForCausalLM": "rms",
    "MistralForCausalLM": "rms",
    "MixtralForCausalLM": "rms",
    "PhiForCausalLM": "standard",
    "Phi3ForCausalLM": "rms",
    "GPT2LMHeadModel": "standard",
    "XLMRobertaXLForMaskedLM": "standard",
    "Qwen2ForCausalLM": "rms",  # to check if Qwen2RMS is similar to the others
}

act_table = {
    "LlamaForCausalLM": "gated-silu",
    "MistralForCausalLM": "gated-silu",
    "MixtralForCausalLM": "gated-silu",
    "PhiForCausalLM": "gelu",
    "Phi3ForCausalLM": "gated-silu",
    "GPT2LMHeadModel": "gelu",
    "XLMRobertaXLForMaskedLM": "gelu",
    "Qwen2ForCausalLM": "gated-silu",
}

arch_table = {
    "LlamaForCausalLM": TransformerLMModelConfig,
    "MistralForCausalLM": TransformerLMModelConfig,
    "MixtralForCausalLM": TransformerLMModelConfig,
    "PhiForCausalLM": TransformerLMModelConfig,
    "Phi3ForCausalLM": TransformerLMModelConfig,
    "GPT2LMHeadModel": TransformerLMModelConfig,
    "XLMRobertaXLForMaskedLM": TransformerEncoderModelConfig,
    "Qwen2ForCausalLM": TransformerLMModelConfig,
}

tok_table = {
    "LlamaForCausalLM": "huggingface_tokenize",
    "MistralForCausalLM": "huggingface_tokenize",
    "MixtralForCausalLM": "huggingface_tokenize",
    "PhiForCausalLM": "huggingface_tokenize",
    "Phi3ForCausalLM": "huggingface_tokenize",
    "GPT2LMHeadModel": "huggingface_tokenize",
    "XLMRobertaXLForMaskedLM": "huggingface_tokenize",
}


class Tokenizer:
    def __init__(self, model_path: str):
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()
        self.vocab = [self.sp_model.id_to_piece(i) for i in range(self.n_words)]


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
            help="Specify which dtype to save model parameters into, "
            "default will keep the same as the input.",
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
        os.makedirs(args.output, exist_ok=True)
        if os.path.exists(args.model_dir):
            if os.path.exists(os.path.join(args.model_dir, "config.json")):
                config_path = os.path.join(args.model_dir, "config.json")
            else:
                raise ValueError(
                    "You used a local directory but config.json is missing"
                )
            if os.path.exists(
                os.path.join(args.model_dir, "model.safetensors.index.json")
            ):
                wmap_path = os.path.join(args.model_dir, "model.safetensors.index.json")
            elif os.path.exists(
                os.path.join(args.model_dir, "pytorch_model.bin.index.json")
            ):
                wmap_path = os.path.join(args.model_dir, "pytorch_model.bin.index.json")
            elif os.path.exists(os.path.join(args.model_dir, "model.safetensors")):
                wmap_path = None
                model_path = os.path.join(args.model_dir, "model.safetensors")
            elif os.path.exists(os.path.join(args.model_dir, "pytorch_model.bin")):
                wmap_path = None
                model_path = os.path.join(args.model_dir, "pytorch_model.bin")
            else:
                raise ValueError(
                    "Could not find any proper model configuration, please check your files"
                )
            if os.path.exists(os.path.join(args.model_dir, "tokenizer.model")):
                tokenizer_model = os.path.join(args.model_dir, "tokenizer.model")
            elif os.path.exists(
                os.path.join(args.model_dir, "sentencepiece.bpe.model")
            ):
                tokenizer_model = os.path.join(
                    args.model_dir, "sentencepiece.bpe.model"
                )
            else:
                tokenizer_model = None
            if os.path.exists(os.path.join(args.model_dir, "tokenizer.json")):
                tokenizer_json = os.path.join(args.model_dir, "tokenizer.json")

            elif tokenizer_model is None:
                raise ValueError(
                    "You used a local directory but tokenizer.model",
                    " and/or tokenizer.json are missing",
                )
            if os.path.exists(os.path.join(args.model_dir, "tokenizer_config.json")):
                tokenizer_config_json = os.path.join(
                    args.model_dir, "tokenizer_config.json"
                )
            else:
                tokenizer_config_json = None
            if os.path.exists(os.path.join(args.model_dir, "special_tokens_map.json")):
                tokenizer_config_json = os.path.join(
                    args.model_dir, "special_tokens_map.json"
                )
            else:
                tokenizer_config_json = None
            if os.path.exists(os.path.join(args.model_dir, "generation_config.json")):
                generation_config_json = os.path.join(
                    args.model_dir, "generation_config.json"
                )
            else:
                generation_config_json = None
        else:
            huggingface_model = args.model_dir
            directory_path = args.output
            os.makedirs(directory_path, exist_ok=True)
            try:
                tokenizer_model = huggingface_hub.hf_hub_download(
                    repo_id=args.model_dir,
                    filename="tokenizer.model",
                    token=args.token,
                    local_dir=args.output,
                )
            except huggingface_hub.utils.EntryNotFoundError:
                try:
                    tokenizer_model = huggingface_hub.hf_hub_download(
                        repo_id=args.model_dir,
                        filename="sentencepiece.bpe.model",
                        token=args.token,
                        local_dir=args.output,
                    )
                except huggingface_hub.utils.EntryNotFoundError:
                    tokenizer_model = None
            try:
                tokenizer_json = huggingface_hub.hf_hub_download(
                    repo_id=args.model_dir,
                    filename="tokenizer.json",
                    token=args.token,
                    local_dir=args.output,
                )
            except huggingface_hub.utils.EntryNotFoundError:
                tokenizer_json = None
                if tokenizer_model is None:
                    raise huggingface_hub.utils.EntryNotFoundError(
                        "Make sure the repo contains tokenizer.model or tokenizer.json"
                    )
            try:
                config_path = huggingface_hub.hf_hub_download(
                    repo_id=args.model_dir,
                    filename="config.json",
                    token=args.token,
                )
            except huggingface_hub.utils.EntryNotFoundError:
                raise huggingface_hub.utils.EntryNotFoundError(
                    "Something went wrong the repo does not contain any config.json file"
                )
            try:
                tokenizer_config_json = huggingface_hub.hf_hub_download(
                    repo_id=args.model_dir,
                    filename="tokenizer_config.json",
                    token=args.token,
                )
            except huggingface_hub.utils.EntryNotFoundError:
                raise huggingface_hub.utils.EntryNotFoundError(
                    "Something went wrong the repo does not contain any tokenizer_config.json file"
                )
            try:
                generation_config_json = huggingface_hub.hf_hub_download(
                    repo_id=args.model_dir,
                    filename="generation_config.json",
                    token=args.token,
                )
            except huggingface_hub.utils.EntryNotFoundError:
                raise huggingface_hub.utils.EntryNotFoundError(
                    "Something went wrong the repo does not contain any generation_config.json file"
                )
            try:
                wmap_path = huggingface_hub.hf_hub_download(
                    repo_id=args.model_dir,
                    filename="model.safetensors.index.json",
                    token=args.token,
                )
            except huggingface_hub.utils.EntryNotFoundError:
                try:
                    wmap_path = huggingface_hub.hf_hub_download(
                        repo_id=args.model_dir,
                        filename="pytorch_model.bin.index.json",
                        token=args.token,
                    )
                except huggingface_hub.utils.EntryNotFoundError:
                    try:
                        model_path = huggingface_hub.hf_hub_download(
                            repo_id=args.model_dir,
                            filename="model.safetensors",
                            token=args.token,
                        )
                        wmap_path = None
                    except huggingface_hub.utils.EntryNotFoundError:
                        try:
                            model_path = huggingface_hub.hf_hub_download(
                                repo_id=args.model_dir,
                                filename="pytorch_model.bin",
                                token=args.token,
                            )
                            wmap_path = None
                        except huggingface_hub.utils.EntryNotFoundError:
                            raise huggingface_hub.utils.EntryNotFoundError(
                                "No valid model files found"
                            )
            try:
                try:
                    special_tokens_json = huggingface_hub.hf_hub_download(
                        repo_id=args.model_dir,
                        filename="special_tokens_map.json",
                        token=args.token,
                    )
                except huggingface_hub.utils.EntryNotFoundError:
                    raise huggingface_hub.utils.EntryNotFoundError(
                        "Something went wrong the repo does not contain"
                        "any special_tokens_map.json file"
                    )
            except Exception as e:
                if isinstance(e, huggingface_hub.utils.EntryNotFoundError):
                    special_tokens_json = None
                    print(e)

        with open(config_path, encoding="utf-8") as fconfig:
            config = json.load(fconfig)

        arch = config["architectures"][0]

        # FROM THIS n_layers is the same for decoder/encoder
        # for encoder/decoder models like T5 if the number differs will require adaptation
        if "num_hidden_layers" in config.keys():
            n_layers = config["num_hidden_layers"]
        elif "n_layer" in config.keys():
            n_layers = config["n_layer"]
        else:
            raise ValueError("Can't find the number of layers in the config.json file")
        if "hidden_size" in config.keys():
            src_word_vec_size = config["hidden_size"]
            tgt_word_vec_size = config["hidden_size"]
            hidden_size = config["hidden_size"]
        elif "n_embd" in config.keys():
            src_word_vec_size = config["n_embd"]
            tgt_word_vec_size = config["n_embd"]
            hidden_size = config["n_embd"]
        else:
            raise ValueError("can't find the model hidden size in the config.json file")
        if "num_attention_heads" in config.keys():
            heads = config["num_attention_heads"]
        elif "n_head" in config.keys():
            heads = config["n_head"]
        else:
            raise ValueError("can't find the number of heads in the config.json file")

        vocab_size = config["vocab_size"]
        if "intermediate_size" in config.keys():
            transformer_ff = config["intermediate_size"]
        else:
            transformer_ff = hidden_size * 4
        mlp_activation_fn = act_table[arch]
        layer_norm = ln_table[arch]

        if "multi_query" in config.keys() and config["multi_query"]:
            heads_kv = 1  # might be usefull for old config
        elif (
            "num_key_value_heads" in config.keys()
            and config["num_key_value_heads"] != heads
        ):
            heads_kv = config["num_key_value_heads"]
        elif "num_kv_heads" in config.keys() and config["num_kv_heads"] != heads:
            heads_kv = config["num_kv_heads"]
        elif "n_head_kv" in config.keys() and config["n_head_kv"] != heads:
            heads_kv = config["n_head_kv"]
        else:
            heads_kv = heads

        if "head_dim" in config.keys():
            head_dim = config["head_dim"]
        else:
            head_dim = None

        if "parallel_attn" in config.keys():
            parallel_residual = config["parallel_attn"]
        else:
            parallel_residual = False

        if "rms_norm_eps" in config.keys():
            norm_eps = config["rms_norm_eps"]
        elif "layer_norm_epsilon" in config.keys():
            norm_eps = config["layer_norm_epsilon"]
        elif "layer_norm_eps" in config.keys():
            norm_eps = config["layer_norm_eps"]
        else:
            norm_eps = 1e-6
        rope_config = {}
        if "rope_theta" in config.keys():
            rope_config["rotary_theta"] = config["rope_theta"]
        if "rotary_dim" in config.keys():
            rope_config["rotary_dim"] = config["rotary_dim"]
        elif "partial_rotary_factor" in config.keys():
            rope_config["rotary_dim"] = int(
                config["partial_rotary_factor"] * (hidden_size // heads)
            )
        if config.get("rope_scaling", None) is not None:
            rope_config["scaling_type"] = config["rope_scaling"].get("rope_type", None)
            rope_config["scaling_factor"] = config["rope_scaling"].get("factor", 8.0)
            rope_config["low_freq_factor"] = config["rope_scaling"].get(
                "low_freq_factor", 1.0
            )
            rope_config["high_freq_factor"] = config["rope_scaling"].get(
                "high_freq_factor", 4.0
            )
            rope_config["original_max_position_embeddings"] = config[
                "rope_scaling"
            ].get("original_max_position_embeddings", 8192)
        if "sliding_window" in config.keys():
            sliding_window = config["sliding_window"]
            if sliding_window is None:
                sliding_window = 4096
        else:
            sliding_window = 0
        if "num_local_experts" in config.keys():
            num_experts = config["num_local_experts"]
        else:
            num_experts = 0
        if "num_experts_per_tok" in config.keys():
            num_experts_per_tok = config["num_experts_per_tok"]
        else:
            num_experts_per_tok = 0
        if "quantization_config" in config.keys():
            if (
                "quant_method" in config["quantization_config"].keys()
                and config["quantization_config"]["quant_method"] == "awq"
            ):
                if "backend" in config["quantization_config"].keys():
                    backend = config["quantization_config"]["backend"]
                    if backend == "llm-awq":
                        quant_type = "awq_gemv"
                    elif backend == "autoawq":
                        if config["quantization_config"]["version"].lower() == "gemm":
                            quant_type = "awq_gemm"
                        elif config["quantization_config"]["version"].lower() == "gemv":
                            quant_type = "awq_gemv"
                        else:
                            raise ValueError("Unknown quantization config")
                    else:
                        raise ValueError("Unknown backend config")
                else:
                    print("Backend not specified in config, using Autoawq")
                    if config["quantization_config"]["version"].lower() == "gemm":
                        quant_type = "awq_gemm"
                    elif config["quantization_config"]["version"].lower() == "gemv":
                        quant_type = "awq_gemv"
                    else:
                        raise ValueError("Unknown quantization config")
            else:
                raise ValueError("Can convert only awq models for now")
            if "bits" in config["quantization_config"].keys():
                w_bit = config["quantization_config"]["bits"]
            else:
                w_bit = config["quantization_config"]["w_bit"]
            if "group_size" in config["quantization_config"].keys():
                group_size = config["quantization_config"]["group_size"]
            else:
                group_size = config["quantization_config"]["q_group_size"]

            quant_layers = [
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
            quant_type = ""
            w_bit = 0
            group_size = 0
            quant_layers = []
            params = ["weight", "bias"]

        share_decoder_embeddings = config.get("tie_word_embeddings", False)

        add_qkvbias = False
        add_final_linear_bias = False
        add_ffnbias = False
        shared_layer_norm = False
        rope_config["rotary_interleave"] = False
        position_encoding = {
            "position_encoding_type": "Rotary",
            "n_positions": 0,
        }
        left_pad = True
        optional_eos = []
        mapped_tokens = []
        gpt2_pretok = False
        generator_bias = False

        # ALL THESE IF SHOULD BE HANDLED IN MAPPINGS
        if arch == "PhiForCausalLM":
            parallel_residual = True
            shared_layer_norm = True
            add_qkvbias = True
            add_final_linear_bias = True
            add_ffnbias = True
            rope_config["rotary_interleave"] = False
        if arch == "GPT2LMHeadModel":
            parallel_residual = False
            shared_layer_norm = True
            add_qkvbias = True
            add_final_linear_bias = True
            add_ffnbias = True
            position_encoding = {
                "position_encoding_type": "Learned",
                "n_positions": 1024,
            }
            left_pad = False
        if arch == "XLMRobertaXLForMaskedLM":
            add_qkvbias = True
            add_final_linear_bias = True
            add_ffnbias = True
            position_encoding = {
                "position_encoding_type": "Learned",
                "n_positions": 514,
                "position_shift": 2,
            }
            left_pad = False
        if arch == "Qwen2ForCausalLM":
            add_qkvbias = True
            add_final_linear_bias = False

        decoder_layer_prefix = key_maps[arch].get("decoder_layer_prefix", None)
        encoder_layer_prefix = key_maps[arch].get("encoder_layer_prefix", None)

        if wmap_path:
            with open(wmap_path, encoding="utf-8") as fweights:
                wmap = json.load(fweights)

        def get_load_ckpt(dir_path, file_path):
            if os.path.exists(os.path.join(dir_path, file_path)):
                ckpt_path = os.path.join(dir_path, file_path)
            else:
                try:
                    ckpt_path = huggingface_hub.hf_hub_download(
                        repo_id=args.model_dir,
                        filename=file_path,
                        token=args.token,
                    )
                except huggingface_hub.utils.EntryNotFoundError:
                    raise huggingface_hub.utils.EntryNotFoundError(
                        "Checkpoint not found on the hub"
                    )
                except PermissionError:
                    ckpt_path = os.path.join(dir_path, file_path)
            if ckpt_path[-4:] == ".bin":
                checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
            else:
                checkpoint = ckpt_path

            return checkpoint

        def get_weight(checkpoint, tensor_name):
            if isinstance(checkpoint, dict):
                if tensor_name in checkpoint.keys():
                    return checkpoint[tensor_name].contiguous()
                else:
                    return None
            else:
                with safetensors.safe_open(
                    checkpoint, framework="pt", device="cpu"
                ) as f:
                    if tensor_name in f.keys():
                        return f.get_tensor(tensor_name).contiguous()
                    else:
                        return None

        # Deduce dtype from args or config, or default to fp16
        if args.dtype is not None:
            compute_dtype = args.dtype
        elif "torch_dtype" in config.keys():
            compute_dtype = config["torch_dtype"]
        else:
            compute_dtype = "fp16"

        for shard in range(args.nshards):

            print("starting output shard: %d/%d" % (shard + 1, args.nshards))
            eole_safetensor = {}

            if shard == 0:
                targetlist = [
                    "tgt_emb.embeddings.weight",
                    "tgt_emb.pe.weight",
                    "decoder.layer_norm.weight",
                    "decoder.layer_norm.bias",
                    "src_emb.embeddings.weight",
                    "src_emb.pe.weight",
                    "encoder.layer_norm.weight",
                    "encoder.layer_norm.bias",
                    "generator.weight",
                ]
                if share_decoder_embeddings:
                    targetlist.remove("generator.weight")
                for target in targetlist:
                    if target in key_maps[arch].keys():
                        source = key_maps[arch][target]
                        if wmap_path:
                            checkpoint = get_load_ckpt(
                                os.path.split(wmap_path)[0], wmap["weight_map"][source]
                            )
                        else:
                            checkpoint = get_load_ckpt(*os.path.split(model_path))
                        w = get_weight(checkpoint, source)
                        if w is not None:
                            eole_safetensor[target] = w

                        if target == "generator.bias":
                            generator_bias = True

            if wmap_path:
                weightmap = wmap["weight_map"]
                ckpt_list = []
                for key in weightmap.keys():
                    if (
                        (
                            (
                                decoder_layer_prefix is not None
                                and key.startswith(decoder_layer_prefix)
                            )
                            or (
                                encoder_layer_prefix is not None
                                and key.startswith(encoder_layer_prefix)
                            )
                        )
                        and int(key.split(".")[2])
                        in range(
                            -(n_layers // -args.nshards) * shard,
                            min(
                                -(n_layers // -args.nshards) * (shard + 1),
                                n_layers,
                            ),
                            1,
                        )
                        and weightmap[key] not in ckpt_list
                    ):
                        ckpt_list.append(weightmap[key])
                        print(weightmap[key])
            else:
                ckpt_list = [model_path]

            for ckpt in ckpt_list:
                print("Loading %s" % ckpt)
                if wmap_path:
                    checkpoint = get_load_ckpt(os.path.split(wmap_path)[0], ckpt)
                else:
                    checkpoint = get_load_ckpt(*os.path.split(model_path))
                for i in range(
                    -(n_layers // -args.nshards) * shard,
                    min(-(n_layers // -args.nshards) * (shard + 1), n_layers),
                    1,
                ):

                    # this is an ugly fix to handle decoder only, encoder only,
                    # encoder-decoder models
                    for layer_prefix in [
                        prefix
                        for prefix in [
                            decoder_layer_prefix,
                            encoder_layer_prefix,
                        ]
                        if prefix is not None
                    ]:
                        if layer_prefix == decoder_layer_prefix:
                            eole_prefix = "decoder.transformer_layers."
                        elif layer_prefix == encoder_layer_prefix:
                            eole_prefix = "encoder.transformer_layers."
                        else:
                            print("error.")
                            exit()
                        for param in params:
                            targetlist = [
                                ".self_attn.linear_query.",
                                ".self_attn.linear_keys.",
                                ".self_attn.linear_values.",
                                ".self_attn.final_linear.",
                                ".mlp.gate_up_proj.",
                                ".mlp.down_proj.",
                                ".mlp.up_proj.",
                                ".mlp.experts.0.gate_up_proj.",
                                ".mlp.experts.0.down_proj.",
                                ".mlp.experts.0.up_proj.",
                                ".mlp.experts.1.gate_up_proj.",
                                ".mlp.experts.1.down_proj.",
                                ".mlp.experts.1.up_proj.",
                                ".mlp.experts.2.gate_up_proj.",
                                ".mlp.experts.2.down_proj.",
                                ".mlp.experts.2.up_proj.",
                                ".mlp.experts.3.gate_up_proj.",
                                ".mlp.experts.3.down_proj.",
                                ".mlp.experts.3.up_proj.",
                                ".mlp.experts.4.gate_up_proj.",
                                ".mlp.experts.4.down_proj.",
                                ".mlp.experts.4.up_proj.",
                                ".mlp.experts.5.gate_up_proj.",
                                ".mlp.experts.5.down_proj.",
                                ".mlp.experts.5.up_proj.",
                                ".mlp.experts.6.gate_up_proj.",
                                ".mlp.experts.6.down_proj.",
                                ".mlp.experts.6.up_proj.",
                                ".mlp.experts.7.gate_up_proj.",
                                ".mlp.experts.7.down_proj.",
                                ".mlp.experts.7.up_proj.",
                            ]
                            for target in targetlist:
                                if target in key_maps[arch].keys():
                                    source = key_maps[arch][target]
                                    if type(source) == tuple:
                                        srckey = source[0]
                                        srcmap = source[1]
                                    else:
                                        srckey = source
                                    w = get_weight(
                                        checkpoint,
                                        layer_prefix + str(i) + srckey + param,
                                    )

                                    if w is not None:
                                        if type(source) == tuple:
                                            w = eval("w" + srcmap).contiguous()
                                        eole_safetensor[
                                            eole_prefix + str(i) + target + param
                                        ] = w

                        if shared_layer_norm:
                            idx = 0
                        else:
                            idx = 1
                        for p in ["weight", "bias"]:
                            if ".input_layernorm." + p in key_maps[arch].keys():
                                if (
                                    type(key_maps[arch][".input_layernorm." + p])
                                    == tuple
                                ):
                                    w = get_weight(
                                        checkpoint,
                                        layer_prefix
                                        + str(i)
                                        + key_maps[arch][".input_layernorm." + p][idx],
                                    )
                                else:
                                    w = get_weight(
                                        checkpoint,
                                        layer_prefix
                                        + str(i)
                                        + key_maps[arch][".input_layernorm." + p],
                                    )
                                if w is not None:
                                    eole_safetensor[
                                        eole_prefix + str(i) + ".input_layernorm." + p
                                    ] = w
                            if ".layer_norm_res." + p in key_maps[arch].keys():
                                w = get_weight(
                                    checkpoint,
                                    layer_prefix
                                    + str(i)
                                    + key_maps[arch][".layer_norm_res." + p],
                                )
                                if w is not None:
                                    eole_safetensor[
                                        eole_prefix + str(i) + ".layer_norm_res." + p
                                    ] = w
                            if (
                                ".post_attention_layernorm." + p
                                in key_maps[arch].keys()
                            ):
                                w = get_weight(
                                    checkpoint,
                                    layer_prefix
                                    + str(i)
                                    + key_maps[arch][".post_attention_layernorm." + p],
                                )
                                if w is not None:
                                    eole_safetensor[
                                        eole_prefix
                                        + str(i)
                                        + ".post_attention_layernorm."
                                        + p
                                    ] = w

                            if ".mlp.gate." + p in key_maps[arch].keys():
                                w = get_weight(
                                    checkpoint,
                                    layer_prefix
                                    + str(i)
                                    + key_maps[arch][".mlp.gate." + p],
                                )
                                if w is not None:
                                    eole_safetensor[
                                        eole_prefix + str(i) + ".mlp.gate." + p
                                    ] = w

                            for j in range(num_experts):
                                if (
                                    f".mlp.experts.{j}.layer_norm." + p
                                    in key_maps[arch].keys()
                                ):
                                    w = get_weight(
                                        checkpoint,
                                        layer_prefix
                                        + str(i)
                                        + key_maps[arch][
                                            f".mlp.experts.{j}.layer_norm." + p
                                        ],
                                    )
                                    if w is not None:
                                        eole_safetensor[
                                            eole_prefix
                                            + str(i)
                                            + f".mlp.experts.{j}.layer_norm."
                                            + p
                                        ] = w

            # Convert to another dtype if specified
            if args.dtype is not None:
                for key in eole_safetensor.keys():
                    eole_safetensor[key] = eole_safetensor[key].to(
                        TORCH_DTYPES[compute_dtype]
                    )
            print("Saving output model shard: %d" % shard)
            save_file(
                eole_safetensor,
                os.path.join(args.output, "model.{:02d}.safetensors".format(shard)),
            )

        directory_path = args.output
        if directory_path != "":
            os.makedirs(directory_path, exist_ok=True)

        if tokenizer_config_json is not None:
            with open(tokenizer_config_json, encoding="utf-8") as f:
                data = json.load(f)
                if "add_bos_token" in data.keys():
                    add_bos_token = data["add_bos_token"]
                elif "bos_token" in data.keys():
                    add_bos_token = True
                else:
                    add_bos_token = False
                if "chat_template" in data.keys():
                    chat_template = {"chat_template": data["chat_template"]}
                else:
                    chat_template = {}
                # Not sure if we could do much cleaner to retrieve optional eos tokens
                eos_token_id = config.get("eos_token_id", None)
                if isinstance(eos_token_id, list):
                    if "added_tokens_decoder" in data.keys():
                        eos_tokens = [
                            data["added_tokens_decoder"][str(index)]["content"]
                            for index in eos_token_id
                        ]
                        optional_eos = eos_tokens[1:]
                # Automatically convert added_tokens into mapped_tokens
                if "added_tokens_decoder" in data.keys():
                    mapped_tokens = [
                        (
                            token["content"],
                            re.sub(
                                r"<\|([^|]*)\|>", "\uff5f\\1\uff60", token["content"]
                            ),
                        )
                        for token in data["added_tokens_decoder"].values()
                    ]
        else:
            add_bos_token = True

        vocabs = {"specials": {}}

        if special_tokens_json is not None:
            with open(special_tokens_json, encoding="utf-8") as f:
                special_tokens_map = json.load(f)
                for token_name in ["bos_token", "unk_token", "eos_token", "pad_token"]:
                    token = special_tokens_map.get(token_name, None)
                    if isinstance(token, list):
                        vocabs["specials"][token_name] = token[0]
                    elif isinstance(token, str):
                        vocabs["specials"][token_name] = token
                    elif isinstance(token, dict):
                        vocabs["specials"][token_name] = token["content"]
        elif tokenizer_json is not None:
            with open(tokenizer_json, encoding="utf-8") as f:
                data = json.load(f)
                vocab = {v: k for k, v in data["model"]["vocab"].items()}

        if generation_config_json is not None:
            with open(generation_config_json, encoding="utf-8") as f:
                data = json.loads(
                    f.read().replace(",\n}", "\n}")
                )  # dirty patch to remove trailing comma...
                generation_config_dict = {}
                # we probably need a better mapping at some point
                keys = ["top_k", "top_p", "temperature", "max_length"]
                for key in keys:
                    if key in data.keys():
                        generation_config_dict[key] = data[key]
        if (
            tokenizer_model is not None
        ):  # sentencepiece mode (might be good to check it's a SP model)
            src_subword_type = "sentencepiece"
            tokenizer_basename = os.path.basename(tokenizer_model)
            tokenizer = Tokenizer(model_path=tokenizer_model)
            vocab = tokenizer.vocab
            if tokenizer_json is not None:
                # We need to add 'added_tokens' that are not in the SP model
                with open(tokenizer_json, encoding="utf-8") as f:
                    data = json.load(f)
                newtokens = [
                    tok["content"]
                    for tok in data["added_tokens"]
                    if tok["content"] not in vocab
                ]
                vocab.extend(newtokens)
                for tok in data["added_tokens"]:
                    vocab[tok["id"]] = tok["content"]
            src_vocab = pyonmttok.build_vocab_from_tokens(
                vocab,
            )
        else:  # # BPE mode - we leverage the HF tokenizer.json info
            src_subword_type = "bpe"
            with open(tokenizer_json, encoding="utf-8") as f:
                data = json.load(f)
                # gpt2_pretok
                pretokenizers = data.get("pre_tokenizer", {}).get("pretokenizers", [{}])
                pre_tokenizer = data.get("pre_tokenizer", None)
                pretokenizers = pre_tokenizer.get("pretokenizers", None)
                if pretokenizers is None:
                    pretokenizers = [pre_tokenizer]
                for pretokenizer in pretokenizers:
                    if pretokenizer.get("type", None) == "ByteLevel":
                        gpt2_pretok = True
                vocab = [
                    tok if tok != "Ā" else DefaultTokens.PAD
                    # "Ā" is '\x00' in unicode (cf tokenize.py gpt2 mapping)
                    for tok in data["model"]["vocab"]
                ]
                if DefaultTokens.PAD in vocab:
                    vocabs["specials"]["pad_token"] = DefaultTokens.PAD
            voc_size = len(vocab)
            if vocab_size > voc_size:
                for i in range(vocab_size - voc_size):
                    vocab.append(DefaultTokens.VOCAB_PAD + str(i))
            for tok in data["added_tokens"]:
                vocab[tok["id"]] = tok["content"]
            src_vocab = pyonmttok.build_vocab_from_tokens(vocab)

            for token_name in ["bos_token", "unk_token", "eos_token", "pad_token"]:
                if f"{token_name}_id" in config.keys():
                    token = config[f"{token_name}_id"]
                    if isinstance(token, list):
                        vocabs["specials"][token_name] = vocab[token[0]]
                    elif isinstance(token, str):
                        vocabs["specials"][token_name] = vocab[token]

            tokenizer_basename = "bpe.model"

            with open(
                os.path.join(directory_path, tokenizer_basename), "w", encoding="utf-8"
            ) as bpemodel:
                bpemodel.write("v3;false;false;false;Ġ;Ġ\n")
                for merge in data["model"]["merges"]:
                    if isinstance(merge, str):
                        bpemodel.write(merge + "\n")
                    elif isinstance(merge, list):
                        bpemodel.write(" ".join(merge) + "\n")
                    else:
                        raise NotImplementedError(
                            f"Type {type(merge)} is not supported for BPE merges."
                        )

        if arch in tok_table.keys() and args.tokenizer == "hf":
            transforms = [
                tok_table[arch]
            ]  # , "filtertoolong"] filtertoolong not plug-n-play with id_tokenize
        else:
            transforms = ["onmt_tokenize"]

        match tok_table.get(arch, None), args.tokenizer:
            case "huggingface_tokenize", "hf":
                transforms_configs = {
                    tok_table[arch]: {"max_length": 512},
                }
            case _:
                # not used right now, but keeping for reference
                transforms_configs = {
                    "filtertoolong": {"src_seq_length": 512, "tgt_seq_length": 512},
                    "onmt_tokenize": {
                        "src_subword_type": src_subword_type,
                        "src_subword_model": os.path.join(
                            "${MODEL_PATH}", tokenizer_basename
                        ),
                        "gpt2_pretok": gpt2_pretok,
                        "mapped_tokens": mapped_tokens,
                    },
                }

        vocabs["src"] = src_vocab
        vocabs["tgt"] = src_vocab
        if add_bos_token:
            vocabs["decoder_start_token"] = vocabs["specials"]["bos_token"]
        else:
            vocabs["decoder_start_token"] = ""
        vocab_dict = vocabs_to_dict(vocabs)
        with open(
            os.path.join(directory_path, "vocab.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(vocab_dict, f, indent=2, ensure_ascii=False)

        with open(
            os.path.join(directory_path, "vocab.txt"), "w", encoding="utf-8"
        ) as vocabfile:
            for tok in vocab_dict["src"]:
                vocabfile.write(tok + "\n")

        config = TrainConfig(
            data=None,
            skip_empty_level="silent",  # default is "warning"
            save_data=None,
            n_sample=0,
            src_vocab=None,
            tgt_vocab=None,
            share_vocab=True,
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,
            vocab_size_multiple=8,
            decoder_start_token=vocabs["decoder_start_token"],
            **vocabs["specials"],
            transforms=transforms,
            transforms_configs=transforms_configs,
            model=arch_table[arch](
                layers=n_layers,
                hidden_size=hidden_size,
                heads=heads,
                transformer_ff=transformer_ff,
                embeddings=EmbeddingsConfig(
                    src_word_vec_size=src_word_vec_size,
                    tgt_word_vec_size=tgt_word_vec_size,
                    **position_encoding,
                ),
                # src_word_vec_size=src_word_vec_size,
                # tgt_word_vec_size=tgt_word_vec_size,
                layer_norm=layer_norm,
                norm_eps=norm_eps,
                mlp_activation_fn=mlp_activation_fn,
                rope_config=rope_config,
                sliding_window=sliding_window,
                heads_kv=heads_kv,
                head_dim=head_dim,
                parallel_residual=parallel_residual,
                shared_layer_norm=shared_layer_norm,
                add_qkvbias=add_qkvbias,
                add_final_linear_bias=add_final_linear_bias,
                add_ffnbias=add_ffnbias,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                left_pad=left_pad,
                huggingface_model=huggingface_model,
                share_decoder_embeddings=share_decoder_embeddings,
                generator_bias=generator_bias,
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
                quant_layers=quant_layers,
                quant_type=quant_type,
                w_bit=w_bit,
                group_size=group_size,
            ),
        )
        config_dict = recursive_model_fields_set(config)

        inference_dict = {
            "optional_eos": optional_eos,
            # TODO: map other settings from HF decoding_config.json
            **generation_config_dict,
            **chat_template,
        }

        config_dict["inference"] = inference_dict

        with open(
            os.path.join(directory_path, "config.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
