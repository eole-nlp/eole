#!/usr/bin/env python
import json
import os
import shutil

import pyonmttok
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from sentencepiece import SentencePieceProcessor

from eole.bin import BaseBin, register_bin
from eole.config import recursive_model_fields_set
from eole.config.models import EmbeddingsConfig, TransformerEncoderDecoderScorerModelConfig
from eole.config.run import TrainConfig
from eole.config.training import TrainingConfig
from eole.inputters.inputter import vocabs_to_dict
from eole.utils.metricx_scorer import metricx_input_templates


def _slug(name):
    return name.replace("/", "--")


def _dtype(name):
    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[name]


def _load_state_dict(model_dir):
    state_dict = {}
    safetensors_files = sorted(name for name in os.listdir(model_dir) if name.endswith(".safetensors"))
    if safetensors_files:
        for filename in safetensors_files:
            state_dict.update(load_file(os.path.join(model_dir, filename), device="cpu"))
        return state_dict

    for filename in ["pytorch_model.bin", "model.bin"]:
        path = os.path.join(model_dir, filename)
        if os.path.exists(path):
            return torch.load(path, map_location="cpu")
    raise FileNotFoundError(f"No supported checkpoint file found in {model_dir}")


def _copy_tokenizer_assets(tokenizer_dir, output_dir):
    for filename in ["spiece.model", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src = os.path.join(tokenizer_dir, filename)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(output_dir, filename))


def _build_vocab(tokenizer_dir, output_dir):
    tokenizer_model = os.path.join(tokenizer_dir, "spiece.model")
    if not os.path.exists(tokenizer_model):
        raise FileNotFoundError(f"MetricX conversion requires spiece.model in tokenizer repo {tokenizer_dir}")

    sp_model = SentencePieceProcessor(model_file=tokenizer_model)
    vocab = [sp_model.id_to_piece(i) for i in range(sp_model.vocab_size())]
    src_vocab = pyonmttok.build_vocab_from_tokens(vocab)
    vocabs = {
        "src": src_vocab,
        "tgt": src_vocab,
        "decoder_start_token": "<pad>",
        "specials": {
            "bos_token": "<pad>",
            "pad_token": "<pad>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
        },
    }
    vocab_dict = vocabs_to_dict(vocabs)
    with open(os.path.join(output_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "vocab.txt"), "w", encoding="utf-8") as f:
        for token in vocab_dict["src"]:
            f.write(token + "\n")


def _get(state_dict, *names):
    for name in names:
        if name in state_dict:
            return state_dict[name]
    raise KeyError("Missing any of checkpoint keys: " + ", ".join(names))


def _convert_state_dict(state_dict, params, dtype):
    converted = {}
    decoder_layers = params["num_decoder_layers"]
    encoder_layers = params["num_layers"]
    hidden_size = params["d_model"]
    heads = params["num_heads"]
    dimperhead = hidden_size // heads

    shared = _get(state_dict, "shared.weight", "encoder.embed_tokens.weight")
    converted["src_emb.embeddings.weight"] = shared.to(dtype)
    converted["tgt_emb.embeddings.weight"] = shared.to(dtype)
    converted["encoder.layer_norm.weight"] = _get(state_dict, "encoder.final_layer_norm.weight").to(dtype)
    converted["decoder.layer_norm.weight"] = _get(state_dict, "decoder.final_layer_norm.weight").to(dtype)
    converted["generator.weight"] = _get(state_dict, "lm_head.weight", "shared.weight").to(dtype)

    for i in range(max(encoder_layers, decoder_layers)):
        if i < encoder_layers:
            src_prefix = f"encoder.block.{i}."
            dst_prefix = f"encoder.transformer_layers.{i}."
            converted[dst_prefix + "self_attn.linear_query.weight"] = (
                _get(state_dict, src_prefix + "layer.0.SelfAttention.q.weight") / (dimperhead**-0.5)
            ).to(dtype)
            converted[dst_prefix + "self_attn.linear_keys.weight"] = _get(
                state_dict, src_prefix + "layer.0.SelfAttention.k.weight"
            ).to(dtype)
            converted[dst_prefix + "self_attn.linear_values.weight"] = _get(
                state_dict, src_prefix + "layer.0.SelfAttention.v.weight"
            ).to(dtype)
            converted[dst_prefix + "self_attn.final_linear.weight"] = _get(
                state_dict, src_prefix + "layer.0.SelfAttention.o.weight"
            ).to(dtype)
            converted[dst_prefix + "self_attn.relative_attention_bias.weight"] = _get(
                state_dict, "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
            ).to(dtype)
            converted[dst_prefix + "input_layernorm.weight"] = _get(
                state_dict, src_prefix + "layer.0.layer_norm.weight"
            ).to(dtype)
            converted[dst_prefix + "post_attention_layernorm.weight"] = _get(
                state_dict, src_prefix + "layer.1.layer_norm.weight"
            ).to(dtype)
            converted[dst_prefix + "mlp.gate_up_proj.weight"] = _get(
                state_dict, src_prefix + "layer.1.DenseReluDense.wi_0.weight"
            ).to(dtype)
            converted[dst_prefix + "mlp.down_proj.weight"] = _get(
                state_dict, src_prefix + "layer.1.DenseReluDense.wo.weight"
            ).to(dtype)
            converted[dst_prefix + "mlp.up_proj.weight"] = _get(
                state_dict, src_prefix + "layer.1.DenseReluDense.wi_1.weight"
            ).to(dtype)

        if i < decoder_layers:
            src_prefix = f"decoder.block.{i}."
            dst_prefix = f"decoder.transformer_layers.{i}."
            converted[dst_prefix + "self_attn.linear_query.weight"] = (
                _get(state_dict, src_prefix + "layer.0.SelfAttention.q.weight") / (dimperhead**-0.5)
            ).to(dtype)
            converted[dst_prefix + "self_attn.linear_keys.weight"] = _get(
                state_dict, src_prefix + "layer.0.SelfAttention.k.weight"
            ).to(dtype)
            converted[dst_prefix + "self_attn.linear_values.weight"] = _get(
                state_dict, src_prefix + "layer.0.SelfAttention.v.weight"
            ).to(dtype)
            converted[dst_prefix + "self_attn.final_linear.weight"] = _get(
                state_dict, src_prefix + "layer.0.SelfAttention.o.weight"
            ).to(dtype)
            converted[dst_prefix + "self_attn.relative_attention_bias.weight"] = _get(
                state_dict, "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
            ).to(dtype)
            converted[dst_prefix + "input_layernorm.weight"] = _get(
                state_dict, src_prefix + "layer.0.layer_norm.weight"
            ).to(dtype)
            converted[dst_prefix + "precontext_layernorm.weight"] = _get(
                state_dict, src_prefix + "layer.1.layer_norm.weight"
            ).to(dtype)
            converted[dst_prefix + "context_attn.linear_query.weight"] = (
                _get(state_dict, src_prefix + "layer.1.EncDecAttention.q.weight") / (dimperhead**-0.5)
            ).to(dtype)
            converted[dst_prefix + "context_attn.linear_keys.weight"] = _get(
                state_dict, src_prefix + "layer.1.EncDecAttention.k.weight"
            ).to(dtype)
            converted[dst_prefix + "context_attn.linear_values.weight"] = _get(
                state_dict, src_prefix + "layer.1.EncDecAttention.v.weight"
            ).to(dtype)
            converted[dst_prefix + "context_attn.final_linear.weight"] = _get(
                state_dict, src_prefix + "layer.1.EncDecAttention.o.weight"
            ).to(dtype)
            converted[dst_prefix + "post_attention_layernorm.weight"] = _get(
                state_dict, src_prefix + "layer.2.layer_norm.weight"
            ).to(dtype)
            converted[dst_prefix + "mlp.gate_up_proj.weight"] = _get(
                state_dict, src_prefix + "layer.2.DenseReluDense.wi_0.weight"
            ).to(dtype)
            converted[dst_prefix + "mlp.down_proj.weight"] = _get(
                state_dict, src_prefix + "layer.2.DenseReluDense.wo.weight"
            ).to(dtype)
            converted[dst_prefix + "mlp.up_proj.weight"] = _get(
                state_dict, src_prefix + "layer.2.DenseReluDense.wi_1.weight"
            ).to(dtype)

    return converted


def _infer_metricx_metadata(model_name, qe):
    lowered = model_name.lower()
    version = "24" if "metricx-24" in lowered else "23"
    input_mode = "qe" if qe or "metricx-23-qe" in lowered else "reference"
    supported_input_modes = ["reference", "qe"] if version == "24" else [input_mode]
    max_length = 1536 if version == "24" else 1024
    return version, input_mode, supported_input_modes, max_length


def _default_tokenizer(model_name):
    lowered = model_name.lower()
    if "xxl" in lowered:
        return "google/mt5-xxl"
    if "xl" in lowered:
        return "google/mt5-xl"
    return "google/mt5-large"


def _activation_name(params):
    if params.get("dense_act_fn") == "gelu_new":
        return "gated-gelu-tanh"
    return params.get("feed_forward_proj", "gated-gelu")


@register_bin(name="MetricX")
class MetricXConverter(BaseBin):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--model", type=str, required=True, help="MetricX HF model id")
        parser.add_argument("--output", type=str, required=False, help="Output directory for converted model")
        parser.add_argument("--token", type=str, default=None, help="Hugging Face token for gated models")
        parser.add_argument(
            "--tokenizer",
            type=str,
            default=None,
            help="Tokenizer HF model id. Defaults to matching google/mt5-{large,xl,xxl}.",
        )
        parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
        parser.add_argument("--qe", action="store_true", help="Convert metadata for QE/reference-free scoring")

    @classmethod
    def run(cls, args):
        model_root = os.environ.get("EOLE_MODEL_DIR")
        default_output = os.path.join(model_root, "metricx", _slug(args.model)) if model_root else _slug(args.model)
        output_dir = args.output or default_output
        os.makedirs(output_dir, exist_ok=True)

        snapshot_dir = snapshot_download(repo_id=args.model, token=args.token)
        tokenizer_name = args.tokenizer or _default_tokenizer(args.model)
        tokenizer_dir = snapshot_download(repo_id=tokenizer_name, token=args.token)
        with open(os.path.join(snapshot_dir, "config.json"), encoding="utf-8") as f:
            params = json.load(f)

        version, input_mode, supported_input_modes, max_length = _infer_metricx_metadata(args.model, args.qe)
        dtype = _dtype(args.dtype)
        state_dict = _load_state_dict(snapshot_dir)
        converted = _convert_state_dict(state_dict, params, dtype)
        converted = {key: value.contiguous().clone() for key, value in converted.items()}
        save_file(converted, os.path.join(output_dir, "model.00.safetensors"))

        _copy_tokenizer_assets(tokenizer_dir, output_dir)
        _build_vocab(tokenizer_dir, output_dir)

        position_encoding = {
            "position_encoding_type": "Relative",
            "n_positions": params["relative_attention_max_distance"],
        }
        config = TrainConfig(
            data=None,
            skip_empty_level="silent",
            save_data=None,
            n_sample=0,
            src_vocab=None,
            tgt_vocab=None,
            share_vocab=True,
            src_vocab_size=params["vocab_size"],
            tgt_vocab_size=params["vocab_size"],
            vocab_size_multiple=8,
            decoder_start_token="<pad>",
            transforms=[],
            model=TransformerEncoderDecoderScorerModelConfig(
                layers=params["num_decoder_layers"],
                hidden_size=params["d_model"],
                heads=params["num_heads"],
                transformer_ff=params["d_ff"],
                embeddings=EmbeddingsConfig(
                    src_word_vec_size=params["d_model"],
                    tgt_word_vec_size=params["d_model"],
                    **position_encoding,
                ),
                layer_norm="rms",
                norm_eps=params.get("layer_norm_epsilon", 1e-6),
                mlp_activation_fn=_activation_name(params),
                relative_positions_buckets=params["relative_attention_num_buckets"],
                parallel_residual=False,
                add_qkvbias=False,
                add_ffnbias=False,
                generator_bias=False,
                share_embeddings=True,
                scoring_type="metricx",
                input_mode=input_mode,
                default_input_mode=input_mode,
                supported_input_modes=supported_input_modes,
                input_templates=metricx_input_templates(version),
                requires_reference=input_mode == "reference",
                max_length=max_length,
                score_position=0,
                decoder_input_length=2,
                output_scale=bool(params.get("tie_word_embeddings", False)),
                encoder={"layers": params["num_layers"]},
                decoder={"layers": params["num_decoder_layers"]},
            ),
            training=TrainingConfig(compute_dtype=args.dtype, batch_size=1, valid_batch_size=1),
        )
        with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(recursive_model_fields_set(config), f, indent=2, ensure_ascii=False)

        print(f"Converted MetricX model written to: {output_dir}")
