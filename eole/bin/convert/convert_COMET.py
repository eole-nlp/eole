#!/usr/bin/env python
import json
import os
import shutil
from types import SimpleNamespace

import torch
import yaml
from huggingface_hub import snapshot_download
from safetensors.torch import save_file

from eole.bin import BaseBin, register_bin
from eole.bin.convert.convert_HF import check_sentencepiece_tokenizer, check_special_tokens, save_vocab


SUPPORTED_ENCODERS = {"XLMRobertaForMaskedLM", "XLMRobertaXLForMaskedLM"}


def _slug(name):
    return name.replace("/", "--")


def _validate_class_identifier(class_identifier):
    if class_identifier in ["regression_metric", "referenceless_regression_metric", "unified_metric"]:
        return
    raise ValueError(f"Unsupported COMET class_identifier: {class_identifier}")


def _copy_tokenizer_assets(src_dir, output_dir):
    for filename in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "sentencepiece.bpe.model",
        "sentencepiece.model",
    ]:
        src = os.path.join(src_dir, filename)
        dst = os.path.join(output_dir, filename)
        if os.path.exists(src):
            shutil.copyfile(src, dst)


def _build_vocab(encoder_snapshot, output_dir, token):
    tokenizer_model = None
    for filename in ["sentencepiece.bpe.model", "sentencepiece.model"]:
        path = os.path.join(encoder_snapshot, filename)
        if os.path.exists(path):
            tokenizer_model = path
            break
    tokenizer_json_path = os.path.join(encoder_snapshot, "tokenizer.json")
    tokenizer_json = None
    if os.path.exists(tokenizer_json_path):
        with open(tokenizer_json_path, encoding="utf-8") as f:
            tokenizer_json = json.load(f)
    special_tokens_path = os.path.join(encoder_snapshot, "special_tokens_map.json")
    special_tokens = None
    if os.path.exists(special_tokens_path):
        with open(special_tokens_path, encoding="utf-8") as f:
            special_tokens = json.load(f)

    if tokenizer_model is not None:
        hf = SimpleNamespace(tokenizer_model=tokenizer_model, tokenizer=tokenizer_json, tokenizer_json=tokenizer_json)
        src_vocab, _ = check_sentencepiece_tokenizer(hf)
    else:
        raise FileNotFoundError(f"No sentencepiece model found in {encoder_snapshot}")

    if special_tokens is not None:
        hf_for_specials = SimpleNamespace(special_tokens=special_tokens, tokenizer_config=None)
        vocabs = check_special_tokens(hf_for_specials)
    else:
        vocabs = {"specials": {"bos_token": "<s>", "pad_token": "<pad>", "eos_token": "</s>", "unk_token": "<unk>"}}
    save_vocab(vocabs, src_vocab, output_dir)


def _build_model_config(args_model, hparams, encoder_config):
    encoder_architecture = encoder_config["architectures"][0]
    if encoder_architecture not in SUPPORTED_ENCODERS:
        raise ValueError(
            f"Unsupported COMET encoder architecture: {encoder_architecture}. "
            "Native COMET conversion currently supports only XLM-R/XLM-R-XL."
        )

    layer_transformation = hparams.get("layer_transformation", "softmax")
    if layer_transformation == "sparsemax_patch":
        layer_transformation = "softmax"

    input_segments = hparams.get("input_segments", ["mt", "src", "ref"])
    class_identifier = hparams["class_identifier"]
    requires_reference = class_identifier == "regression_metric" or (
        class_identifier == "unified_metric" and "ref" in input_segments
    )

    return {
        "architecture": "transformer_encoder_scorer",
        "scoring_type": "comet",
        "class_identifier": class_identifier,
        "requires_reference": requires_reference,
        "pretrained_model": args_model,
        "pool": hparams.get("pool", "avg"),
        "layer": str(hparams.get("sent_layer", hparams.get("layer", "mix"))),
        "layer_transformation": layer_transformation,
        "layer_norm": bool(hparams.get("layer_norm", False)),
        "input_segments": input_segments,
        "hidden_sizes": hparams.get("hidden_sizes", [3072, 1024]),
        "activations": hparams.get("activations", "Tanh"),
        "final_activation": hparams.get("final_activation"),
        "dropout": float(hparams.get("dropout", 0.1)),
        "encoder_architecture": encoder_architecture,
        "embeddings": {
            "embedding_type": "roberta",
            "src_word_vec_size": encoder_config["hidden_size"],
            "tgt_word_vec_size": encoder_config["hidden_size"],
            "word_vec_size": encoder_config["hidden_size"],
            "position_encoding_type": "Learned",
            "n_positions": encoder_config.get("max_position_embeddings", 514),
            "position_shift": 2 if encoder_architecture == "XLMRobertaXLForMaskedLM" else 0,
            "embedding_layer_norm": encoder_architecture != "XLMRobertaXLForMaskedLM",
            "token_type_vocab_size": 1,
        },
        "encoder": {
            "encoder_type": "transformer",
            "layers": encoder_config["num_hidden_layers"],
            "hidden_size": encoder_config["hidden_size"],
            "heads": encoder_config["num_attention_heads"],
            "transformer_ff": encoder_config["intermediate_size"],
            "mlp_activation_fn": "gelu",
            "layer_norm": "standard",
            "norm_eps": encoder_config.get("layer_norm_eps", 1e-5),
            "add_qkvbias": True,
            "add_key_bias": True,
            "add_final_linear_bias": True,
            "add_ffnbias": True,
            "position_encoding_type": "Learned",
            "n_positions": encoder_config.get("max_position_embeddings", 514),
            "encoder_layer_style": "standard" if encoder_architecture == "XLMRobertaXLForMaskedLM" else "postnorm",
            "final_encoder_layer_norm": encoder_architecture == "XLMRobertaXLForMaskedLM",
        },
    }


def _convert_state_dict(state_dict, model_config):
    encoder_cfg = model_config["encoder"]
    embeddings_cfg = model_config["embeddings"]
    converted = {
        "encoder.embeddings.word_embeddings.weight": state_dict["encoder.model.embeddings.word_embeddings.weight"],
        "encoder.embeddings.position_embeddings.weight": state_dict[
            "encoder.model.embeddings.position_embeddings.weight"
        ],
        "encoder.embeddings.token_type_embeddings.weight": state_dict[
            "encoder.model.embeddings.token_type_embeddings.weight"
        ],
    }
    if embeddings_cfg.get("embedding_layer_norm", False):
        converted["encoder.embeddings.layer_norm.weight"] = state_dict["encoder.model.embeddings.LayerNorm.weight"]
        converted["encoder.embeddings.layer_norm.bias"] = state_dict["encoder.model.embeddings.LayerNorm.bias"]
    if encoder_cfg.get("final_encoder_layer_norm", False):
        converted["encoder.layer_norm.weight"] = state_dict["encoder.model.encoder.LayerNorm.weight"]
        converted["encoder.layer_norm.bias"] = state_dict["encoder.model.encoder.LayerNorm.bias"]

    layers = encoder_cfg["layers"]
    is_xl = model_config["encoder_architecture"] == "XLMRobertaXLForMaskedLM"
    for i in range(layers):
        src_prefix = f"encoder.model.encoder.layer.{i}."
        dst_prefix = f"encoder.transformer_layers.{i}."
        converted[dst_prefix + "self_attn.linear_query.weight"] = state_dict[src_prefix + "attention.self.query.weight"]
        converted[dst_prefix + "self_attn.linear_query.bias"] = state_dict[src_prefix + "attention.self.query.bias"]
        converted[dst_prefix + "self_attn.linear_keys.weight"] = state_dict[src_prefix + "attention.self.key.weight"]
        converted[dst_prefix + "self_attn.linear_keys.bias"] = state_dict[src_prefix + "attention.self.key.bias"]
        converted[dst_prefix + "self_attn.linear_values.weight"] = state_dict[
            src_prefix + "attention.self.value.weight"
        ]
        converted[dst_prefix + "self_attn.linear_values.bias"] = state_dict[src_prefix + "attention.self.value.bias"]
        converted[dst_prefix + "self_attn.final_linear.weight"] = state_dict[
            src_prefix + "attention.output.dense.weight"
        ]
        converted[dst_prefix + "self_attn.final_linear.bias"] = state_dict[src_prefix + "attention.output.dense.bias"]
        if is_xl:
            converted[dst_prefix + "input_layernorm.weight"] = state_dict[
                src_prefix + "attention.self_attn_layer_norm.weight"
            ]
            converted[dst_prefix + "input_layernorm.bias"] = state_dict[
                src_prefix + "attention.self_attn_layer_norm.bias"
            ]
            converted[dst_prefix + "post_attention_layernorm.weight"] = state_dict[src_prefix + "LayerNorm.weight"]
            converted[dst_prefix + "post_attention_layernorm.bias"] = state_dict[src_prefix + "LayerNorm.bias"]
        else:
            converted[dst_prefix + "attention_layer_norm.weight"] = state_dict[
                src_prefix + "attention.output.LayerNorm.weight"
            ]
            converted[dst_prefix + "attention_layer_norm.bias"] = state_dict[
                src_prefix + "attention.output.LayerNorm.bias"
            ]
            converted[dst_prefix + "output_layer_norm.weight"] = state_dict[src_prefix + "output.LayerNorm.weight"]
            converted[dst_prefix + "output_layer_norm.bias"] = state_dict[src_prefix + "output.LayerNorm.bias"]
        converted[dst_prefix + "mlp.gate_up_proj.weight"] = state_dict[src_prefix + "intermediate.dense.weight"]
        converted[dst_prefix + "mlp.gate_up_proj.bias"] = state_dict[src_prefix + "intermediate.dense.bias"]
        converted[dst_prefix + "mlp.down_proj.weight"] = state_dict[src_prefix + "output.dense.weight"]
        converted[dst_prefix + "mlp.down_proj.bias"] = state_dict[src_prefix + "output.dense.bias"]

    for key, value in state_dict.items():
        if key.startswith("estimator."):
            converted[key] = value
        elif (
            key.startswith("layerwise_attention.")
            and not key.endswith("dropout_mask")
            and not key.endswith("dropout_fill")
        ):
            converted[key] = value
    return converted


@register_bin(name="COMET")
class CometConverter(BaseBin):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--model", type=str, required=True, help="COMET HF model id")
        parser.add_argument("--output", type=str, required=False, help="Output directory for converted model")
        parser.add_argument("--token", type=str, default=None, help="Hugging Face token for gated models")

    @classmethod
    def run(cls, args):
        model_root = os.environ.get("EOLE_MODEL_DIR")
        default_output = os.path.join(model_root, "comet", _slug(args.model)) if model_root else _slug(args.model)
        output_dir = args.output or default_output
        os.makedirs(output_dir, exist_ok=True)

        snapshot_dir = snapshot_download(repo_id=args.model, token=args.token)
        model_ckpt = os.path.join(snapshot_dir, "checkpoints", "model.ckpt")
        hparams_file = os.path.join(snapshot_dir, "hparams.yaml")
        if not os.path.exists(model_ckpt) or not os.path.exists(hparams_file):
            raise FileNotFoundError("Could not find COMET checkpoint files (hparams.yaml and checkpoints/model.ckpt).")

        with open(hparams_file, encoding="utf-8") as f:
            hparams = yaml.safe_load(f)
        _validate_class_identifier(hparams["class_identifier"])

        encoder_snapshot = snapshot_download(repo_id=hparams["pretrained_model"], token=args.token)
        with open(os.path.join(encoder_snapshot, "config.json"), encoding="utf-8") as f:
            encoder_config = json.load(f)

        model_config = _build_model_config(args.model, hparams, encoder_config)
        train_config = {
            "src_vocab": "dummy",
            "tgt_vocab": "dummy",
            "data": {},
            "share_vocab": True,
            "training": {"compute_dtype": "fp32"},
            "model": model_config,
        }

        ckpt = torch.load(model_ckpt, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        converted = _convert_state_dict(state_dict, model_config)
        if not any(key.startswith("estimator.") for key in converted):
            raise ValueError("Converted checkpoint is missing estimator.* weights.")

        for obsolete in ["model.ckpt", "eole_comet_config.json"]:
            path = os.path.join(output_dir, obsolete)
            if os.path.exists(path):
                os.remove(path)

        _copy_tokenizer_assets(encoder_snapshot, output_dir)
        _build_vocab(encoder_snapshot, output_dir, args.token)
        save_file(converted, os.path.join(output_dir, "model.00.safetensors"))

        with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(train_config, f, indent=2)

        print(f"Converted COMET model written to: {output_dir}")
