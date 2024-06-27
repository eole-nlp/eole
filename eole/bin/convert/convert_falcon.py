#!/usr/bin/env python
import json
import torch
import pyonmttok
from eole.inputters.inputter import vocabs_to_dict
import os
from safetensors.torch import save_file

from eole.config.models import (
    EmbeddingsConfig,
    TransformerLMModelConfig,
)
from eole.config.run import TrainConfig
from eole.config.training import TrainingConfig
from eole.config import recursive_model_fields_set
from eole.bin import BaseBin, register_bin


@register_bin(name="falcon")
class FalconConverter(BaseBin):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--model_dir",
            type=str,
            required=True,
            help="""Path to the model directory""",
        )
        parser.add_argument(
            "--vocab_file",
            type=str,
            required=True,
            help="""Path to the tokenizer model""",
        )
        parser.add_argument(
            "--output", type=str, required=True, help="""Path to the model directory"""
        )
        parser.add_argument(
            "--nshards", type=int, default=1, help="""Path to the model directory"""
        )

    @classmethod
    def run(cls, args):

        params_json = os.path.join(args.model_dir, "config.json")
        with open(params_json, encoding="utf-8") as fparam:
            params = json.load(fparam)

        decoder_layers = params["num_hidden_layers"]
        src_word_vec_size = params["hidden_size"]
        tgt_word_vec_size = params["hidden_size"]
        hidden_size = params["hidden_size"]
        heads = params["num_attention_heads"]
        if "n_head_kv" in params.keys():
            num_kv = params["n_head_kv"]
        else:
            num_kv = 1
        vocab_size = params["vocab_size"]
        transformer_ff = params["hidden_size"] * 4
        shared_layer = num_kv == 1

        ckpt_split_json = os.path.join(args.model_dir, "pytorch_model.bin.index.json")
        with open(ckpt_split_json, encoding="utf-8") as fweights:
            wmap = json.load(fweights)

        for shard in range(args.nshards):

            print("starting output shard: %d/%d" % (shard + 1, args.nshards))
            eole_safetensor = {}

            if shard == 0:
                ckpt = wmap["weight_map"]["transformer.word_embeddings.weight"]
                checkpoint = torch.load(
                    os.path.join(args.model_dir, ckpt), map_location=torch.device("cpu")
                )
                eole_safetensor["tgt_emb.embeddings.weight"] = checkpoint[
                    "transformer.word_embeddings.weight"
                ].to(torch.float16)

                ckpt = wmap["weight_map"]["transformer.ln_f.weight"]
                checkpoint = torch.load(
                    os.path.join(args.model_dir, ckpt), map_location=torch.device("cpu")
                )
                eole_safetensor["decoder.layer_norm.weight"] = checkpoint[
                    "transformer.ln_f.weight"
                ].to(torch.float16)
                eole_safetensor["decoder.layer_norm.bias"] = checkpoint[
                    "transformer.ln_f.bias"
                ].to(torch.float16)

                ckpt = wmap["weight_map"]["lm_head.weight"]
                checkpoint = torch.load(
                    os.path.join(args.model_dir, ckpt), map_location=torch.device("cpu")
                )

                eole_safetensor["generator.weight"] = checkpoint["lm_head.weight"].to(
                    torch.float16
                )
                eole_safetensor["generator.bias"] = torch.zeros(
                    eole_safetensor["generator.weight"].size(0), dtype=torch.float16
                )

            weightmap = wmap["weight_map"]
            ckpt_list = []
            for key in weightmap.keys():
                if (
                    key.startswith("transformer.h.")
                    and int(key.split(".")[2])
                    in range(
                        -(decoder_layers // -args.nshards) * shard,
                        min(
                            -(decoder_layers // -args.nshards) * (shard + 1),
                            decoder_layers,
                        ),
                        1,
                    )
                    and weightmap[key] not in ckpt_list
                ):
                    ckpt_list.append(weightmap[key])

            for ckpt in ckpt_list:
                print("Loading %s" % (os.path.join(args.model_dir, ckpt)))
                checkpoint = torch.load(
                    os.path.join(args.model_dir, ckpt), map_location=torch.device("cpu")
                )

                for i in range(
                    -(decoder_layers // -args.nshards) * shard,
                    min(
                        -(decoder_layers // -args.nshards) * (shard + 1), decoder_layers
                    ),
                    1,
                ):

                    # Falcon stores QKV in one single tensor but it is not simply piled up Q+K+V
                    # it is heads interleaved to we need to slice first
                    # also it uses the HF rotary so we need to permute Q and K interleave

                    if (
                        "transformer.h."
                        + str(i)
                        + ".self_attention.query_key_value.weight"
                        in checkpoint.keys()
                    ):
                        qkv_W = (
                            checkpoint[
                                "transformer.h."
                                + str(i)
                                + ".self_attention.query_key_value.weight"
                            ]
                            .view(
                                -1,
                                heads // num_kv + 2,
                                hidden_size // heads,
                                hidden_size,
                            )  # shape 40B [8, 18, 64, 8192] ou 7B [1, 73, 64, 4544]
                            .to(torch.float16)
                        )

                        eole_safetensor[
                            "decoder.transformer_layers."
                            + str(i)
                            + ".self_attn.linear_query.weight"
                        ] = (
                            qkv_W[:, :-2]  # [8, 16, 64, 8192] ou [1, 71, 64, 4544]
                            .reshape(
                                heads, 2, hidden_size // heads // 2, hidden_size
                            )  # [128, 2, 32, 8192] ou [71, 2, 32, 4544]
                            .transpose(
                                1, 2
                            )  # invert 1,2 for rotary "llama original way"
                            .reshape(
                                hidden_size, hidden_size
                            )  # [8192, 8192] ou [4544, 4544]
                        )

                        eole_safetensor[
                            "decoder.transformer_layers."
                            + str(i)
                            + ".self_attn.linear_keys.weight"
                        ] = (
                            qkv_W[:, [-2]]  # [8, 1, 64, 8192] ou [1, 1, 64, 4544]
                            .reshape(
                                num_kv, 2, hidden_size // heads // 2, hidden_size
                            )  # [8, 2, 32, 8192] ou [1, 2, 32, 4544]
                            .transpose(1, 2)
                            .reshape(
                                hidden_size // heads * num_kv, hidden_size
                            )  # [512, 8192] ou [64, 4544]
                        )

                        eole_safetensor[
                            "decoder.transformer_layers."
                            + str(i)
                            + ".self_attn.linear_values.weight"
                        ] = qkv_W[:, [-1]].reshape(
                            hidden_size // heads * num_kv, hidden_size
                        )
                    if (
                        "transformer.h." + str(i) + ".self_attention.dense.weight"
                        in checkpoint.keys()
                    ):
                        eole_safetensor[
                            "decoder.transformer_layers."
                            + str(i)
                            + ".self_attn.final_linear.weight"
                        ] = checkpoint[
                            "transformer.h." + str(i) + ".self_attention.dense.weight"
                        ].to(
                            torch.float16
                        )
                    if shared_layer:
                        if (
                            "transformer.h." + str(i) + ".input_layernorm.weight"
                            in checkpoint.keys()
                        ):
                            eole_safetensor[
                                "decoder.transformer_layers."
                                + str(i)
                                + ".layer_norm_1.weight"
                            ] = checkpoint[
                                "transformer.h." + str(i) + ".input_layernorm.weight"
                            ].to(
                                torch.float16
                            )
                        if (
                            "transformer.h." + str(i) + ".input_layernorm.bias"
                            in checkpoint.keys()
                        ):
                            eole_safetensor[
                                "decoder.transformer_layers."
                                + str(i)
                                + ".layer_norm_1.bias"
                            ] = checkpoint[
                                "transformer.h." + str(i) + ".input_layernorm.bias"
                            ].to(
                                torch.float16
                            )
                    else:
                        if (
                            "transformer.h." + str(i) + ".ln_attn.weight"
                            in checkpoint.keys()
                        ):
                            eole_safetensor[
                                "decoder.transformer_layers."
                                + str(i)
                                + ".layer_norm_1.weight"
                            ] = checkpoint[
                                "transformer.h." + str(i) + ".ln_attn.weight"
                            ].to(
                                torch.float16
                            )
                        if (
                            "transformer.h." + str(i) + ".ln_attn.weight"
                            in checkpoint.keys()
                        ):
                            eole_safetensor[
                                "decoder.transformer_layers."
                                + str(i)
                                + ".layer_norm_1.bias"
                            ] = checkpoint[
                                "transformer.h." + str(i) + ".ln_attn.bias"
                            ].to(
                                torch.float16
                            )
                        if (
                            "transformer.h." + str(i) + ".ln_mlp.weight"
                            in checkpoint.keys()
                        ):
                            eole_safetensor[
                                "decoder.transformer_layers."
                                + str(i)
                                + ".layer_norm_res.weight"
                            ] = checkpoint[
                                "transformer.h." + str(i) + ".ln_mlp.weight"
                            ].to(
                                torch.float16
                            )
                        if (
                            "transformer.h." + str(i) + ".ln_mlp.bias"
                            in checkpoint.keys()
                        ):
                            eole_safetensor[
                                "decoder.transformer_layers."
                                + str(i)
                                + ".layer_norm_res.bias"
                            ] = checkpoint[
                                "transformer.h." + str(i) + ".ln_mlp.bias"
                            ].to(
                                torch.float16
                            )

                    if (
                        "transformer.h." + str(i) + ".mlp.dense_h_to_4h.weight"
                        in checkpoint.keys()
                    ):
                        eole_safetensor[
                            "decoder.transformer_layers."
                            + str(i)
                            + ".feed_forward.w_1.weight"
                        ] = checkpoint[
                            "transformer.h." + str(i) + ".mlp.dense_h_to_4h.weight"
                        ].to(
                            torch.float16
                        )
                    if (
                        "transformer.h." + str(i) + ".mlp.dense_4h_to_h.weight"
                        in checkpoint.keys()
                    ):
                        eole_safetensor[
                            "decoder.transformer_layers."
                            + str(i)
                            + ".feed_forward.w_2.weight"
                        ] = checkpoint[
                            "transformer.h." + str(i) + ".mlp.dense_4h_to_h.weight"
                        ].to(
                            torch.float16
                        )

            if shard == 0:
                transformer_ff = eole_safetensor[
                    "decoder.transformer_layers.0.feed_forward.w_1.weight"
                ].size(0)
                vocab_size = eole_safetensor["generator.weight"].size(0)
            print("Saving output model shard: %d" % shard)
            save_file(
                eole_safetensor,
                os.path.join(args.output, "model.{:02d}.safetensors".format(shard)),
            )

        vocabs = {}
        with open(args.vocab_file, "r", encoding="utf-8") as vocab:
            src_vocab = pyonmttok.build_vocab_from_lines(vocab)

        vocabs["src"] = src_vocab
        vocabs["tgt"] = src_vocab
        vocabs["decoder_start_token"] = "</s>"

        vocab_dict = vocabs_to_dict(vocabs)
        with open(os.path.join(args.output, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(vocab_dict, f, indent=2, ensure_ascii=False)

        with open(
            os.path.join(args.output, "vocab.txt"), "w", encoding="utf-8"
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
            transforms=["filtertoolong"],
            transforms_configs={
                "filtertoolong": {"src_seq_length": 512, "tgt_seq_length": 512}
            },
            model=TransformerLMModelConfig(
                layers=decoder_layers,
                hidden_size=hidden_size,
                heads=heads,
                transformer_ff=transformer_ff,
                embeddings=EmbeddingsConfig(
                    src_word_vec_size=src_word_vec_size,
                    tgt_word_vec_size=tgt_word_vec_size,
                ),
                # src_word_vec_size=src_word_vec_size,
                # tgt_word_vec_size=tgt_word_vec_size,
                model_type="text",
                pos_ffn_activation_fn="gelu",
                self_attn_type="scaled-dot",  # not sure if scaled-dot-flash is fine
                max_relative_positions=-1,
                num_kv=num_kv,
                parallel_residual=True,
                shared_layer_norm=shared_layer,
                add_qkvbias=False,
                add_ffnbias=False,
            ),
            training=TrainingConfig(
                model_dtype="fp16",
                batch_size=896,
                batch_size_multiple=1,
                batch_type="tokens",
                normalization="tokens",
                accum_count=[32],
                accum_steps=[0],
                valid_batch_size=256,
                optim="fusedadam",
            ),
        )

        config_dict = recursive_model_fields_set(config)
        with open(os.path.join(args.output, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
