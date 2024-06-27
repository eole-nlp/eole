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


@register_bin(name="mpt")
class MPTConverter(BaseBin):
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
        print("Loading the full model in Cpu Ram ...")
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            torch_dtype=torch.float16,
            device_map={"": "cpu"},
            trust_remote_code=True,
        )
        checkpoint = model.state_dict()

        params_json = os.path.join(args.model_dir, "config.json")
        with open(params_json, encoding="utf-8") as fparam:
            params = json.load(fparam)

        decoder_layers = params["n_layers"]
        src_word_vec_size = params["d_model"]
        tgt_word_vec_size = params["d_model"]
        hidden_size = params["d_model"]
        heads = params["n_heads"]
        vocab_size = params["vocab_size"]
        transformer_ff = params["expansion_ratio"] * params["d_model"]

        for shard in range(args.nshards):

            print("starting output shard: %d/%d" % (shard + 1, args.nshards))
            eole_safetensor = {}

            if shard == 0:
                eole_safetensor["tgt_emb.embeddings.weight"] = checkpoint[
                    "transformer.wte.weight"
                ]
                eole_safetensor["decoder.layer_norm.weight"] = checkpoint[
                    "transformer.norm_f.weight"
                ]
                eole_safetensor["decoder.layer_norm.bias"] = torch.zeros(
                    eole_safetensor["decoder.layer_norm.weight"].size(0),
                    dtype=torch.float16,
                )

                eole_safetensor["generator.weight"] = checkpoint[
                    "transformer.wte.weight"
                ].clone()
                eole_safetensor["generator.bias"] = torch.zeros(
                    eole_safetensor["generator.weight"].size(0), dtype=torch.float16
                )

            for i in range(
                -(decoder_layers // -args.nshards) * shard,
                min(-(decoder_layers // -args.nshards) * (shard + 1), decoder_layers),
                1,
            ):
                eole_safetensor[
                    "decoder.transformer_layers."
                    + str(i)
                    + ".self_attn.linear_query.weight"
                ] = checkpoint["transformer.blocks." + str(i) + ".attn.Wqkv.weight"][
                    :hidden_size, :
                ].clone()
                eole_safetensor[
                    "decoder.transformer_layers."
                    + str(i)
                    + ".self_attn.linear_keys.weight"
                ] = checkpoint["transformer.blocks." + str(i) + ".attn.Wqkv.weight"][
                    hidden_size : (hidden_size * 2), :
                ].clone()
                eole_safetensor[
                    "decoder.transformer_layers."
                    + str(i)
                    + ".self_attn.linear_values.weight"
                ] = checkpoint["transformer.blocks." + str(i) + ".attn.Wqkv.weight"][
                    (hidden_size * 2) :, :
                ].clone()

                eole_safetensor[
                    "decoder.transformer_layers."
                    + str(i)
                    + ".self_attn.final_linear.weight"
                ] = checkpoint["transformer.blocks." + str(i) + ".attn.out_proj.weight"]

                eole_safetensor[
                    "decoder.transformer_layers." + str(i) + ".layer_norm_1.weight"
                ] = checkpoint["transformer.blocks." + str(i) + ".norm_1.weight"]
                eole_safetensor[
                    "decoder.transformer_layers." + str(i) + ".layer_norm_1.bias"
                ] = torch.zeros(
                    eole_safetensor[
                        "decoder.transformer_layers." + str(i) + ".layer_norm_1.weight"
                    ].size(0),
                    dtype=torch.float16,
                )

                eole_safetensor[
                    "decoder.transformer_layers." + str(i) + ".feed_forward.w_1.weight"
                ] = checkpoint["transformer.blocks." + str(i) + ".ffn.up_proj.weight"]

                eole_safetensor[
                    "decoder.transformer_layers." + str(i) + ".feed_forward.w_2.weight"
                ] = checkpoint["transformer.blocks." + str(i) + ".ffn.down_proj.weight"]

                eole_safetensor[
                    "decoder.transformer_layers."
                    + str(i)
                    + ".feed_forward.layer_norm.weight"
                ] = checkpoint["transformer.blocks." + str(i) + ".norm_2.weight"]
                eole_safetensor[
                    "decoder.transformer_layers."
                    + str(i)
                    + ".feed_forward.layer_norm.bias"
                ] = torch.zeros(
                    eole_safetensor[
                        "decoder.transformer_layers."
                        + str(i)
                        + ".feed_forward.layer_norm.weight"
                    ].size(0),
                    dtype=torch.float16,
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
                layer_norm="standard",
                pos_ffn_activation_fn="gelu",
                self_attn_type="scaled-dot",
                max_relative_positions=-2,
                parallel_residual=False,
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
