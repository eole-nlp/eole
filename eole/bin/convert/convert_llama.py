#!/usr/bin/env python
import json
import torch
import pyonmttok
from eole.inputters.inputter import vocabs_to_dict
from eole.constants import DefaultTokens
from sentencepiece import SentencePieceProcessor
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


@register_bin(name="llama_legacy")
class LlamaLegacyConverter(BaseBin):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--model_dir",
            type=str,
            required=True,
            help="""Path to the model directory""",
        )
        parser.add_argument(
            "--tokenizer_model",
            type=str,
            required=True,
            help="""Path to the tokenizer model""",
        )
        parser.add_argument(
            "--output",
            type=str,
            required=True,
            help="""Path and filename of the saved model""",
        )
        parser.add_argument(
            "--nshards", type=int, default=1, help="""Path to the model directory"""
        )

    @classmethod
    def run(cls, args):
        params_json = os.path.join(args.model_dir, "params.json")
        with open(params_json, encoding="utf-8") as fparam:
            params = json.load(fparam)

        decoder_layers = params["n_layers"]
        src_word_vec_size = params["dim"]
        tgt_word_vec_size = params["dim"]
        hidden_size = params["dim"]
        heads = params["n_heads"]
        norm_eps = params["norm_eps"]
        if "n_kv_heads" in params.keys():
            num_kv = params["n_kv_heads"]
        else:
            num_kv = 0
        if "sliding_window" in params.keys():
            sliding_window = params["sliding_window"]
        else:
            sliding_window = 0

        for shard in range(args.nshards):

            print("starting output shard: %d/%d" % (shard + 1, args.nshards))
            eole_safetensor = {}
            print("Reading input shard: ", 0)
            checkpoint = torch.load(
                os.path.join(args.model_dir, "consolidated.00.pth"),
                map_location=torch.device("cpu"),
            )

            if shard == 0:
                eole_safetensor["tgt_emb.embeddings.weight"] = checkpoint[
                    "tok_embeddings.weight"
                ]
                eole_safetensor["decoder.layer_norm.weight"] = checkpoint["norm.weight"]

                eole_safetensor["generator.weight"] = checkpoint["output.weight"]
                eole_safetensor["generator.bias"] = torch.zeros(
                    eole_safetensor["generator.weight"].size(0),
                    dtype=checkpoint["output.weight"].dtype,
                )

            for i in range(
                -(decoder_layers // -args.nshards) * shard,
                min(-(decoder_layers // -args.nshards) * (shard + 1), decoder_layers),
                1,
            ):
                eole_safetensor[
                    "decoder.transformer_layers."
                    + str(i)
                    + ".self_attn.linear_keys.weight"
                ] = checkpoint["layers." + str(i) + ".attention.wk.weight"]

                eole_safetensor[
                    "decoder.transformer_layers."
                    + str(i)
                    + ".self_attn.linear_values.weight"
                ] = checkpoint["layers." + str(i) + ".attention.wv.weight"]

                eole_safetensor[
                    "decoder.transformer_layers."
                    + str(i)
                    + ".self_attn.linear_query.weight"
                ] = checkpoint["layers." + str(i) + ".attention.wq.weight"]

                eole_safetensor[
                    "decoder.transformer_layers."
                    + str(i)
                    + ".self_attn.final_linear.weight"
                ] = checkpoint["layers." + str(i) + ".attention.wo.weight"]

                eole_safetensor[
                    "decoder.transformer_layers." + str(i) + ".layer_norm_1.weight"
                ] = checkpoint["layers." + str(i) + ".attention_norm.weight"].clone()

                eole_safetensor[
                    "decoder.transformer_layers." + str(i) + ".feed_forward.w_1.weight"
                ] = checkpoint["layers." + str(i) + ".feed_forward.w1.weight"]

                eole_safetensor[
                    "decoder.transformer_layers." + str(i) + ".feed_forward.w_2.weight"
                ] = checkpoint["layers." + str(i) + ".feed_forward.w2.weight"]

                eole_safetensor[
                    "decoder.transformer_layers." + str(i) + ".feed_forward.w_3.weight"
                ] = checkpoint["layers." + str(i) + ".feed_forward.w3.weight"]

                eole_safetensor[
                    "decoder.transformer_layers."
                    + str(i)
                    + ".feed_forward.layer_norm.weight"
                ] = checkpoint["layers." + str(i) + ".ffn_norm.weight"].clone()

            input_shard = 1
            while os.path.exists(
                os.path.join(
                    args.model_dir, "consolidated.0" + str(input_shard) + ".pth"
                )
            ):
                print("Reading input shard: ", input_shard)
                checkpoint = torch.load(
                    os.path.join(
                        args.model_dir, "consolidated.0" + str(input_shard) + ".pth"
                    ),
                    map_location=torch.device("cpu"),
                )

                if shard == 0:
                    eole_safetensor["tgt_emb.embeddings.weight"] = torch.cat(
                        (
                            eole_safetensor["tgt_emb.embeddings.weight"],
                            checkpoint["tok_embeddings.weight"],
                        ),
                        dim=1,
                    )
                    eole_safetensor["generator.weight"] = torch.cat(
                        (
                            eole_safetensor["generator.weight"],
                            checkpoint["output.weight"],
                        ),
                        dim=0,
                    )
                    eole_safetensor["generator.bias"] = torch.zeros(
                        eole_safetensor["generator.weight"].size(0),
                        dtype=checkpoint["output.weight"].dtype,
                    )

                for i in range(
                    -(decoder_layers // -args.nshards) * shard,
                    min(
                        -(decoder_layers // -args.nshards) * (shard + 1), decoder_layers
                    ),
                    1,
                ):
                    eole_safetensor[
                        "decoder.transformer_layers."
                        + str(i)
                        + ".self_attn.linear_keys.weight"
                    ] = torch.cat(
                        (
                            eole_safetensor[
                                "decoder.transformer_layers."
                                + str(i)
                                + ".self_attn.linear_keys.weight"
                            ],
                            checkpoint["layers." + str(i) + ".attention.wk.weight"],
                        ),
                        dim=0,
                    )

                    eole_safetensor[
                        "decoder.transformer_layers."
                        + str(i)
                        + ".self_attn.linear_values.weight"
                    ] = torch.cat(
                        (
                            eole_safetensor[
                                "decoder.transformer_layers."
                                + str(i)
                                + ".self_attn.linear_values.weight"
                            ],
                            checkpoint["layers." + str(i) + ".attention.wv.weight"],
                        ),
                        dim=0,
                    )

                    eole_safetensor[
                        "decoder.transformer_layers."
                        + str(i)
                        + ".self_attn.linear_query.weight"
                    ] = torch.cat(
                        (
                            eole_safetensor[
                                "decoder.transformer_layers."
                                + str(i)
                                + ".self_attn.linear_query.weight"
                            ],
                            checkpoint["layers." + str(i) + ".attention.wq.weight"],
                        ),
                        dim=0,
                    )

                    eole_safetensor[
                        "decoder.transformer_layers."
                        + str(i)
                        + ".self_attn.final_linear.weight"
                    ] = torch.cat(
                        (
                            eole_safetensor[
                                "decoder.transformer_layers."
                                + str(i)
                                + ".self_attn.final_linear.weight"
                            ],
                            checkpoint["layers." + str(i) + ".attention.wo.weight"],
                        ),
                        dim=1,
                    )

                    eole_safetensor[
                        "decoder.transformer_layers."
                        + str(i)
                        + ".feed_forward.w_1.weight"
                    ] = torch.cat(
                        (
                            eole_safetensor[
                                "decoder.transformer_layers."
                                + str(i)
                                + ".feed_forward.w_1.weight"
                            ],
                            checkpoint["layers." + str(i) + ".feed_forward.w1.weight"],
                        ),
                        dim=0,
                    )

                    eole_safetensor[
                        "decoder.transformer_layers."
                        + str(i)
                        + ".feed_forward.w_2.weight"
                    ] = torch.cat(
                        (
                            eole_safetensor[
                                "decoder.transformer_layers."
                                + str(i)
                                + ".feed_forward.w_2.weight"
                            ],
                            checkpoint["layers." + str(i) + ".feed_forward.w2.weight"],
                        ),
                        dim=1,
                    )

                    eole_safetensor[
                        "decoder.transformer_layers."
                        + str(i)
                        + ".feed_forward.w_3.weight"
                    ] = torch.cat(
                        (
                            eole_safetensor[
                                "decoder.transformer_layers."
                                + str(i)
                                + ".feed_forward.w_3.weight"
                            ],
                            checkpoint["layers." + str(i) + ".feed_forward.w3.weight"],
                        ),
                        dim=0,
                    )

                input_shard += 1
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

        tokenizer = Tokenizer(model_path=args.tokenizer_model)
        vocabs = {}
        vocab = tokenizer.vocab
        vocab[3] = DefaultTokens.PAD
        src_vocab = pyonmttok.build_vocab_from_tokens(
            vocab,
            maximum_size=tokenizer.n_words,
            special_tokens=["<unk>", "<s>", "</s>"],
        )
        vocabs["src"] = src_vocab
        vocabs["tgt"] = src_vocab
        vocabs["decoder_start_token"] = "<s>"

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
            decoder_start_token="<s>",
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
                layer_norm="rms",
                norm_eps=norm_eps,
                pos_ffn_activation_fn="silu",
                self_attn_type="scaled-dot",
                max_relative_positions=-1,
                rotary_interleave=True,
                rotary_theta=10000,
                rotary_dim=0,
                sliding_window=sliding_window,
                parallel_residual=False,
                add_qkvbias=False,
                add_ffnbias=False,
                num_kv=num_kv,
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
