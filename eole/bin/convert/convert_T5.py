#!/usr/bin/env python
import eole.bin.tools.sentencepiece_model_pb2 as spmodel
import json
import torch
import pyonmttok
from eole.inputters.inputter import vocabs_to_dict
from sentencepiece import SentencePieceProcessor
import os
from safetensors.torch import save_file

# from transformers import T5ForConditionalGeneration

from eole.config.models import (
    EmbeddingsConfig,
    TransformerModelConfig,
)
from eole.config.run import TrainConfig
from eole.config.training import TrainingConfig
from eole.config import recursive_model_fields_set
from eole.bin import BaseBin, register_bin


"""
Special Note to T5 models conversion.

T5 does not rescale MHE query (by dividing by square_root of dim_per_head)
query /= math.sqrt(self.dim_per_head)

So all query weights are rescaled directly in the checkpoint so that
no change is needed in the legacy MHA code.
"""


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


@register_bin(name="T5")
class T5Converter(BaseBin):
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
            "--output", type=str, required=True, help="""Path to the model directory"""
        )
        parser.add_argument(
            "--nshards", type=int, default=1, help="""Path to the model directory"""
        )

    @classmethod
    def run(cls, args):
        # load this from pytorch_model.bin directly instead
        # model = T5ForConditionalGeneration.from_pretrained(
        #     args.model_dir,
        #     torch_dtype=torch.float16,
        #     device_map={"": "cpu"},
        #     trust_remote_code=True,
        # )
        # checkpoint = model.state_dict()
        checkpoint = torch.load(
            os.path.join(args.model_dir, "pytorch_model.bin"), map_location="cpu"
        )
        checkpoint = {k: v.half() for k, v in checkpoint.items()}

        params_json = os.path.join(args.model_dir, "config.json")
        with open(params_json, encoding="utf-8") as fparam:
            params = json.load(fparam)

        decoder_layers = params["num_decoder_layers"]
        encoder_layers = params["num_layers"]
        src_word_vec_size = params["d_model"]
        tgt_word_vec_size = params["d_model"]
        hidden_size = params["d_model"]
        heads = params["num_heads"]
        vocab_size = params["vocab_size"]
        transformer_ff = params["d_ff"]
        dimperhead = hidden_size // heads

        for shard in range(args.nshards):

            print("starting output shard: %d/%d" % (shard + 1, args.nshards))
            eole_safetensor = {}

            if shard == 0:
                eole_safetensor["tgt_emb.embeddings.weight"] = checkpoint[
                    "decoder.embed_tokens.weight"
                ].to(torch.float16)
                eole_safetensor["decoder.layer_norm.weight"] = checkpoint[
                    "decoder.final_layer_norm.weight"
                ].to(torch.float16)
                eole_safetensor["src_emb.embeddings.weight"] = checkpoint[
                    "encoder.embed_tokens.weight"
                ].to(torch.float16)
                eole_safetensor["encoder.layer_norm.weight"] = checkpoint[
                    "encoder.final_layer_norm.weight"
                ].to(torch.float16)
                eole_safetensor["generator.weight"] = checkpoint["lm_head.weight"].to(
                    torch.float16
                )
                eole_safetensor["generator.bias"] = torch.zeros(
                    eole_safetensor["generator.weight"].size(0), dtype=torch.float16
                )

            for i in range(
                -(decoder_layers // -args.nshards) * shard,
                min(-(decoder_layers // -args.nshards) * (shard + 1), decoder_layers),
                1,
            ):
                eole_safetensor[
                    "encoder.transformer_layers."
                    + str(i)
                    + ".self_attn.linear_query.weight"
                ] = (
                    checkpoint[
                        "encoder.block." + str(i) + ".layer.0.SelfAttention.q.weight"
                    ]
                    / (dimperhead**-0.5)
                ).to(
                    torch.float16
                )
                eole_safetensor[
                    "decoder.transformer_layers."
                    + str(i)
                    + ".self_attn.linear_query.weight"
                ] = (
                    checkpoint[
                        "decoder.block." + str(i) + ".layer.0.SelfAttention.q.weight"
                    ]
                    / (dimperhead**-0.5)
                ).to(
                    torch.float16
                )
                eole_safetensor[
                    "encoder.transformer_layers."
                    + str(i)
                    + ".self_attn.linear_keys.weight"
                ] = checkpoint[
                    "encoder.block." + str(i) + ".layer.0.SelfAttention.k.weight"
                ].to(
                    torch.float16
                )
                eole_safetensor[
                    "decoder.transformer_layers."
                    + str(i)
                    + ".self_attn.linear_keys.weight"
                ] = checkpoint[
                    "decoder.block." + str(i) + ".layer.0.SelfAttention.k.weight"
                ].to(
                    torch.float16
                )
                eole_safetensor[
                    "encoder.transformer_layers."
                    + str(i)
                    + ".self_attn.linear_values.weight"
                ] = checkpoint[
                    "encoder.block." + str(i) + ".layer.0.SelfAttention.v.weight"
                ].to(
                    torch.float16
                )
                eole_safetensor[
                    "decoder.transformer_layers."
                    + str(i)
                    + ".self_attn.linear_values.weight"
                ] = checkpoint[
                    "decoder.block." + str(i) + ".layer.0.SelfAttention.v.weight"
                ].to(
                    torch.float16
                )

                eole_safetensor[
                    "encoder.transformer_layers."
                    + str(i)
                    + ".self_attn.final_linear.weight"
                ] = checkpoint[
                    "encoder.block." + str(i) + ".layer.0.SelfAttention.o.weight"
                ].to(
                    torch.float16
                )
                eole_safetensor[
                    "decoder.transformer_layers."
                    + str(i)
                    + ".self_attn.final_linear.weight"
                ] = checkpoint[
                    "decoder.block." + str(i) + ".layer.0.SelfAttention.o.weight"
                ].to(
                    torch.float16
                )

                eole_safetensor[
                    "encoder.transformer_layers."
                    + str(i)
                    + ".self_attn.relative_attention_bias.weight"
                ] = checkpoint[
                    "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
                ].to(
                    torch.float16
                )
                eole_safetensor[
                    "decoder.transformer_layers."
                    + str(i)
                    + ".self_attn.relative_attention_bias.weight"
                ] = checkpoint[
                    "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
                ].to(
                    torch.float16
                )

                eole_safetensor[
                    "encoder.transformer_layers." + str(i) + ".layer_norm.weight"
                ] = checkpoint[
                    "encoder.block." + str(i) + ".layer.0.layer_norm.weight"
                ].to(
                    torch.float16
                )
                eole_safetensor[
                    "decoder.transformer_layers." + str(i) + ".layer_norm_1.weight"
                ] = checkpoint[
                    "decoder.block." + str(i) + ".layer.0.layer_norm.weight"
                ].to(
                    torch.float16
                )

                eole_safetensor[
                    "encoder.transformer_layers." + str(i) + ".feed_forward.w_1.weight"
                ] = checkpoint[
                    "encoder.block." + str(i) + ".layer.1.DenseReluDense.wi_0.weight"
                ].to(
                    torch.float16
                )

                eole_safetensor[
                    "encoder.transformer_layers." + str(i) + ".feed_forward.w_2.weight"
                ] = checkpoint[
                    "encoder.block." + str(i) + ".layer.1.DenseReluDense.wo.weight"
                ].to(
                    torch.float16
                )
                eole_safetensor[
                    "encoder.transformer_layers." + str(i) + ".feed_forward.w_3.weight"
                ] = checkpoint[
                    "encoder.block." + str(i) + ".layer.1.DenseReluDense.wi_1.weight"
                ].to(
                    torch.float16
                )

                eole_safetensor[
                    "decoder.transformer_layers."
                    + str(i)
                    + ".context_attn.linear_query.weight"
                ] = (
                    checkpoint[
                        "decoder.block." + str(i) + ".layer.1.EncDecAttention.q.weight"
                    ]
                    / (dimperhead**-0.5)
                ).to(
                    torch.float16
                )
                eole_safetensor[
                    "decoder.transformer_layers."
                    + str(i)
                    + ".context_attn.linear_keys.weight"
                ] = checkpoint[
                    "decoder.block." + str(i) + ".layer.1.EncDecAttention.k.weight"
                ].to(
                    torch.float16
                )
                eole_safetensor[
                    "decoder.transformer_layers."
                    + str(i)
                    + ".context_attn.linear_values.weight"
                ] = checkpoint[
                    "decoder.block." + str(i) + ".layer.1.EncDecAttention.v.weight"
                ].to(
                    torch.float16
                )
                eole_safetensor[
                    "decoder.transformer_layers."
                    + str(i)
                    + ".context_attn.final_linear.weight"
                ] = checkpoint[
                    "decoder.block." + str(i) + ".layer.1.EncDecAttention.o.weight"
                ].to(
                    torch.float16
                )

                eole_safetensor[
                    "decoder.transformer_layers." + str(i) + ".layer_norm_2.weight"
                ] = checkpoint[
                    "decoder.block." + str(i) + ".layer.1.layer_norm.weight"
                ].to(
                    torch.float16
                )

                eole_safetensor[
                    "decoder.transformer_layers." + str(i) + ".feed_forward.w_1.weight"
                ] = checkpoint[
                    "decoder.block." + str(i) + ".layer.2.DenseReluDense.wi_0.weight"
                ].to(
                    torch.float16
                )

                eole_safetensor[
                    "decoder.transformer_layers." + str(i) + ".feed_forward.w_2.weight"
                ] = checkpoint[
                    "decoder.block." + str(i) + ".layer.2.DenseReluDense.wo.weight"
                ].to(
                    torch.float16
                )
                eole_safetensor[
                    "decoder.transformer_layers." + str(i) + ".feed_forward.w_3.weight"
                ] = checkpoint[
                    "decoder.block." + str(i) + ".layer.2.DenseReluDense.wi_1.weight"
                ].to(
                    torch.float16
                )

                eole_safetensor[
                    "encoder.transformer_layers."
                    + str(i)
                    + ".feed_forward.layer_norm.weight"
                ] = checkpoint[
                    "encoder.block." + str(i) + ".layer.1.layer_norm.weight"
                ].to(
                    torch.float16
                )
                eole_safetensor[
                    "decoder.transformer_layers."
                    + str(i)
                    + ".feed_forward.layer_norm.weight"
                ] = checkpoint[
                    "decoder.block." + str(i) + ".layer.2.layer_norm.weight"
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

        tokenizer = Tokenizer(model_path=args.tokenizer_model)
        vocabs = {}
        vocab = tokenizer.vocab

        src_vocab = pyonmttok.build_vocab_from_tokens(
            vocab,
            maximum_size=tokenizer.n_words,
            special_tokens=["<blank>", "</s>", "<unk>"],
        )
        vocabs["src"] = src_vocab
        vocabs["tgt"] = src_vocab

        vocabs["decoder_start_token"] = "<blank>"

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
            model=TransformerModelConfig(
                layers=decoder_layers,
                hidden_size=hidden_size,
                heads=heads,
                transformer_ff=transformer_ff,
                embeddings=EmbeddingsConfig(
                    src_word_vec_size=src_word_vec_size,
                    tgt_word_vec_size=tgt_word_vec_size,
                ),
                layer_norm="rms",
                pos_ffn_activation_fn="gated-gelu",
                self_attn_type="scaled-dot",
                max_relative_positions=params["relative_attention_max_distance"],
                relative_positions_buckets=params["relative_attention_num_buckets"],
                parallel_residual=False,
                add_qkvbias=False,
                add_ffnbias=False,
                encoder={"layers": encoder_layers},
                decoder={"layers": decoder_layers},
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

        serializedStr = open(args.tokenizer_model, "rb").read()
        m = spmodel.ModelProto()
        m.ParseFromString(serializedStr)
        m.pieces[0].piece = "<blank>"
        with open(args.tokenizer_model + ".new", "wb") as f:
            f.write(m.SerializeToString())

        print(
            "The sentencepiece model %s has been patched with <blank> instead of <pad>",
            args.tokenizer_model,
        )
        print("With eole use the model with the .new extension or rename it")
