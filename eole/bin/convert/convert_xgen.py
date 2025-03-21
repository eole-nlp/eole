#!/usr/bin/env python
import json
import torch
import pyonmttok
from eole.inputters.inputter import vocabs_to_dict
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


@register_bin(name="xgen")
class XgenConverter(BaseBin):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--model_dir",
            type=str,
            required=True,
            help="""Path to the model directory""",
        )
        parser.add_argument("--vocab_file", type=str, required=True, help="""Path to the vocab file""")
        parser.add_argument("--output", type=str, required=True, help="""Path to the model directory""")
        parser.add_argument("--nshards", type=int, default=1, help="""Path to the model directory""")

    @classmethod
    def run(cls, args):
        from transformers import LlamaForCausalLM

        model = LlamaForCausalLM.from_pretrained(
            args.model_dir,
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
            trust_remote_code=True,
        )
        checkpoint = model.state_dict()

        params_json = os.path.join(args.model_dir, "config.json")
        with open(params_json, encoding="utf-8") as fparam:
            params = json.load(fparam)

        decoder_layers = params["num_hidden_layers"]
        src_word_vec_size = params["hidden_size"]
        tgt_word_vec_size = params["hidden_size"]
        hidden_size = params["hidden_size"]
        heads = params["num_attention_heads"]
        vocab_size = params["vocab_size"]
        transformer_ff = params["intermediate_size"]

        for shard in range(args.nshards):

            print("starting output shard: %d/%d" % (shard + 1, args.nshards))
            eole_safetensor = {}

            if shard == 0:
                eole_safetensor["tgt_emb.make_embedding.emb_luts.0.weight"] = checkpoint[
                    "model.embed_tokens.weight"
                ].to(torch.float16)
                eole_safetensor["decoder.layer_norm.weight"] = checkpoint["model.norm.weight"].to(torch.float16)

                eole_safetensor["generator.weight"] = checkpoint["lm_head.weight"].to(torch.float16)
                eole_safetensor["generator.bias"] = torch.zeros(
                    eole_safetensor["generator.weight"].size(0), dtype=torch.float16
                )

            for i in range(
                -(decoder_layers // -args.nshards) * shard,
                min(-(decoder_layers // -args.nshards) * (shard + 1), decoder_layers),
                1,
            ):
                eole_safetensor["decoder.transformer_layers." + str(i) + ".self_attn.linear_query.weight"] = (
                    checkpoint["model.layers." + str(i) + ".self_attn.q_proj.weight"]
                    .view(heads, 2, hidden_size // heads // 2, hidden_size)
                    .transpose(1, 2)
                    .reshape(hidden_size, hidden_size)
                ).to(torch.float16)
                eole_safetensor["decoder.transformer_layers." + str(i) + ".self_attn.linear_keys.weight"] = (
                    checkpoint["model.layers." + str(i) + ".self_attn.k_proj.weight"]
                    .view(heads, 2, hidden_size // heads // 2, hidden_size)
                    .transpose(1, 2)
                    .reshape(hidden_size, hidden_size)
                ).to(torch.float16)
                eole_safetensor["decoder.transformer_layers." + str(i) + ".self_attn.linear_values.weight"] = (
                    checkpoint["model.layers." + str(i) + ".self_attn.v_proj.weight"].to(torch.float16)
                )

                eole_safetensor["decoder.transformer_layers." + str(i) + ".self_attn.final_linear.weight"] = checkpoint[
                    "model.layers." + str(i) + ".self_attn.o_proj.weight"
                ].to(torch.float16)

                eole_safetensor["decoder.transformer_layers." + str(i) + ".layer_norm_1.weight"] = checkpoint[
                    "model.layers." + str(i) + ".input_layernorm.weight"
                ].to(torch.float16)

                eole_safetensor["decoder.transformer_layers." + str(i) + ".feed_forward.w_1.weight"] = checkpoint[
                    "model.layers." + str(i) + ".mlp.gate_proj.weight"
                ].to(torch.float16)

                eole_safetensor["decoder.transformer_layers." + str(i) + ".feed_forward.w_2.weight"] = checkpoint[
                    "model.layers." + str(i) + ".mlp.down_proj.weight"
                ].to(torch.float16)
                eole_safetensor["decoder.transformer_layers." + str(i) + ".feed_forward.w_3.weight"] = checkpoint[
                    "model.layers." + str(i) + ".mlp.up_proj.weight"
                ].to(torch.float16)

                eole_safetensor["decoder.transformer_layers." + str(i) + ".feed_forward.layer_norm.weight"] = (
                    checkpoint["model.layers." + str(i) + ".post_attention_layernorm.weight"].to(torch.float16)
                )

            if shard == 0:
                transformer_ff = eole_safetensor["decoder.transformer_layers.0.feed_forward.w_1.weight"].size(0)
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

        with open(os.path.join(args.output, "vocab.txt"), "w", encoding="utf-8") as vocabfile:
            for tok in vocab_dict["src"]:
                vocabfile.write(tok + "\n")

        position_encoding = {
            "position_encoding_type": "Rotary",
            "n_positions": 0,
        }

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
            transforms_configs={"filtertoolong": {"src_seq_length": 512, "tgt_seq_length": 512}},
            model=TransformerLMModelConfig(
                layers=decoder_layers,
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
                layer_norm="rms",
                pos_ffn_activation_fn="silu",
                parallel_residual=False,
                add_qkvbias=False,
                add_ffnbias=False,
            ),
            training=TrainingConfig(
                compute_dtype="fp16",
                batch_size=896,
                batch_size_multiple=1,
                batch_type="tokens",
                normalization="tokens",
                accum_count=[32],
                accum_steps=[0],
                valid_batch_size=256,
            ),
        )

        config_dict = recursive_model_fields_set(config)
        with open(os.path.join(args.output, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
