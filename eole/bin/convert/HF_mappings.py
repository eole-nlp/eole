from copy import deepcopy
from collections import defaultdict

from eole.config.models import (
    TransformerEncoderModelConfig,
    TransformerModelConfig,
    TransformerLMModelConfig,
    VisionTransformerLMModelConfig,
)
from eole.config import recursive_update_dict


# Default tensor key mappings, based on Llama
BASE_KEY_MAP = {
    "decoder_layer_prefix": "model.layers.",
    # keys outside of encoder/decoder subsections will be centralized in the first shard
    "tgt_emb.embeddings.weight": "model.embed_tokens.weight",
    "decoder.layer_norm.weight": "model.norm.weight",
    "generator.weight": "lm_head.weight",
    "decoder": {
        ".self_attn.linear_query.": ".self_attn.q_proj.",
        ".self_attn.linear_keys.": ".self_attn.k_proj.",
        ".self_attn.linear_values.": ".self_attn.v_proj.",
        ".self_attn.final_linear.": ".self_attn.o_proj.",
        ".mlp.gate_up_proj.": ".mlp.gate_proj.",
        ".mlp.down_proj.": ".mlp.down_proj.",
        ".mlp.up_proj.": ".mlp.up_proj.",
        ".input_layernorm.": ".input_layernorm.",
        ".post_attention_layernorm.": ".post_attention_layernorm.",
    },
}


# Model-specific overrides for key mappings and configurations
# root keys are weights to be added in the first shard
# encoder/decoder sections are modules of each encoder/decoder layer
MODEL_OVERRIDES = {
    "LlamaForCausalLM": {},  # default
    "MistralForCausalLM": {},
    "Qwen2ForCausalLM": { # for bagel, but we need to add some conditions to keep supporting real qwen2...
        "decoder_layer_prefix": "language_model.model.layers.",
        "decoder.layer_norm.weight": "language_model.model.norm.weight",
        "decoder.layer_norm_moe_gen.weight": "language_model.model.norm_moe_gen.weight",
        "encoder_layer_prefix": "vit_model.vision_model.encoder.layers.",
        "encoder.patch_conv.weight": "vit_model.vision_model.embeddings.patch_embedding.weight",
        "encoder.patch_conv.bias": "vit_model.vision_model.embeddings.patch_embedding.bias",
        "encoder.position_embeddings.weight": "vit_model.vision_model.embeddings.position_embedding.weight",
        "encoder.post_layernorm.weight": "vit_model.vision_model.post_layernorm.weight",
        "encoder.post_layernorm.bias": "vit_model.vision_model.post_layernorm.bias",
        "tgt_emb.embeddings.weight": "language_model.model.embed_tokens.weight",
        "generator.weight": "language_model.lm_head.weight",
        # vision_adapter
        "adapter.w_in.weight": "connector.fc1.weight",
        "adapter.w_in.bias": "connector.fc1.bias",
        "adapter.w_out.weight": "connector.fc2.weight",
        "adapter.w_out.bias": "connector.fc2.bias",
        # additional stuff, mostly replicated as-is for now
        "vit_pos_embed.pos_embed": "vit_pos_embed.pos_embed",
        "latent_pos_embed.pos_embed": "latent_pos_embed.pos_embed",
        "time_embedder.mlp.0.weight": "time_embedder.mlp.0.weight",
        "time_embedder.mlp.0.bias": "time_embedder.mlp.0.bias",
        "time_embedder.mlp.2.weight": "time_embedder.mlp.2.weight",
        "time_embedder.mlp.2.bias": "time_embedder.mlp.2.bias",
        "vae2llm.weight": "vae2llm.weight",
        "vae2llm.bias": "vae2llm.bias",
        "llm2vae.weight": "llm2vae.weight",
        "llm2vae.bias": "llm2vae.bias",
        # TODO: not sure how to properly grab VAE stuff
        "decoder": {
            ".self_attn.q_norm.": ".self_attn.q_norm.",
            ".self_attn.k_norm.": ".self_attn.k_norm.",
            # MOE GEN (simplify with loop?)
            ".self_attn.q_norm_moe_gen.": ".self_attn.q_norm_moe_gen.",
            ".self_attn.k_norm_moe_gen.": ".self_attn.k_norm_moe_gen.",
            ".self_attn.linear_query_moe_gen.": ".self_attn.q_proj_moe_gen.",
            ".self_attn.linear_keys_moe_gen.": ".self_attn.k_proj_moe_gen.",
            ".self_attn.linear_values_moe_gen.": ".self_attn.v_proj_moe_gen.",
            ".self_attn.final_linear_moe_gen.": ".self_attn.o_proj_moe_gen.",
            ".mlp_moe_gen.gate_up_proj.": ".mlp_moe_gen.gate_proj.",
            ".mlp_moe_gen.down_proj.": ".mlp_moe_gen.down_proj.",
            ".mlp_moe_gen.up_proj.": ".mlp_moe_gen.up_proj.",
            ".input_layernorm_moe_gen.": ".input_layernorm_moe_gen.",
            ".post_attention_layernorm_moe_gen.": ".post_attention_layernorm_moe_gen.",
        },
        "encoder": {
            ".self_attn.linear_query.": ".self_attn.q_proj.",
            ".self_attn.linear_keys.": ".self_attn.k_proj.",
            ".self_attn.linear_values.": ".self_attn.v_proj.",
            ".self_attn.final_linear.": ".self_attn.out_proj.",
            ".mlp.gate_up_proj.": ".mlp.fc1.",
            ".mlp.down_proj.": ".mlp.fc2.",
            ".input_layernorm.": ".layer_norm1.",
            ".post_attention_layernorm.": ".layer_norm2.",
        },
        "config": {
            "add_qkvbias": True,
            "add_final_linear_bias": False,
            # "ffn_layernorm": True,
            "decoder": {
                "query_norm": True,
                "key_norm": True,
            },
        }
    },
    "Qwen3ForCausalLM": {
        "decoder": {
            ".self_attn.q_norm.": ".self_attn.q_norm.",
            ".self_attn.k_norm.": ".self_attn.k_norm.",
        },
        "config": {
            "decoder": {
                "query_norm": True,
                "key_norm": True,
            }
        },
    },
    "Qwen3MoeForCausalLM": {
        "decoder": {
            ".self_attn.q_norm.": ".self_attn.q_norm.",
            ".self_attn.k_norm.": ".self_attn.k_norm.",
            ".mlp.gate.": ".mlp.gate.",
            **{f".mlp.experts.{i}.gate_up_proj.": f".mlp.experts.{i}.gate_proj." for i in range(128)},
            **{f".mlp.experts.{i}.up_proj.": f".mlp.experts.{i}.up_proj." for i in range(128)},
            **{f".mlp.experts.{i}.down_proj.": f".mlp.experts.{i}.down_proj." for i in range(128)},
        },
        "config": {
            "decoder": {
                "query_norm": True,
                "key_norm": True,
            }
        },
    },
    "Gemma2ForCausalLM": {
        "decoder": {
            ".pre_feedforward_layernorm.": ".pre_feedforward_layernorm.",
            ".post_feedforward_layernorm.": ".post_feedforward_layernorm.",
        },
        "config": {
            "share_decoder_embeddings": True,
            "ffn_layernorm": True,
            "embeddings": {
                "normalize": True,
            },
        },
    },
    "MixtralForCausalLM": {
        "decoder": {
            ".mlp.gate.": ".block_sparse_moe.gate.",
            **{
                f".mlp.experts.{i}.{attr}": f".block_sparse_moe.experts.{i}.w{j+1}."
                for i in range(8)
                for j, attr in enumerate(["gate_up_proj.", "down_proj.", "up_proj."])
            },
            ".post_attention_layernorm.weight": ".post_attention_layernorm.weight",
        },
        "config": {
            "decoder": {"transformer_ff_moe": 14336},
            "shared_layer_norm": True,
        },
    },
    "PhiForCausalLM": {
        "decoder.layer_norm.weight": "model.final_layernorm.weight",
        "decoder.layer_norm.bias": "model.final_layernorm.bias",
        "generator.bias": "lm_head.bias",
        "decoder": {
            ".self_attn.final_linear.": ".self_attn.dense.",
            ".mlp.gate_up_proj.": ".mlp.fc1.",
            ".mlp.down_proj.": ".mlp.fc2.",
            ".input_layernorm.": (".input_layernorm.", ""),
        },
        "config": {
            "parallel_residual": True,
            "shared_layer_norm": True,
            "add_qkvbias": True,
            "add_final_linear_bias": True,
            "add_ffnbias": True,
            "post_attention_layernorm": False,
        },
    },
    "Phi3ForCausalLM": {
        "decoder": {
            ".self_attn.linear_query.": (".self_attn.qkv_proj.", "[:hidden_size, :]"),
            ".self_attn.linear_keys.": (
                ".self_attn.qkv_proj.",
                "[hidden_size:2*hidden_size, :]",
            ),
            ".self_attn.linear_values.": (".self_attn.qkv_proj.", "[-hidden_size:, :]"),
            ".mlp.gate_up_proj.": (".mlp.gate_up_proj.", "[:transformer_ff, :]"),
            ".mlp.up_proj.": (".mlp.gate_up_proj.", "[transformer_ff:, :]"),
        }
    },
    "GPT2LMHeadModel": {
        "decoder_layer_prefix": "h.",
        "decoder.layer_norm.weight": "ln_f.weight",
        "decoder.layer_norm.bias": "ln_f.bias",
        "tgt_emb.pe.weight": "wpe.weight",
        "tgt_emb.embeddings.weight": "wte.weight",
        "generator.weight": "wte.weight",  # shared with embeddings
        "decoder": {
            ".self_attn.linear_query.": (".attn.c_attn.", ".t()[:hidden_size, ...]"),
            ".self_attn.linear_keys.": (
                ".attn.c_attn.",
                ".t()[hidden_size:2*hidden_size, ...]",
            ),
            ".self_attn.linear_values.": (".attn.c_attn.", ".t()[-hidden_size:, ...]"),
            ".self_attn.final_linear.": (".attn.c_proj.", ".t()"),
            ".mlp.gate_up_proj.": (".mlp.c_fc.", ".t()"),
            ".mlp.down_proj.": (".mlp.c_proj.", ".t()"),
            ".input_layernorm.": ".ln_1.",
            ".post_attention_layernorm.": ".ln_2.",
        },
        "config": {
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
    },
    "XLMRobertaXLForMaskedLM": {
        "encoder_layer_prefix": "roberta.encoder.layer.",
        "encoder.layer_norm.weight": "roberta.encoder.LayerNorm.weight",
        "encoder.layer_norm.bias": "roberta.encoder.LayerNorm.bias",
        "src_emb.embeddings.weight": "roberta.embeddings.word_embeddings.weight",
        "src_emb.pe.weight": "roberta.embeddings.position_embeddings.weight",
        "encoder": {
            ".self_attn.linear_query.": ".attention.self.query.",
            ".self_attn.linear_keys.": ".attention.self.key.",
            ".self_attn.linear_values.": ".attention.self.value.",
            ".self_attn.final_linear.": ".attention.output.dense.",
            ".mlp.gate_up_proj.": ".intermediate.dense.",
            ".mlp.down_proj.": ".output.dense.",
            ".input_layernorm.": ".attention.self_attn_layer_norm.",
            ".post_attention_layernorm.": ".LayerNorm.",
        },
        "config": {
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
    },
    "LlavaForConditionalGeneration": {
        "decoder_layer_prefix": "language_model.model.layers.",
        "tgt_emb.embeddings.weight": "language_model.model.embed_tokens.weight",
        "decoder.layer_norm.weight": "language_model.model.norm.weight",
        "generator.weight": "language_model.lm_head.weight",
        "encoder.patch_conv.weight": "vision_tower.patch_conv.weight",
        "encoder.ln_pre.weight": "vision_tower.ln_pre.weight",
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
            ".input_layernorm.": ".attention_norm.",  # not sure about this one
            ".post_attention_layernorm.": ".ffn_norm.",
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
            ".input_layernorm.": ".attention_norm.",  # not sure about this one
            ".post_attention_layernorm.": ".ffn_norm.",
        },
        # vision_adapter
        "adapter.w_in.weight": "multi_modal_projector.linear_1.weight",
        "adapter.w_out.weight": "multi_modal_projector.linear_2.weight",
        "adapter.layernorm.weight": "multi_modal_projector.norm.weight",
        "adapter.patch_merger.merging_layer.weight": "multi_modal_projector.patch_merger.merging_layer.weight",
    },
    "Gemma3ForConditionalGeneration": {
        "decoder_layer_prefix": "language_model.model.layers.",
        "tgt_emb.embeddings.weight": "language_model.model.embed_tokens.weight",
        "decoder.layer_norm.weight": "language_model.model.norm.weight",
        # "generator.weight": "language_model.lm_head.weight", # probably shared with embeddings
        # decoder layer modules
        "decoder": {
            ".self_attn.q_norm.": ".self_attn.q_norm.",
            ".self_attn.k_norm.": ".self_attn.k_norm.",
            ".pre_feedforward_layernorm.": ".pre_feedforward_layernorm.",
            ".post_feedforward_layernorm.": ".post_feedforward_layernorm.",
        },
        "encoder_layer_prefix": "vision_tower.vision_model.encoder.layers.",
        "encoder.patch_conv.weight": "vision_tower.vision_model.embeddings.patch_embedding.weight",
        "encoder.patch_conv.bias": "vision_tower.vision_model.embeddings.patch_embedding.bias",
        "encoder.post_layernorm.weight": "vision_tower.vision_model.post_layernorm.weight",
        "encoder.post_layernorm.bias": "vision_tower.vision_model.post_layernorm.bias",
        "encoder.position_embeddings.weight": "vision_tower.vision_model.embeddings.position_embedding.weight",
        # "encoder.ln_pre.weight": "vision_tower.ln_pre.weight", # no ln_pre in Gemma3
        # encoder layers modules
        "encoder": {
            ".self_attn.linear_query.": ".self_attn.q_proj.",
            ".self_attn.linear_keys.": ".self_attn.k_proj.",
            ".self_attn.linear_values.": ".self_attn.v_proj.",
            ".self_attn.final_linear.": ".self_attn.out_proj.",
            ".mlp.gate_up_proj.": ".mlp.fc1.",
            ".mlp.down_proj.": ".mlp.fc2.",
            ".input_layernorm.": ".layer_norm1.",
            ".post_attention_layernorm.": ".layer_norm2.",
        },
        "adapter.w_in.weight": ("multi_modal_projector.mm_input_projection_weight", ".t()"),
        "adapter.norm.weight": "multi_modal_projector.mm_soft_emb_norm.weight",
        "config": {
            "share_decoder_embeddings": True,
            "ffn_layernorm": True,
            "embeddings": {
                "normalize": True,
            },
            "decoder": {
                "query_norm": True,
                "key_norm": True,
            },
        },
    },
    "M2M100ForConditionalGeneration": {
        "decoder_layer_prefix": "model.decoder.layers.",
        "src_emb.embeddings.weight": "model.encoder.embed_tokens.weight",
        "tgt_emb.embeddings.weight": "model.decoder.embed_tokens.weight",
        "decoder.layer_norm.weight": "model.decoder.layer_norm.weight",
        "decoder.layer_norm.bias": "model.decoder.layer_norm.bias",
        "decoder": {
            ".self_attn.linear_query.": ".self_attn.q_proj.",
            ".self_attn.linear_keys.": ".self_attn.k_proj.",
            ".self_attn.linear_values.": ".self_attn.v_proj.",
            ".self_attn.final_linear.": ".self_attn.out_proj.",
            ".precontext_layernorm.": ".encoder_attn_layer_norm.",
            ".context_attn.linear_query.": ".encoder_attn.q_proj.",
            ".context_attn.linear_keys.": ".encoder_attn.k_proj.",
            ".context_attn.linear_values.": ".encoder_attn.v_proj.",
            ".context_attn.final_linear.": ".encoder_attn.out_proj.",
            ".mlp.gate_up_proj.": ".fc1.",
            ".mlp.down_proj.": ".fc2.",
            ".input_layernorm.": ".self_attn_layer_norm.",
            ".post_attention_layernorm.": ".final_layer_norm.",
        },
        "encoder_layer_prefix": "model.encoder.layers.",
        "encoder.layer_norm.weight": "model.encoder.layer_norm.weight",
        "encoder.layer_norm.bias": "model.encoder.layer_norm.bias",
        "encoder": {
            ".self_attn.linear_query.": ".self_attn.q_proj.",
            ".self_attn.linear_keys.": ".self_attn.k_proj.",
            ".self_attn.linear_values.": ".self_attn.v_proj.",
            ".self_attn.final_linear.": ".self_attn.out_proj.",
            ".mlp.gate_up_proj.": ".fc1.",
            ".mlp.down_proj.": ".fc2.",
            ".input_layernorm.": ".self_attn_layer_norm.",
            ".post_attention_layernorm.": ".final_layer_norm.",
        },
        "config": {
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
    },
}

# Combine base mappings with overrides
# KEY_MAPS = {model: {**BASE_KEY_MAP, **overrides} for model, overrides in MODEL_OVERRIDES.items()}
KEY_MAPS = {
    model: recursive_update_dict(deepcopy(BASE_KEY_MAP), overrides, {}) for model, overrides in MODEL_OVERRIDES.items()
}

# Layer norm type
LN_TABLE = defaultdict(
    lambda: "rms",
    {
        "PhiForCausalLM": "standard",
        "GPT2LMHeadModel": "standard",
        "XLMRobertaXLForMaskedLM": "standard",
        "Gemma2ForCausalLM": "gemma-rms",
        "M2M100ForConditionalGeneration": "standard",
        "Gemma3ForConditionalGeneration": "gemma-rms",
        "Qwen2ForCausalLM": "rms",
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
        "Gemma3ForConditionalGeneration": "gated-gelu-tanh",
        "M2M100ForConditionalGeneration": "relu",
    },
)

# Not used anymore since Gemma3 PR, not sure if needed
# VISION_ACT_TABLE = defaultdict(
#     lambda: "gated-silu",
#     {
#         "Mistral3ForConditionalGeneration": "gated-gelu",
#     },
# )

# Eole config class
ARCH_TABLE = defaultdict(
    lambda: TransformerLMModelConfig,
    {
        "XLMRobertaXLForMaskedLM": TransformerEncoderModelConfig,
        "LlavaForConditionalGeneration": VisionTransformerLMModelConfig,
        "Mistral3ForConditionalGeneration": VisionTransformerLMModelConfig,
        "Gemma3ForConditionalGeneration": VisionTransformerLMModelConfig,
        "M2M100ForConditionalGeneration": TransformerModelConfig,
        "Qwen2ForCausalLM": VisionTransformerLMModelConfig,
    },
)

# Default tokenization transform
TOK_TABLE = defaultdict(lambda: "huggingface_tokenize")
