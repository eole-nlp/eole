from copy import deepcopy
from collections import defaultdict

from eole.config.models import (
    TransformerEncoderModelConfig,
    TransformerModelConfig,
    TransformerLMModelConfig,
    VisionTransformerLMModelConfig,
)
from eole.config import recursive_update_dict
from eole.constants import PositionEncodingType

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
    "Qwen2ForCausalLM": {
        "config": {
            "add_qkvbias": True,
            "add_final_linear_bias": False,
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
        },
        "config": {
            "decoder": {"moe_transformer_ff": 14336},
            "moe_softmax_after": True,
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
    "PhiMoEForCausalLM": {
        "decoder": {
            ".mlp.gate.": ".block_sparse_moe.gate.",
            **{
                f".mlp.experts.{i}.{attr}": f".block_sparse_moe.experts.{i}.w{j+1}."
                for i in range(16)
                for j, attr in enumerate(["gate_up_proj.", "down_proj.", "up_proj."])
            },
        },
        "config": {
            "decoder": {"moe_transformer_ff": 6400},
            "add_qkvbias": True,
            "add_final_linear_bias": True,
        },
    },
    "HunYuanDenseV1ForCausalLM": {
        "decoder": {
            ".self_attn.q_norm.": ".self_attn.query_layernorm.",
            ".self_attn.k_norm.": ".self_attn.key_layernorm.",
        },
        "config": {
            "query_norm": True,
            "key_norm": True,
            "qk_norm_post_rope": True,
        },
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
            ".input_layernorm.": ".attention_norm.",
            ".post_attention_layernorm.": ".ffn_norm.",
        },
        # vision_adapter
        "adapter.w_in.weight": "multi_modal_projector.linear_1.weight",
        "adapter.w_in.bias": "multi_modal_projector.linear_1.bias",
        "adapter.w_out.weight": "multi_modal_projector.linear_2.weight",
        "adapter.w_out.bias": "multi_modal_projector.linear_2.bias",
        "config": {
            "encoder": {
                "num_channels": 3,
                "image_token_id": 10,
                "layernorm_pre": True,
                "patch_conv_bias": False,
            },
        },
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
            ".input_layernorm.": ".attention_norm.",
            ".post_attention_layernorm.": ".ffn_norm.",
        },
        # vision_adapter
        "adapter.w_in.weight": "multi_modal_projector.linear_1.weight",
        "adapter.w_out.weight": "multi_modal_projector.linear_2.weight",
        "adapter.layernorm.weight": "multi_modal_projector.norm.weight",
        "adapter.patch_merger.merging_layer.weight": "multi_modal_projector.patch_merger.merging_layer.weight",
        "config": {
            "encoder": {
                "num_channels": 3,
                "image_token_id": 10,
                "layernorm_pre": True,
                "patch_conv_bias": False,
            },
        },
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
            "adapter": "gemma3",
            "decoder": {
                "query_norm": True,
                "key_norm": True,
                "rope_config": {
                    "rotary_theta": 1000000,
                    "scaling_type": "gemma3",
                    "rotary_theta_local": 10000,
                    "interleave_local": 6,
                    "tmax_index": 1,
                },
            },
            "encoder": {
                "mlp_activation_fn": "gelu-tanh",
                "position_encoding_type": PositionEncodingType.Learned,
                "layer_norm": "standard",
                "add_ffnbias": True,
                "add_final_linear_bias": True,
                "add_qkvbias": True,
                "layernorm_pre": False,
                "layernorm_post": True,
                "patch_conv_bias": True,
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
    "DeepseekOCRForCausalLM": {
        "decoder": {
            ".mlp.gate.": ".mlp.gate.",
            **{
                f".mlp.experts.{i}.{k}": f".mlp.experts.{i}.{v}"
                for i in range(64)
                for k, v in {"gate_up_proj.": "gate_proj.", "down_proj.": "down_proj.", "up_proj.": "up_proj."}.items()
            },
            **{
                f".mlp.shared_experts.{k}": f".mlp.shared_experts.{v}"
                for k, v in {"gate_up_proj.": "gate_proj.", "down_proj.": "down_proj.", "up_proj.": "up_proj."}.items()
            },
        },
        "encoder.patch_conv.weight": "model.vision_model.embeddings.patch_embedding.weight",
        "encoder.position_embeddings.weight": "model.vision_model.embeddings.position_embedding.weight",
        "encoder.class_embedding.weight": "model.vision_model.embeddings.class_embedding",
        "encoder.ln_pre.weight": "model.vision_model.pre_layrnorm.weight",
        "encoder.ln_pre.bias": "model.vision_model.pre_layrnorm.bias",
        # vision_tower
        "encoder_layer_prefix": "model.vision_model.transformer.layers.",
        "encoder": {
            "layers": 24,
            ".self_attn.linear_query.weight": (".self_attn.qkv_proj.weight", "[:hidden_size, :]"),
            ".self_attn.linear_keys.weight": (".self_attn.qkv_proj.weight", "[hidden_size:2*hidden_size, :]"),
            ".self_attn.linear_values.weight": (".self_attn.qkv_proj.weight", "[-hidden_size:, :]"),
            ".self_attn.linear_query.bias": (".self_attn.qkv_proj.bias", "[:hidden_size]"),
            ".self_attn.linear_keys.bias": (".self_attn.qkv_proj.bias", "[hidden_size:2*hidden_size]"),
            ".self_attn.linear_values.bias": (".self_attn.qkv_proj.bias", "[-hidden_size:]"),
            ".self_attn.final_linear.": ".self_attn.out_proj.",
            ".mlp.gate_up_proj.": ".mlp.fc1.",
            ".mlp.down_proj.": ".mlp.fc2.",
            ".input_layernorm.": ".layer_norm1.",
            ".post_attention_layernorm.": ".layer_norm2.",
        },
        # vision_adapter
        "adapter.w_in.weight": "model.projector.layers.weight",
        "adapter.w_in.bias": "model.projector.layers.bias",
        # Misc deepseek ocr
        "image_newline": "model.image_newline",
        "view_separator": "model.view_seperator",
        # sam_model
        "encoder.sam.patch_embed.proj.weight": "model.sam_model.patch_embed.proj.weight",
        "encoder.sam.patch_embed.proj.bias": "model.sam_model.patch_embed.proj.bias",
        "encoder.sam.pos_embed": "model.sam_model.pos_embed",
        "encoder.sam.neck.0.weight": "model.sam_model.neck.0.weight",
        "encoder.sam.neck.1.weight": "model.sam_model.neck.1.weight",
        "encoder.sam.neck.1.bias": "model.sam_model.neck.1.bias",
        "encoder.sam.neck.2.weight": "model.sam_model.neck.2.weight",
        "encoder.sam.neck.3.weight": "model.sam_model.neck.3.weight",
        "encoder.sam.neck.3.bias": "model.sam_model.neck.3.bias",
        "encoder.sam.net_2.weight": "model.sam_model.net_2.weight",
        "encoder.sam.net_3.weight": "model.sam_model.net_3.weight",
        "encoder_sam_layer_prefix": "model.sam_model.blocks.",
        "encoder.sam": {
            "layers": 12,
            ".attn.proj.": ".attn.proj.",
            ".attn.qkv.": ".attn.qkv.",
            ".attn.rel_pos_h": ".attn.rel_pos_h",
            ".attn.rel_pos_w": ".attn.rel_pos_w",
            ".mlp.lin1.": ".mlp.lin1.",
            ".mlp.lin2.": ".mlp.lin2.",
            ".norm1.": ".norm1.",
            ".norm2.": ".norm2.",
        },
        "config": {
            "adapter": "deepseekocr",
            "adapter_bias": True,
            "decoder": {
                "layer_norm": "rms",
                "norm_eps": 1e-6,
            },
            "encoder": {
                "mlp_activation_fn": "quick_gelu",
                "layer_norm": "standard",
                "norm_eps": 1e-5,
                "position_encoding_type": PositionEncodingType.Learned,
                "add_ffnbias": True,
                "add_final_linear_bias": True,
                "add_qkvbias": True,
                "num_channels": 3,
                "image_token_id": 128815,
                "layernorm_pre": True,
                "patch_conv_bias": False,
                "encoder_sam": True,
                "use_class_embedding": True,
            },
        },
    },
    "HunYuanVLForConditionalGeneration": {
        "decoder": {
            ".self_attn.q_norm.": ".self_attn.query_layernorm.",
            ".self_attn.k_norm.": ".self_attn.key_layernorm.",
        },
        "config": {
            "adapter": "hunyuanocr",
            "decoder": {
                "query_norm": True,
                "key_norm": True,
                "qk_norm_post_rope": True,
            },
            "encoder": {
                "position_encoding_type": PositionEncodingType.Learned,
                "interpolate_mode": "bilinear",
                "n_positions": 16384,
                "layer_norm": "standard",
                "layernorm_pre": False,
                "mlp_activation_fn": "gelu",
                "add_ffnbias": True,
                "add_qkvbias": True,
                "patch_conv_bias": True,
                "image_token_id": 120120,
            },
        },
        "encoder_layer_prefix": "vit.layers.",
        "encoder.patch_conv.weight": "vit.embeddings.patch_embedding.weight",
        "encoder.patch_conv.bias": "vit.embeddings.patch_embedding.bias",
        "encoder.position_embeddings.weight": "vit.embeddings.position_embedding.weight",
        # encoder layers modules
        "encoder": {
            ".self_attn.linear_query.": ".self_attn.q_proj.",
            ".self_attn.linear_keys.": ".self_attn.k_proj.",
            ".self_attn.linear_values.": ".self_attn.v_proj.",
            ".self_attn.final_linear.": ".self_attn.o_proj.",
            ".mlp.gate_up_proj.": ".mlp.dense_h_to_4h.",
            ".mlp.down_proj.": ".mlp.dense_4h_to_h.",
            ".input_layernorm.": ".input_layernorm.",
            ".post_attention_layernorm.": ".post_attention_layernorm.",
        },
        "adapter.proj.0.weight": "vit.perceive.proj.0.weight",
        "adapter.proj.2.weight": "vit.perceive.proj.2.weight",
        "adapter.mlp.weight": "vit.perceive.mlp.weight",
        "adapter.proj.0.bias": "vit.perceive.proj.0.bias",
        "adapter.proj.2.bias": "vit.perceive.proj.2.bias",
        "adapter.mlp.bias": "vit.perceive.mlp.bias",
        "adapter.after_rms.weight": "vit.perceive.after_rms.weight",
        "adapter.before_rms.weight": "vit.perceive.before_rms.weight",
        "adapter.image_begin": "vit.perceive.image_begin",
        "adapter.image_end": "vit.perceive.image_end",
        "adapter.image_newline": "vit.perceive.image_newline",
        "adapter.image_sep": "vit.perceive.image_sep",
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
        "DeepseekOCRForCausalLM": VisionTransformerLMModelConfig,
        "HunYuanVLForConditionalGeneration": VisionTransformerLMModelConfig,
    },
)

# Default tokenization transform
TOK_TABLE = defaultdict(lambda: "huggingface_tokenize")
