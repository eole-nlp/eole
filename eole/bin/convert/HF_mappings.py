from copy import deepcopy
from collections import defaultdict

from eole.config.models import (
    TransformerEncoderModelConfig,
    TransformerModelConfig,
    TransformerLMModelConfig,
    VisionTransformerLMModelConfig,
    WhisperModelConfig,
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


# ---------------------------------------------------------------------------
# MODEL_OVERRIDES: per-arch tensor-name mappings and config-building overrides.
#
# Each entry may contain:
#   • Flat string keys: tensor-name mappings merged into BASE_KEY_MAP.
#   • Sub-dict keys ("decoder", "encoder", …): per-layer key maps.
#   • "config_from_hf": callable(top, text_cfg, vis_cfg) -> dict
#     Deep-merged into the model_config built by build_config_dict in convert_HF.py.
#
# "config_from_hf" is NOT included in KEY_MAPS (filtered out in
# _LazyKeyMaps.__missing__); it is accessed directly from MODEL_OVERRIDES in
# convert_HF.py's build_config_dict.
#
# For simple (single-expression) config overrides an inline lambda is used.
# For complex multi-statement overrides a named function is defined immediately
# above the MODEL_OVERRIDES entry that uses it.  Shared helpers used by several
# arches are placed just before the first arch that needs them.
# ---------------------------------------------------------------------------

MODEL_OVERRIDES: dict = {}

MODEL_OVERRIDES["LlamaForCausalLM"] = {}  # default / base
MODEL_OVERRIDES["MistralForCausalLM"] = {}

MODEL_OVERRIDES["Qwen2ForCausalLM"] = {
    "config_from_hf": lambda top, text, vis: {
        "add_qkvbias": True,
    },
}

MODEL_OVERRIDES["Qwen3ForCausalLM"] = {
    "decoder": {
        ".self_attn.q_norm.": ".self_attn.q_norm.",
        ".self_attn.k_norm.": ".self_attn.k_norm.",
    },
    "config_from_hf": lambda top, text, vis: {
        "decoder": {
            "query_norm": True,
            "key_norm": True,
        }
    },
}

MODEL_OVERRIDES["Qwen3MoeForCausalLM"] = {
    "decoder": {
        ".self_attn.q_norm.": ".self_attn.q_norm.",
        ".self_attn.k_norm.": ".self_attn.k_norm.",
        ".mlp.gate.": ".mlp.gate.",
        **{f".mlp.experts.{i}.gate_up_proj.": f".mlp.experts.{i}.gate_proj." for i in range(128)},
        **{f".mlp.experts.{i}.up_proj.": f".mlp.experts.{i}.up_proj." for i in range(128)},
        **{f".mlp.experts.{i}.down_proj.": f".mlp.experts.{i}.down_proj." for i in range(128)},
    },
    "config_from_hf": lambda top, text, vis: {
        "decoder": {
            "query_norm": True,
            "key_norm": True,
        }
    },
}

MODEL_OVERRIDES["Gemma2ForCausalLM"] = {
    "decoder": {
        ".pre_feedforward_layernorm.": ".pre_feedforward_layernorm.",
        ".post_feedforward_layernorm.": ".post_feedforward_layernorm.",
    },
    "config_from_hf": lambda top, text, vis: {
        "share_decoder_embeddings": True,
        "ffn_layernorm": True,
        "embeddings": {
            "normalize": True,
        },
    },
}

MODEL_OVERRIDES["MixtralForCausalLM"] = {
    "decoder": {
        ".mlp.gate.": ".block_sparse_moe.gate.",
        **{
            f".mlp.experts.{i}.{attr}": f".block_sparse_moe.experts.{i}.w{j+1}."
            for i in range(8)
            for j, attr in enumerate(["gate_up_proj.", "down_proj.", "up_proj."])
        },
    },
    "config_from_hf": lambda top, text, vis: {
        "decoder": {"moe_transformer_ff": 14336},
        "moe_softmax_after": True,
        "shared_layer_norm": True,
    },
}

MODEL_OVERRIDES["PhiForCausalLM"] = {
    "decoder.layer_norm.weight": "model.final_layernorm.weight",
    "decoder.layer_norm.bias": "model.final_layernorm.bias",
    "generator.bias": "lm_head.bias",
    "decoder": {
        ".self_attn.final_linear.": ".self_attn.dense.",
        ".mlp.gate_up_proj.": ".mlp.fc1.",
        ".mlp.down_proj.": ".mlp.fc2.",
        ".input_layernorm.": (".input_layernorm.", ""),
    },
    "config_from_hf": lambda top, text, vis: {
        "parallel_residual": True,
        "shared_layer_norm": True,
        "add_qkvbias": True,
        "add_final_linear_bias": True,
        "add_ffnbias": True,
    },
}

MODEL_OVERRIDES["Phi3ForCausalLM"] = {
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
}

MODEL_OVERRIDES["PhiMoEForCausalLM"] = {
    "decoder": {
        ".mlp.gate.": ".block_sparse_moe.gate.",
        **{
            f".mlp.experts.{i}.{attr}": f".block_sparse_moe.experts.{i}.w{j+1}."
            for i in range(16)
            for j, attr in enumerate(["gate_up_proj.", "down_proj.", "up_proj."])
        },
    },
    "config_from_hf": lambda top, text, vis: {
        "decoder": {"moe_transformer_ff": 6400},
        "add_qkvbias": True,
        "add_final_linear_bias": True,
    },
}

MODEL_OVERRIDES["HunYuanDenseV1ForCausalLM"] = {
    "decoder": {
        ".self_attn.q_norm.": ".self_attn.query_layernorm.",
        ".self_attn.k_norm.": ".self_attn.key_layernorm.",
    },
    "config_from_hf": lambda top, text, vis: {
        "query_norm": True,
        "key_norm": True,
        "qk_norm_post_rope": True,
    },
}

MODEL_OVERRIDES["GPT2LMHeadModel"] = {
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
    "config_from_hf": lambda top, text, vis: {
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
}

MODEL_OVERRIDES["XLMRobertaXLForMaskedLM"] = {
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
    "config_from_hf": lambda top, text, vis: {
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
}

MODEL_OVERRIDES["LlavaForConditionalGeneration"] = {
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
    "config_from_hf": lambda top, text, vis: {
        "adapter_bias": top.get("multimodal_projector_bias", False),
        "projector_activation_fn": top.get("projector_hidden_act", "gelu"),
        "spatial_merge_size": top.get("spatial_merge_size", None),
        "encoder": {
            "mlp_activation_fn": ACT_TABLE["LlavaForConditionalGeneration"],
            "layer_norm": LN_TABLE["LlavaForConditionalGeneration"],
            "norm_eps": text.get("rms_norm_eps", text.get("layer_norm_epsilon", text.get("layer_norm_eps", 1e-5))),
            "rope_config": {
                "rotary_theta": vis["rope_theta"],
                "rotary_interleave": False,
            },
            # static encoder properties
            "num_channels": 3,
            "image_token_id": 10,
            "layernorm_pre": True,
        },
    },
}

MODEL_OVERRIDES["Mistral3ForConditionalGeneration"] = {
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
    "config_from_hf": lambda top, text, vis: {
        "adapter_bias": top.get("multimodal_projector_bias", False),
        "projector_activation_fn": top.get("projector_hidden_act", "gelu"),
        "spatial_merge_size": top.get("spatial_merge_size", None),
        "encoder": {
            "mlp_activation_fn": ACT_TABLE["Mistral3ForConditionalGeneration"],
            "layer_norm": LN_TABLE["Mistral3ForConditionalGeneration"],
            "norm_eps": text.get("rms_norm_eps", text.get("layer_norm_epsilon", text.get("layer_norm_eps", 1e-5))),
            "rope_config": {
                "rotary_theta": vis["rope_theta"],
                "rotary_interleave": False,
            },
            # static encoder properties
            "num_channels": 3,
            "image_token_id": 10,
            "layernorm_pre": True,
        },
    },
}

MODEL_OVERRIDES["Gemma3ForCausalLM"] = {
    "decoder": {
        ".self_attn.q_norm.": ".self_attn.q_norm.",
        ".self_attn.k_norm.": ".self_attn.k_norm.",
        ".pre_feedforward_layernorm.": ".pre_feedforward_layernorm.",
        ".post_feedforward_layernorm.": ".post_feedforward_layernorm.",
    },
    "config_from_hf": lambda top, text, vis: {
        "share_decoder_embeddings": True,
        "ffn_layernorm": True,
        "embeddings": {
            "normalize": True,
        },
        "decoder": {
            "query_norm": True,
            "key_norm": True,
            "rope_config": {
                "rotary_theta": 1000000,
                "rotary_theta_local": 10000,
                "interleave_local": 6,
                "tmax_index": 1,
                "rotary_interleave": False,
            },
            "max_position_embeddings": 131072,
        },
    },
}


def _build_gemma4_decoder_patch(text):
    """Build the Gemma4-specific decoder config patch from HF text config.

    Returns a partial model_config dict to be deep-merged into the full config.
    Handles layer_types, per-layer-type rope parameters, global_head_dim,
    global_heads_kv, per-layer input embedding, value_norm, kv sharing,
    and double-wide MLP.
    """
    patch = {"decoder": {}, "rope_config": {}}

    # --- 1. layer_types ---
    layer_types = text.get("layer_types", None)
    if layer_types is None:
        n_layers = text.get("num_hidden_layers", 0)
        if n_layers:
            sliding_window_pattern = 6
            layer_types = [
                "sliding_attention" if bool((i + 1) % sliding_window_pattern) else "full_attention"
                for i in range(n_layers)
            ]
            if layer_types and layer_types[-1] != "full_attention":
                layer_types[-1] = "full_attention"
    if layer_types is not None:
        patch["decoder"]["layer_types"] = layer_types

    # --- 2. rope_parameters (Gemma4 per-layer-type format) ---
    rope_params = text.get("rope_parameters", None)
    sliding_rope = None
    full_rope = None
    if isinstance(rope_params, dict) and "sliding_attention" in rope_params:
        sliding_rope = rope_params.get("sliding_attention", {})
        full_rope = rope_params.get("full_attention", {})

    # Head dimensions
    head_dim = text.get("head_dim", None)
    global_head_dim = text.get("global_head_dim", None)
    if global_head_dim is None:
        global_head_dim = head_dim

    heads_kv = text.get("num_key_value_heads", None)
    num_global_kv_heads = text.get("num_global_key_value_heads", None)
    if num_global_kv_heads is None:
        num_global_kv_heads = heads_kv

    # --- 3. Build rope_config ---
    global_theta = 1_000_000
    local_theta = 10_000
    rotary_dim_global = 0
    partial_rotary_factor = 0.0

    if full_rope:
        global_theta = int(full_rope.get("rope_theta", global_theta))
        partial_factor = full_rope.get("partial_rotary_factor", None)
        if partial_factor is not None and global_head_dim:
            partial_rotary_factor = float(partial_factor)
            rotary_dim_global = int(partial_factor * global_head_dim)
    if sliding_rope:
        local_theta = int(sliding_rope.get("rope_theta", local_theta))

    interleave_local = 6
    if layer_types:
        n_full = sum(1 for lt in layer_types if lt == "full_attention")
        n_total = len(layer_types)
        if n_full > 0:
            interleave_local = n_total // n_full

    rope_config_update = {
        "rotary_theta": global_theta,
        "rotary_theta_local": local_theta,
        "interleave_local": interleave_local,
        "tmax_index": 0,
        "rotary_interleave": False,
    }
    if partial_rotary_factor > 0.0:
        rope_config_update["partial_rotary_factor"] = partial_rotary_factor
    if rotary_dim_global > 0:
        rope_config_update["rotary_dim_global"] = rotary_dim_global

    patch["decoder"]["rope_config"] = rope_config_update
    # Also set model-level rope_config so it propagates through _override_values
    patch["rope_config"] = dict(rope_config_update)

    # --- 4. global_head_dim / global_heads_kv ---
    if global_head_dim and global_head_dim != head_dim:
        patch["decoder"]["global_head_dim"] = global_head_dim
    if num_global_kv_heads and num_global_kv_heads != heads_kv:
        patch["decoder"]["global_heads_kv"] = num_global_kv_heads

    # --- 5. per-layer input embedding ---
    hidden_size_per_layer_input = text.get("hidden_size_per_layer_input", 0)
    vocab_size_per_layer_input = text.get("vocab_size_per_layer_input", 0)
    if hidden_size_per_layer_input:
        patch["decoder"]["hidden_size_per_layer_input"] = hidden_size_per_layer_input
        patch["decoder"]["vocab_size_per_layer_input"] = vocab_size_per_layer_input

    # --- 6. value_norm ---
    patch["decoder"]["value_norm"] = True

    # --- 7. num_kv_shared_layers + use_double_wide_mlp ---
    num_kv_shared = text.get("num_kv_shared_layers", 0)
    if num_kv_shared:
        patch["decoder"]["num_kv_shared_layers"] = num_kv_shared

    use_double_wide_mlp = text.get("use_double_wide_mlp", False)
    if use_double_wide_mlp:
        patch["decoder"]["use_double_wide_mlp"] = True

    return patch


MODEL_OVERRIDES["Gemma4ForCausalLM"] = {
    "decoder.embed_tokens_per_layer.weight": "model.embed_tokens_per_layer.weight",
    "decoder.per_layer_model_projection.weight": "model.per_layer_model_projection.weight",
    "decoder.per_layer_projection_norm.weight": "model.per_layer_projection_norm.weight",
    "decoder": {
        ".self_attn.q_norm.": ".self_attn.q_norm.",
        ".self_attn.k_norm.": ".self_attn.k_norm.",
        ".pre_feedforward_layernorm.": ".pre_feedforward_layernorm.",
        ".post_feedforward_layernorm.": ".post_feedforward_layernorm.",
        ".layer_scalar": ".layer_scalar",
        ".per_layer_input_gate.": ".per_layer_input_gate.",
        ".per_layer_projection.": ".per_layer_projection.",
        ".post_per_layer_input_norm.": ".post_per_layer_input_norm.",
    },
    "config_from_hf": lambda top, text, vis: recursive_update_dict(
        {
            "share_decoder_embeddings": True,
            "ffn_layernorm": True,
            "embeddings": {"normalize": True},
            "decoder": {
                "query_norm": True,
                "key_norm": True,
                "attn_scaling": 1.0,
                "max_position_embeddings": 131072,
            },
        },
        _build_gemma4_decoder_patch(text),
        {},
    ),
}

MODEL_OVERRIDES["Gemma3ForConditionalGeneration"] = {
    "decoder_layer_prefix": "language_model.model.layers.",
    "tgt_emb.embeddings.weight": "language_model.model.embed_tokens.weight",
    "decoder.layer_norm.weight": "language_model.model.norm.weight",
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
    "config_from_hf": lambda top, text, vis: {
        "head_dim": text.get("head_dim", 256),
        "heads_kv": text.get("num_key_value_heads", 4),
        "heads": text.get("num_attention_heads", 8),
        "share_decoder_embeddings": True,
        "embeddings": {"normalize": True},
        "adapter": "gemma3",
        "decoder": {
            "ffn_layernorm": True,
            "query_norm": True,
            "key_norm": True,
            "rope_config": {
                "rotary_theta": 1000000,
                "scaling_type": "gemma3",
                "rotary_theta_local": 10000,
                "interleave_local": 6,
                "tmax_index": 1,
                "rotary_interleave": False,
            },
            "max_position_embeddings": 131072,
        },
        "encoder": {
            "n_positions": (vis["image_size"] // vis["patch_size"]) ** 2,
            "mm_tokens_per_image": top["mm_tokens_per_image"],
            "image_token_id": top["image_token_index"],
            "ffn_layernorm": False,
            "attn_scaling": None,
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
}


# Gemma4 multimodal model (text + custom vision encoder + multimodal projector).
#
# Weight layout (Gemma4ForConditionalGeneration):
#   self.model = Gemma4Model(config)
#     ├── self.language_model = Gemma4TextModel  → model.language_model.*
#     ├── self.vision_tower   = Gemma4VisionModel → model.vision_tower.*
#     └── self.embed_vision   = Gemma4MultimodalEmbedder → model.embed_vision.*
#   self.lm_head  (tied to model.language_model.embed_tokens.weight)
#
# IMPORTANT: Gemma4 uses Gemma4ClippableLinear (wraps nn.Linear as self.linear)
# for ALL vision attention and MLP projections, adding ".linear." to every path.
# The text decoder uses standard nn.Linear, so BASE_KEY_MAP decoder suffixes apply.
def _gemma4_multimodal_config_from_hf(top, text, vis):
    """Config for Gemma4ForConditionalGeneration (text + vision encoder + adapter)."""
    _v_image_size = vis.get("image_size") if vis else None
    _v_patch_size = (vis.get("patch_size", 16) if vis else 16) or 16
    _v_pos_emb_size = vis.get("position_embedding_size", 0) if vis else 0
    if _v_image_size is not None and _v_patch_size > 0:
        n_positions = (_v_image_size // _v_patch_size) ** 2
    else:
        n_positions = _v_pos_emb_size
    effective_image_size = _v_image_size or (_v_patch_size * 128)

    cfg = {
        "head_dim": text.get("head_dim", 256),
        "heads_kv": text.get("num_key_value_heads", 4),
        "heads": text.get("num_attention_heads", 8),
        "share_decoder_embeddings": True,
        "embeddings": {"normalize": True},
        "adapter": "gemma4",
        "decoder": {
            "ffn_layernorm": True,
            "query_norm": True,
            "key_norm": True,
            "attn_scaling": 1.0,
            "max_position_embeddings": 131072,
        },
        "encoder": {
            "attn_scaling": 1.0,
            "mlp_activation_fn": "gated-gelu-tanh",
            "position_encoding_type": PositionEncodingType.Rotary,
            "rope_config": {
                "rotary_theta": vis.get("rope_parameters", {}).get("rope_theta", 100.0) if vis else 100.0,
                "rotary_interleave": False,
                "multidimensional_rope": True,
            },
            "layer_norm": "rms",
            "add_ffnbias": False,
            "add_final_linear_bias": False,
            "add_qkvbias": False,
            "query_norm": True,
            "key_norm": True,
            "layernorm_pre": False,
            "layernorm_post": False,
            "patch_conv_bias": False,
            "ffn_layernorm": True,
            "pooling_kernel_size": 3,
            "n_positions": n_positions,
            "image_size": effective_image_size,
            "mm_tokens_per_image": top.get("mm_tokens_per_image", 280),
            "image_token_id": top.get("image_token_id", 258880),
            "value_norm": True,
            "norm_eps": vis.get("rms_norm_eps", 1e-6) if vis else 1e-6,
            "position_embedding_size": _v_pos_emb_size or None,
            "standardize": vis.get("standardize", False) if vis else False,
        },
    }
    return recursive_update_dict(cfg, _build_gemma4_decoder_patch(text), {})


MODEL_OVERRIDES["Gemma4ForConditionalGeneration"] = {
    # --- decoder (Gemma4TextModel inside Gemma4Model.language_model) ---
    "decoder_layer_prefix": "model.language_model.layers.",
    "tgt_emb.embeddings.weight": "model.language_model.embed_tokens.weight",
    "decoder.layer_norm.weight": "model.language_model.norm.weight",
    "decoder.embed_tokens_per_layer.weight": "model.language_model.embed_tokens_per_layer.weight",
    "decoder.per_layer_model_projection.weight": "model.language_model.per_layer_model_projection.weight",
    "decoder.per_layer_projection_norm.weight": "model.language_model.per_layer_projection_norm.weight",
    "decoder": {
        ".self_attn.q_norm.": ".self_attn.q_norm.",
        ".self_attn.k_norm.": ".self_attn.k_norm.",
        ".pre_feedforward_layernorm.": ".pre_feedforward_layernorm.",
        ".post_feedforward_layernorm.": ".post_feedforward_layernorm.",
        ".layer_scalar": ".layer_scalar",
        ".per_layer_input_gate.": ".per_layer_input_gate.",
        ".per_layer_projection.": ".per_layer_projection.",
        ".post_per_layer_input_norm.": ".post_per_layer_input_norm.",
    },
    # IMPORTANT: Gemma4VisionPatchEmbedder.input_proj is nn.Linear(3*P*P, H), not Conv2d.
    # EOLE's patch_conv is Conv2d([H, 3, P, P]).  The Linear weight must be reshaped:
    #   [H, 3*P*P] → [H, 3, P, P]
    "encoder_layer_prefix": "model.vision_tower.encoder.layers.",
    "encoder.patch_conv.weight": (
        "model.vision_tower.patch_embedder.input_proj.weight",
        # HF input_proj.weight is (H, P_h*P_w*C) where the patch pixels are in
        # (P_h, P_w, C) order (channel-last per pixel).  EOLE's Conv2d kernel
        # expects (H, C, P_h, P_w) — channel-first.  We must reshape+permute, not
        # just view, to correctly re-order channels within each patch.
        ".reshape(w.shape[0], int(round((w.shape[1]/3)**0.5)), int(round((w.shape[1]/3)**0.5)), 3).permute(0, 3, 1, 2).contiguous()",  # noqaE501
    ),
    # Gemma4 vision uses a 2D learnable position embedding table [2, P, H] in
    # Gemma4VisionPatchEmbedder; mapped to encoder.position_embedding_table.
    "encoder.position_embedding_table": "model.vision_tower.patch_embedder.position_embedding_table",
    # Gemma4VisionModel optional output standardization buffers (standardize=True).
    "adapter.std_bias": "model.vision_tower.std_bias",
    "adapter.std_scale": "model.vision_tower.std_scale",
    "encoder": {
        ".self_attn.linear_query.": ".self_attn.q_proj.linear.",
        ".self_attn.linear_keys.": ".self_attn.k_proj.linear.",
        ".self_attn.linear_values.": ".self_attn.v_proj.linear.",
        ".self_attn.final_linear.": ".self_attn.o_proj.linear.",
        ".self_attn.q_norm.": ".self_attn.q_norm.",
        ".self_attn.k_norm.": ".self_attn.k_norm.",
        ".mlp.gate_up_proj.": ".mlp.gate_proj.linear.",
        ".mlp.down_proj.": ".mlp.down_proj.linear.",
        ".mlp.up_proj.": ".mlp.up_proj.linear.",
        ".input_layernorm.": ".input_layernorm.",
        ".post_attention_layernorm.": ".post_attention_layernorm.",
        ".pre_feedforward_layernorm.": ".pre_feedforward_layernorm.",
        ".post_feedforward_layernorm.": ".post_feedforward_layernorm.",
    },
    # --- multimodal adapter (Gemma4MultimodalEmbedder inside Gemma4Model.embed_vision) ---
    # embedding_projection is nn.Linear(multimodal_hidden, text_hidden), no transpose.
    "adapter.w_in.weight": "model.embed_vision.embedding_projection.weight",
    # adapter.norm.weight has no source: Gemma4MultimodalEmbedder uses
    # Gemma4RMSNorm(with_scale=False) for pre-projection norm — no learnable weight.
    "config_from_hf": _gemma4_multimodal_config_from_hf,
}

MODEL_OVERRIDES["M2M100ForConditionalGeneration"] = {
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
    "config_from_hf": lambda top, text, vis: {
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
}


def _whisper_config_from_hf(top, text, vis):
    """Config for WhisperForConditionalGeneration."""
    max_target_positions = text.get("max_target_positions", 448)
    return {
        # Core decoder fields (Whisper uses non-standard key names)
        "layers": text.get("decoder_layers", text.get("num_hidden_layers")),
        "hidden_size": text.get("d_model"),
        "heads": text.get("decoder_attention_heads"),
        "heads_kv": text.get("decoder_attention_heads"),
        "transformer_ff": text.get("decoder_ffn_dim"),
        # Embeddings
        "embeddings": {
            "position_encoding_type": "Learned",
            "n_positions": max_target_positions,
        },
        # Static properties
        "add_qkvbias": True,
        "add_key_bias": False,
        "add_final_linear_bias": True,
        "add_ffnbias": True,
        "left_pad": False,
        "share_decoder_embeddings": True,
        "generator_bias": False,
        # Encoder config
        "encoder": {
            "layers": text.get("encoder_layers"),
            "hidden_size": text.get("d_model"),
            "heads": text.get("encoder_attention_heads"),
            "heads_kv": text.get("encoder_attention_heads"),
            "transformer_ff": text.get("encoder_ffn_dim"),
            "num_mel_bins": text.get("num_mel_bins", 80),
            "max_source_positions": text.get("max_source_positions", 1500),
            "layer_norm": "standard",
            "norm_eps": 1e-5,
            "mlp_activation_fn": "gelu",
            "add_qkvbias": True,
            "add_key_bias": False,
            "add_final_linear_bias": True,
            "add_ffnbias": True,
            "position_encoding_type": None,
        },
    }


MODEL_OVERRIDES["WhisperForConditionalGeneration"] = {
    "decoder_layer_prefix": "model.decoder.layers.",
    "encoder_layer_prefix": "model.encoder.layers.",
    "tgt_emb.embeddings.weight": "model.decoder.embed_tokens.weight",
    "tgt_emb.pe.weight": "model.decoder.embed_positions.weight",
    "decoder.layer_norm.weight": "model.decoder.layer_norm.weight",
    "decoder.layer_norm.bias": "model.decoder.layer_norm.bias",
    "encoder.layer_norm.weight": "model.encoder.layer_norm.weight",
    "encoder.layer_norm.bias": "model.encoder.layer_norm.bias",
    "encoder.conv1.weight": "model.encoder.conv1.weight",
    "encoder.conv1.bias": "model.encoder.conv1.bias",
    "encoder.conv2.weight": "model.encoder.conv2.weight",
    "encoder.conv2.bias": "model.encoder.conv2.bias",
    "encoder.embed_positions.weight": "model.encoder.embed_positions.weight",
    "generator.weight": "proj_out.weight",
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
    "config_from_hf": _whisper_config_from_hf,
}


def _deepseek_ocr_config_from_hf(top, text, vis):
    """Config for DeepseekOCRForCausalLM."""
    clip_cfg = (vis or {}).get("width", {}).get("clip-l-14-224", {})
    img_size = clip_cfg.get("image_size", 224)
    patch_size = clip_cfg.get("patch_size", 14)
    heads = clip_cfg.get("heads", None)
    n_pos = (img_size // patch_size) ** 2 + 1
    return {
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
            "encoder_sam": True,
            "use_class_embedding": True,
            "n_positions": n_pos,
            "patch_size": patch_size,
            "heads": heads,
            "heads_kv": heads,
        },
    }


MODEL_OVERRIDES["DeepseekOCRForCausalLM"] = {
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
    "config_from_hf": _deepseek_ocr_config_from_hf,
}

MODEL_OVERRIDES["HunYuanVLForConditionalGeneration"] = {
    "decoder": {
        ".self_attn.q_norm.": ".self_attn.query_layernorm.",
        ".self_attn.k_norm.": ".self_attn.key_layernorm.",
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
    "config_from_hf": lambda top, text, vis: {
        "spatial_merge_size": (vis or {}).get("spatial_merge_size", None),
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
            "image_token_id_list": [120120, 120118, 120119],
        },
    },
}


# Shared helper: vision encoder patch for the Qwen3VL / Qwen3.5VL family.
# Used by Qwen3VLForConditionalGeneration, Qwen3_5ForConditionalGeneration, and
# Qwen3_5MoeForConditionalGeneration.
def _qwen3vl_encoder_patch(top, vis):
    """Shared vision encoder patch for the Qwen3VL/Qwen3.5VL family."""
    vis = vis or {}
    num_pos_embed = vis.get("num_position_embeddings", 0)
    patch_size = vis.get("patch_size", 16)
    num_heads = vis.get("num_heads", 16)
    img_size = int(num_pos_embed**0.5) * patch_size if num_pos_embed else 0
    return {
        "spatial_merge_size": vis.get("spatial_merge_size", 2),
        "encoder": {
            "heads": num_heads,
            "heads_kv": num_heads,
            "head_dim": vis.get("hidden_size", 0) // num_heads if num_heads else 0,
            "layers": vis.get("depth", vis.get("num_hidden_layers", 27)),
            "image_size": img_size,
            "num_position_embeddings": num_pos_embed,
            "position_encoding_type": PositionEncodingType.Rotary,
            "rope_config": {
                "rotary_interleave": False,
                "rotary_theta": 10000,
            },
            "num_channels": vis.get("in_channels", 3),
            "image_token_id": top.get("image_token_id", 151655),
        },
    }


# Shared helper: hybrid decoder fields (layer_types + GatedDeltaNet hyper-params)
# for the Qwen3.5 text/VL/MoE family.
def _qwen35_decoder_fields(text):
    """Hybrid decoder fields (layer_types + GatedDeltaNet hyper-params) for Qwen3.5."""
    decoder = {}
    layer_types = text.get("layer_types")
    if layer_types is not None:
        decoder["layer_types"] = layer_types
    for key in [
        "linear_conv_kernel_dim",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
    ]:
        val = text.get(key)
        if val is not None:
            decoder[key] = val
    return decoder


def _qwen35_vl_config_from_hf(top, text, vis):
    """Config for Qwen3_5ForConditionalGeneration."""
    decoder = {"query_norm": True, "key_norm": True, "q_gating": True}
    decoder.update(_qwen35_decoder_fields(text))
    cfg = {
        "adapter": "qwen3_5vl",
        "decoder": decoder,
        "encoder": {
            "layer_norm": "standard",
            "mlp_activation_fn": "gelu-tanh",
            "add_ffnbias": True,
            "add_final_linear_bias": True,
            "add_qkvbias": True,
            "layernorm_pre": False,
            "layernorm_post": False,
            "temporal_patch_size": 2,
            "patch_conv_bias": True,
        },
    }
    return recursive_update_dict(cfg, _qwen3vl_encoder_patch(top, vis), {})


MODEL_OVERRIDES["Qwen3_5ForConditionalGeneration"] = {
    "decoder_layer_prefix": "model.language_model.layers.",
    "tgt_emb.embeddings.weight": "model.language_model.embed_tokens.weight",
    "decoder.layer_norm.weight": "model.language_model.norm.weight",
    "generator.weight": "lm_head.weight",
    # Vision encoder (same structure as Qwen3VL)
    "encoder_layer_prefix": "model.visual.blocks.",
    "encoder.patch_conv.weight": "model.visual.patch_embed.proj.weight",
    "encoder.patch_conv.bias": "model.visual.patch_embed.proj.bias",
    "encoder.pos_embed.weight": "model.visual.pos_embed.weight",
    "encoder": {
        # Fused QKV → split into Q / K / V
        ".self_attn.linear_query.weight": (".attn.qkv.weight", "[:hidden_size, :]"),
        ".self_attn.linear_keys.weight": (".attn.qkv.weight", "[hidden_size:2*hidden_size, :]"),
        ".self_attn.linear_values.weight": (".attn.qkv.weight", "[-hidden_size:, :]"),
        ".self_attn.linear_query.bias": (".attn.qkv.bias", "[:hidden_size]"),
        ".self_attn.linear_keys.bias": (".attn.qkv.bias", "[hidden_size:2*hidden_size]"),
        ".self_attn.linear_values.bias": (".attn.qkv.bias", "[-hidden_size:]"),
        ".self_attn.final_linear.weight": ".attn.proj.weight",
        ".self_attn.final_linear.bias": ".attn.proj.bias",
        # Vision MLP: linear_fc1 → gate_up_proj, linear_fc2 → down_proj
        ".mlp.gate_up_proj.": ".mlp.linear_fc1.",
        ".mlp.down_proj.": ".mlp.linear_fc2.",
        # Layer norms
        ".input_layernorm.": ".norm1.",
        ".post_attention_layernorm.": ".norm2.",
    },
    # Vision merger (adapter)
    "adapter.norm.weight": "model.visual.merger.norm.weight",
    "adapter.norm.bias": "model.visual.merger.norm.bias",
    "adapter.linear_fc1.weight": "model.visual.merger.linear_fc1.weight",
    "adapter.linear_fc1.bias": "model.visual.merger.linear_fc1.bias",
    "adapter.linear_fc2.weight": "model.visual.merger.linear_fc2.weight",
    "adapter.linear_fc2.bias": "model.visual.merger.linear_fc2.bias",
    # Decoder layers (Qwen3.5 hybrid text model — same as Qwen3_5TextForCausalLM)
    "decoder": {
        ".self_attn.q_norm.": ".self_attn.q_norm.",
        ".self_attn.k_norm.": ".self_attn.k_norm.",
        ".linear_attn.in_proj_qkv.": ".linear_attn.in_proj_qkv.",
        ".linear_attn.in_proj_z.": ".linear_attn.in_proj_z.",
        ".linear_attn.in_proj_b.": ".linear_attn.in_proj_b.",
        ".linear_attn.in_proj_a.": ".linear_attn.in_proj_a.",
        ".linear_attn.conv1d.": ".linear_attn.conv1d.",
        ".linear_attn.dt_bias": ".linear_attn.dt_bias",
        ".linear_attn.A_log": ".linear_attn.A_log",
        ".linear_attn.norm.": ".linear_attn.norm.",
        ".linear_attn.out_proj.": ".linear_attn.out_proj.",
    },
    "config_from_hf": _qwen35_vl_config_from_hf,
}


def _qwen35_moe_vl_config_from_hf(top, text, vis):
    """Config for Qwen3_5MoeForConditionalGeneration."""
    decoder = {
        "query_norm": True,
        "key_norm": True,
        "q_gating": True,
        "shared_expert_gate": True,
        "moe_renormalize": True,
    }
    decoder.update(_qwen35_decoder_fields(text))
    cfg = {
        "adapter": "qwen3_5vl",
        "decoder": decoder,
        "encoder": {
            "layer_norm": "standard",
            "mlp_activation_fn": "gelu-tanh",
            "add_ffnbias": True,
            "add_final_linear_bias": True,
            "add_qkvbias": True,
            "layernorm_pre": False,
            "layernorm_post": False,
            "temporal_patch_size": 2,
            "patch_conv_bias": True,
        },
    }
    # Qwen3.5 MoE: use shared_expert_intermediate_size as shared expert FF size
    if text.get("shared_expert_intermediate_size"):
        moe_ff = text.get("moe_intermediate_size")
        cfg["moe_transformer_ff"] = moe_ff
        cfg["decoder"]["moe_transformer_ff"] = moe_ff
    return recursive_update_dict(cfg, _qwen3vl_encoder_patch(top, vis), {})


MODEL_OVERRIDES["Qwen3_5MoeForConditionalGeneration"] = {
    "decoder_layer_prefix": "model.language_model.layers.",
    "tgt_emb.embeddings.weight": "model.language_model.embed_tokens.weight",
    "decoder.layer_norm.weight": "model.language_model.norm.weight",
    "generator.weight": "lm_head.weight",
    # Vision encoder (identical to Qwen3.5 VL)
    "encoder_layer_prefix": "model.visual.blocks.",
    "encoder.patch_conv.weight": "model.visual.patch_embed.proj.weight",
    "encoder.patch_conv.bias": "model.visual.patch_embed.proj.bias",
    "encoder.pos_embed.weight": "model.visual.pos_embed.weight",
    "encoder": {
        ".self_attn.linear_query.weight": (".attn.qkv.weight", "[:hidden_size, :]"),
        ".self_attn.linear_keys.weight": (".attn.qkv.weight", "[hidden_size:2*hidden_size, :]"),
        ".self_attn.linear_values.weight": (".attn.qkv.weight", "[-hidden_size:, :]"),
        ".self_attn.linear_query.bias": (".attn.qkv.bias", "[:hidden_size]"),
        ".self_attn.linear_keys.bias": (".attn.qkv.bias", "[hidden_size:2*hidden_size]"),
        ".self_attn.linear_values.bias": (".attn.qkv.bias", "[-hidden_size:]"),
        ".self_attn.final_linear.weight": ".attn.proj.weight",
        ".self_attn.final_linear.bias": ".attn.proj.bias",
        ".mlp.gate_up_proj.": ".mlp.linear_fc1.",
        ".mlp.down_proj.": ".mlp.linear_fc2.",
        ".input_layernorm.": ".norm1.",
        ".post_attention_layernorm.": ".norm2.",
    },
    "adapter.norm.weight": "model.visual.merger.norm.weight",
    "adapter.norm.bias": "model.visual.merger.norm.bias",
    "adapter.linear_fc1.weight": "model.visual.merger.linear_fc1.weight",
    "adapter.linear_fc1.bias": "model.visual.merger.linear_fc1.bias",
    "adapter.linear_fc2.weight": "model.visual.merger.linear_fc2.weight",
    "adapter.linear_fc2.bias": "model.visual.merger.linear_fc2.bias",
    # MoE decoder layers
    "decoder": {
        ".self_attn.q_norm.": ".self_attn.q_norm.",
        ".self_attn.k_norm.": ".self_attn.k_norm.",
        ".linear_attn.in_proj_qkv.": ".linear_attn.in_proj_qkv.",
        ".linear_attn.in_proj_z.": ".linear_attn.in_proj_z.",
        ".linear_attn.in_proj_b.": ".linear_attn.in_proj_b.",
        ".linear_attn.in_proj_a.": ".linear_attn.in_proj_a.",
        ".linear_attn.conv1d.": ".linear_attn.conv1d.",
        ".linear_attn.dt_bias": ".linear_attn.dt_bias",
        ".linear_attn.A_log": ".linear_attn.A_log",
        ".linear_attn.norm.": ".linear_attn.norm.",
        ".linear_attn.out_proj.": ".linear_attn.out_proj.",
        ".mlp.gate.weight": ".mlp.gate.weight",
        **{
            f".mlp.experts.{j}.gate_up_proj.weight": (
                ".mlp.experts.gate_up_proj",
                f"[{j}, :moe_transformer_ff, :]",
            )
            for j in range(256)
        },
        **{
            f".mlp.experts.{j}.up_proj.weight": (
                ".mlp.experts.gate_up_proj",
                f"[{j}, moe_transformer_ff:, :]",
            )
            for j in range(256)
        },
        **{f".mlp.experts.{j}.down_proj.weight": (".mlp.experts.down_proj", f"[{j}]") for j in range(256)},
        # Quantized (AutoRound): per-expert storage — gate_proj/up_proj/down_proj stored separately.
        # FP16 layers listed in extra_config (e.g. shared_expert) naturally fall through to
        # param="weight" because qweight/qzeros/scales are absent for those tensors.
        **{f".mlp.experts.{j}.gate_up_proj.": f".mlp.experts.{j}.gate_proj." for j in range(256)},
        **{f".mlp.experts.{j}.up_proj.": f".mlp.experts.{j}.up_proj." for j in range(256)},
        **{f".mlp.experts.{j}.down_proj.": f".mlp.experts.{j}.down_proj." for j in range(256)},
        ".mlp.shared_experts.gate_up_proj.": ".mlp.shared_expert.gate_proj.",
        ".mlp.shared_experts.up_proj.": ".mlp.shared_expert.up_proj.",
        ".mlp.shared_experts.down_proj.": ".mlp.shared_expert.down_proj.",
        ".mlp.shared_expert_gate.weight": ".mlp.shared_expert_gate.weight",
    },
    "config_from_hf": _qwen35_moe_vl_config_from_hf,
}

# Text-only Qwen3.5 variants (hybrid transformer+linear-attention decoder)
MODEL_OVERRIDES["Qwen3_5TextForCausalLM"] = {
    "config_from_hf": lambda top, text, vis: {
        "decoder": {"query_norm": True, "key_norm": True, "q_gating": True, **_qwen35_decoder_fields(text)}
    },
}

MODEL_OVERRIDES["Qwen3_5MoeForCausalLM"] = {
    "config_from_hf": lambda top, text, vis: {
        "decoder": {"query_norm": True, "key_norm": True, "q_gating": True, **_qwen35_decoder_fields(text)}
    },
}

# Qwen3 VL (text-only Qwen3VL variant without Mamba layers)
MODEL_OVERRIDES["Qwen3VLForConditionalGeneration"] = {
    "config_from_hf": lambda top, text, vis: _qwen3vl_encoder_patch(top, vis),
}


class _LazyKeyMaps(dict):
    def __missing__(self, arch):
        # Exclude "config_from_hf": it is a config-building callable, not a tensor
        # name mapping, and must not appear in KEY_MAPS iterations.
        overrides = {k: v for k, v in MODEL_OVERRIDES.get(arch, {}).items() if k != "config_from_hf"}
        value = recursive_update_dict(deepcopy(BASE_KEY_MAP), overrides, {}, silent=True)
        self[arch] = value
        return value

    def __contains__(self, arch):
        # All known arches are "in" the map, even if not yet built
        return arch in MODEL_OVERRIDES


"""
    def __getitem__(self, arch):
        # Always return a fresh copy so callers can't corrupt the cache
        return deepcopy(super().__getitem__(arch) if arch in self else self.__missing__(arch))
"""

KEY_MAPS = _LazyKeyMaps()


# Layer norm type
LN_TABLE = defaultdict(
    lambda: "rms",
    {
        "PhiForCausalLM": "standard",
        "GPT2LMHeadModel": "standard",
        "XLMRobertaXLForMaskedLM": "standard",
        "Gemma2ForCausalLM": "gemma-rms",
        "Gemma3ForCausalLM": "gemma-rms",
        "M2M100ForConditionalGeneration": "standard",
        "WhisperForConditionalGeneration": "standard",
        "Gemma3ForConditionalGeneration": "gemma-rms",
        "Gemma4ForCausalLM": "rms",
        "Gemma4ForConditionalGeneration": "rms",
        "Qwen3_5ForConditionalGeneration": "gemma-rms",
        "Qwen3_5MoeForConditionalGeneration": "gemma-rms",
        "Qwen3_5TextForCausalLM": "gemma-rms",
        "Qwen3_5MoeForCausalLM": "gemma-rms",
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
        "Gemma3ForCausalLM": "gated-gelu-tanh",
        "Gemma3ForConditionalGeneration": "gated-gelu-tanh",
        "Gemma4ForCausalLM": "gated-gelu-tanh",
        "Gemma4ForConditionalGeneration": "gated-gelu-tanh",
        "M2M100ForConditionalGeneration": "relu",
        "WhisperForConditionalGeneration": "gelu",
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
        "Gemma4ForConditionalGeneration": VisionTransformerLMModelConfig,
        "M2M100ForConditionalGeneration": TransformerModelConfig,
        "WhisperForConditionalGeneration": WhisperModelConfig,
        "DeepseekOCRForCausalLM": VisionTransformerLMModelConfig,
        "HunYuanVLForConditionalGeneration": VisionTransformerLMModelConfig,
        "Qwen3VLForConditionalGeneration": VisionTransformerLMModelConfig,
        "Qwen3_5ForConditionalGeneration": VisionTransformerLMModelConfig,
        "Qwen3_5MoeForConditionalGeneration": VisionTransformerLMModelConfig,
    },
)

# Default tokenization transform
TOK_TABLE = defaultdict(lambda: "huggingface_tokenize")
