#!/usr/bin/env python
"""Convert a GGUF model file to EOLE format.

GGUF (llama.cpp) models often store weights in quantized formats (Q4_K, Q5_K,
Q8_0, …).  This converter follows the same philosophy as the AutoRound/GPTQ
backend: quantized weights are kept in their compact binary representation and
stored as ``uint8`` tensors in the output safetensors shard.  At inference time
:class:`~eole.modules.gguf_linear.GGUFLinear` dequantizes them on-the-fly,
achieving the same on-disk storage savings without requiring a separate
dequantization step during conversion.

Float-typed tensors (F16, BF16, F32) are stored as-is, cast to the target dtype
requested via ``--dtype``.

Requires:
  pip install gguf safetensors pyonmttok

Usage example::

    eole convert GGUF \\
        --gguf_path  /path/to/model-Q4_K_M.gguf \\
        --output     /path/to/output_dir \\
        --dtype      fp16
"""

# Standard Library Imports
import json
import os
import re
from typing import Optional

# Third-Party Library Imports
import torch

# Eole Imports
from eole.bin import BaseBin, register_bin
from eole.config import recursive_model_fields_set
from eole.config.run import TrainConfig
from eole.config.training import TrainingConfig
from eole.constants import DefaultTokens, TORCH_DTYPES, PositionEncodingType
from eole.inputters.inputter import vocabs_to_dict

try:
    import pyonmttok
except ImportError:
    pyonmttok = None

try:
    from safetensors.torch import save_file as safetensors_save_file
except ImportError:
    safetensors_save_file = None


# ---------------------------------------------------------------------------
# Float types – linear layers with these types stay as nn.Linear (no GGUFLinear)
# ---------------------------------------------------------------------------

_FLOAT_TYPE_NAMES = frozenset({"F32", "F16", "BF16", "F64", "I8", "I16", "I32", "I64"})


def _is_float_type(tensor_type_name: str) -> bool:
    return tensor_type_name in _FLOAT_TYPE_NAMES


# ---------------------------------------------------------------------------
# GGUF tensor-name → EOLE tensor-name mapping
# ---------------------------------------------------------------------------

# Global tensors (not inside a transformer block)
_GLOBAL_MAP: dict[str, Optional[str]] = {
    "token_embd.weight": "tgt_emb.embeddings.weight",
    "output_norm.weight": "decoder.layer_norm.weight",
    "output_norm.bias": "decoder.layer_norm.bias",
    "output.weight": "generator.weight",
    "output.bias": "generator.bias",
    # RoPE frequency cache – EOLE recomputes this at runtime
    "rope_freqs.weight": None,
    "rope_factors_long.weight": None,
    "rope_factors_short.weight": None,
}

# EOLE weight names that must always be dequantized to float, even when the
# rest of the model stays in quantized (uint8) form.
#
# These correspond to modules that are NOT replaced by GGUFLinear at load time:
#  • nn.Embedding layers (tgt_emb / src_emb) – cannot hold uint8 at all.
#  • The output projection / generator – when a vision encoder is present,
#    replace_gguf_linear targets only self.decoder, not self.generator, so the
#    generator remains a plain nn.Linear and cannot load uint8 weights.
#
# For plain text models (no vision encoder) the generator IS reachable by
# replace_gguf_linear (quant_target = self), so generator.weight can stay
# quantized.  build_safetensors accepts a has_vision_encoder flag and adds
# "generator.weight" to the active must-dequantize set only when needed.
_MUST_DEQUANTIZE_NAMES_BASE: frozenset[str] = frozenset(
    {
        "tgt_emb.embeddings.weight",
        "src_emb.embeddings.weight",
    }
)
# Used for models where the generator cannot be GGUFLinear (VLM / any model
# where replace_gguf_linear does not reach self.generator).
_MUST_DEQUANTIZE_NAMES_WITH_GENERATOR: frozenset[str] = frozenset(
    {
        "tgt_emb.embeddings.weight",
        "src_emb.embeddings.weight",
        "generator.weight",
    }
)
# Keep backwards-compatible module-level name (used in tests).
_MUST_DEQUANTIZE_NAMES = _MUST_DEQUANTIZE_NAMES_WITH_GENERATOR

# Per-block suffix → EOLE decoder-layer suffix.
# A value of None means "known tensor but not yet modelled in EOLE – skip silently".
_BLOCK_MAP: dict[str, Optional[str]] = {
    # Norms
    "attn_norm.weight": "input_layernorm.weight",
    "attn_norm.bias": "input_layernorm.bias",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn_norm.bias": "post_attention_layernorm.bias",
    # Per-layer norms for Gemma2/3
    "pre_feedforward_layernorm.weight": "pre_feedforward_layernorm.weight",
    "post_feedforward_layernorm.weight": "post_feedforward_layernorm.weight",
    # Q-norm / K-norm (Qwen3/Qwen3.5 etc.)
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    # Attention projections (separate Q/K/V)
    "attn_q.weight": "self_attn.linear_query.weight",
    "attn_q.bias": "self_attn.linear_query.bias",
    "attn_k.weight": "self_attn.linear_keys.weight",
    "attn_k.bias": "self_attn.linear_keys.bias",
    "attn_v.weight": "self_attn.linear_values.weight",
    "attn_v.bias": "self_attn.linear_values.bias",
    "attn_output.weight": "self_attn.final_linear.weight",
    "attn_output.bias": "self_attn.final_linear.bias",
    # Combined QKV projection (Phi-2, Phi-3 style)
    "attn_qkv.weight": "self_attn.qkv_proj.weight",
    "attn_qkv.bias": "self_attn.qkv_proj.bias",
    # Feed-forward (SwiGLU: gate + up + down)
    "ffn_gate.weight": "mlp.gate_up_proj.weight",
    "ffn_gate.bias": "mlp.gate_up_proj.bias",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_up.bias": "mlp.up_proj.bias",
    "ffn_down.weight": "mlp.down_proj.weight",
    "ffn_down.bias": "mlp.down_proj.bias",
    # Gate + Up fused (Phi-3 style)
    "ffn_gate_up.weight": "mlp.gate_up_proj.weight",
    "ffn_gate_up.bias": "mlp.gate_up_proj.bias",
    # MoE router gate
    "ffn_gate_inp.weight": "mlp.gate.weight",
    # MoE per-expert weights (stacked tensors)
    "ffn_gate_exps.weight": "mlp.experts_gate_up.weight",
    "ffn_up_exps.weight": "mlp.experts_up.weight",
    "ffn_down_exps.weight": "mlp.experts_down.weight",
    # Post-attention output norm – present in full-attention and linear-attention
    # blocks (Qwen3.5 and similar hybrid models).
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "post_attention_norm.bias": "post_attention_layernorm.bias",
    # Gate / z projection for GatedDeltaNet linear-attention blocks (in_proj_z)
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "attn_gate.bias": "linear_attn.in_proj_z.bias",
    # GatedDeltaNet SSM weights (linear-attention / hybrid models, e.g. Qwen3.5)
    # NOTE: llama.cpp's Qwen3NextModel.modify_tensors converts A_log → -exp(A_log)
    # before writing ssm_a to GGUF.  build_safetensors inverts this with log(-t).
    "ssm_a": "linear_attn.A_log",  # log-scale A decay factor (needs inversion)
    "ssm_a.weight": "linear_attn.A_log",  # alternative naming in some GGUF files
    # ssm_alpha.weight / ssm_beta.weight are the in_proj_a / in_proj_b linear
    # projections.  gguf-python reverses GGUF ne when creating tensor.data, so
    # the numpy array has shape (out_features, in_features) matching the
    # standard PyTorch/HF convention.  However, llama.cpp _LinearAttentionVReorderBase
    # reorders V heads to tiled order when num_k_heads != num_v_heads; we undo
    # that in build_safetensors via _inverse_v_head_reorder_rows.
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_conv1d.bias": "linear_attn.conv1d.bias",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
}

# ---------------------------------------------------------------------------
# V-head reordering helpers
# ---------------------------------------------------------------------------
#
# llama.cpp's _LinearAttentionVReorderBase (used for Qwen3.5 / Qwen3.5-MoE)
# reorders V heads from **grouped-by-K-head** order (HF convention) to
# **tiled** order when writing GGUF:
#
#   HF   grouped: [K0v0, K0v1, …, K0v{r-1},  K1v0, K1v1, …]
#   GGUF   tiled: [K0v0, K1v0, …, Kn v0,    K0v1, K1v1, …]
#   (r = num_v_per_k = num_v_heads // num_k_heads)
#
# This only applies when num_k_heads != num_v_heads.  Our reverse converter
# must undo this reordering (tiled → grouped) so the weights match the format
# that eole's GatedDeltaNet module expects (identical to HF convention).
#
# References:
#   https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py
#   class _LinearAttentionVReorderBase


def _inverse_v_head_reorder_1d(t: "torch.Tensor", num_k_heads: int, num_v_per_k: int) -> "torch.Tensor":
    """Inverse V-head reordering for 1D tensors (A_log, dt_bias).

    Converts tiled order ``[K0v0, K1v0, …, K0v1, K1v1, …]`` back to grouped
    order ``[K0v0, K0v1, …, K1v0, K1v1, …]``.
    """
    if num_v_per_k <= 1:
        return t
    num_v_heads = num_k_heads * num_v_per_k
    return t.view(num_v_per_k, num_k_heads).permute(1, 0).contiguous().view(num_v_heads)


def _inverse_v_head_reorder_rows(
    t: "torch.Tensor", num_k_heads: int, num_v_per_k: int, head_dim: int
) -> "torch.Tensor":
    """Inverse V-head row reordering for 2D tensors.

    Works on both float tensors ``(total_rows, n_cols)`` and uint8 quantized
    tensors ``(total_rows, bytes_per_row)`` where ``total_rows = num_v_heads *
    head_dim``.

    ``head_dim`` is:
    * 1 for ``in_proj_a`` / ``in_proj_b`` (one scalar output per V head)
    * ``head_v_dim`` for ``in_proj_z`` and the V-part of ``in_proj_qkv``
    """
    if num_v_per_k <= 1:
        return t
    total_rows, cols = t.shape
    # num_v_heads = num_k_heads * num_v_per_k
    # Tiled layout stored in GGUF:  (num_v_per_k, num_k_heads, head_dim, cols)
    # Target grouped layout (HF):   (num_k_heads, num_v_per_k, head_dim, cols)
    t = t.view(num_v_per_k, num_k_heads, head_dim, cols)
    t = t.permute(1, 0, 2, 3).contiguous()
    t = t.view(total_rows, cols)
    return t


def _inverse_v_head_reorder_cols(t: "torch.Tensor", num_k_heads: int, num_v_per_k: int) -> "torch.Tensor":
    """Inverse V-head column reordering for 2D tensors (out_proj.weight).

    Works on both float tensors ``(n_rows, total_cols)`` and uint8 tensors when
    ``total_cols`` is divisible by ``num_v_heads`` (i.e. each V head's columns
    are block-aligned, which holds when ``head_v_dim % 32 == 0``).
    """
    if num_v_per_k <= 1:
        return t
    num_v_heads = num_k_heads * num_v_per_k
    n_rows, total_cols = t.shape
    if total_cols % num_v_heads != 0:
        raise ValueError(
            f"Column count {total_cols} not divisible by num_v_heads={num_v_heads}; "
            "cannot apply V-head column reordering"
        )
    cols_per_v_head = total_cols // num_v_heads
    # Tiled: (n_rows, num_v_per_k, num_k_heads, cols_per_v_head)
    # Grouped: (n_rows, num_k_heads, num_v_per_k, cols_per_v_head)
    t = t.view(n_rows, num_v_per_k, num_k_heads, cols_per_v_head)
    t = t.permute(0, 2, 1, 3).contiguous()
    t = t.view(n_rows, total_cols)
    return t


def _gguf_to_eole_name(gguf_name: str, linear_blocks: frozenset = frozenset()) -> Optional[str]:
    """Map a GGUF tensor name to the EOLE tensor name.

    Args:
        gguf_name: The name of the tensor in the GGUF file.
        linear_blocks: Set of block indices that are linear-attention (SSM/GatedDeltaNet)
            blocks, as detected by :func:`_detect_linear_attention_blocks`.  Only
            used to disambiguate ``attn_qkv.*`` which maps to ``linear_attn.in_proj_qkv``
            in linear-attention blocks and ``self_attn.qkv_proj`` otherwise.

    Returns:
        ``None`` for tensors that should be skipped (e.g., RoPE cache).
        The empty string ``""`` for unrecognised names (caller should warn).
        Otherwise the EOLE tensor name string.
    """
    if gguf_name in _GLOBAL_MAP:
        return _GLOBAL_MAP[gguf_name]

    m = re.fullmatch(r"blk\.(\d+)\.(.*)", gguf_name)
    if m:
        block_idx = int(m.group(1))
        suffix = m.group(2)

        # attn_qkv.* maps differently depending on block type: in linear-attention
        # blocks the merged QKV projection feeds GatedDeltaNet (in_proj_qkv), while
        # in full-attention blocks it feeds self_attn (qkv_proj).
        if block_idx in linear_blocks and suffix in ("attn_qkv.weight", "attn_qkv.bias"):
            param = suffix.split(".", 1)[1]  # "weight" or "bias"
            return f"decoder.transformer_layers.{block_idx}.linear_attn.in_proj_qkv.{param}"

        if suffix not in _BLOCK_MAP:
            return ""  # unrecognised
        eole_suffix = _BLOCK_MAP[suffix]
        if eole_suffix is None:
            return None  # known but not yet supported in EOLE, skip silently
        return f"decoder.transformer_layers.{block_idx}.{eole_suffix}"

    return ""  # unrecognised


# ---------------------------------------------------------------------------
# Hybrid-model helpers
# ---------------------------------------------------------------------------


def _detect_linear_attention_blocks(tensors) -> frozenset:
    """Return the set of block indices that contain SSM / linear-attention weights.

    Linear-attention blocks in hybrid models (Qwen3.5, …) always have a ``ssm_a``
    tensor (the GatedDeltaNet log-scale A decay factor).  We use this as a reliable
    sentinel to distinguish them from standard full-attention blocks.
    """
    linear_blocks: set[int] = set()
    for tensor in tensors:
        m = re.fullmatch(r"blk\.(\d+)\.ssm_.*", tensor.name)
        if m:
            linear_blocks.add(int(m.group(1)))
    return frozenset(linear_blocks)


def _infer_linear_attn_config(meta, linear_blocks: frozenset) -> dict:
    """Infer :class:`~eole.modules.gated_delta_net.GatedDeltaNet` hyper-parameters.

    Parameters are read from GGUF metadata when available (preferred), and
    fall back to inference from tensor shapes.  The returned dict contains
    the keys expected by ``TransformerDecoderConfig`` (``layer_types``,
    ``linear_num_value_heads``, ``linear_value_head_dim``,
    ``linear_num_key_heads``, ``linear_key_head_dim``,
    ``linear_conv_kernel_dim``).

    GGUF metadata mapping (written by llama.cpp's Qwen3NextModel):
      ``{arch}.ssm.state_size``     → ``linear_key_head_dim``
      ``{arch}.ssm.group_count``    → ``linear_num_key_heads``
      ``{arch}.ssm.time_step_rank`` → ``linear_num_value_heads``
      ``{arch}.ssm.inner_size``     → ``linear_value_head_dim * linear_num_value_heads``
      ``{arch}.ssm.conv_kernel``    → ``linear_conv_kernel_dim``
    """
    if not linear_blocks:
        return {}

    cfg: dict = {}

    arch = meta.arch  # e.g. "qwen35", "qwen3next"

    # ---------- Read from GGUF metadata (preferred) -----------------------
    # llama.cpp's Qwen3NextModel.set_gguf_parameters writes these keys.
    meta_state_size = meta._scalar(f"{arch}.ssm.state_size")  # = linear_key_head_dim
    meta_group_count = meta._scalar(f"{arch}.ssm.group_count")  # = linear_num_key_heads
    meta_time_step = meta._scalar(f"{arch}.ssm.time_step_rank")  # = linear_num_value_heads
    meta_inner_size = meta._scalar(f"{arch}.ssm.inner_size")  # = value_dim
    meta_conv_kernel = meta._scalar(f"{arch}.ssm.conv_kernel")  # = conv kernel size

    if meta_state_size is not None:
        cfg["linear_key_head_dim"] = int(meta_state_size)
    if meta_group_count is not None:
        cfg["linear_num_key_heads"] = int(meta_group_count)
    if meta_time_step is not None:
        cfg["linear_num_value_heads"] = int(meta_time_step)
    if meta_inner_size is not None and meta_time_step is not None and int(meta_time_step) > 0:
        cfg["linear_value_head_dim"] = int(meta_inner_size) // int(meta_time_step)
    if meta_conv_kernel is not None:
        cfg["linear_conv_kernel_dim"] = int(meta_conv_kernel)

    # ---------- Fall back to shape inference for missing fields -----------
    block_idx = min(linear_blocks)
    shapes: dict[str, list] = {t.name: list(t.shape) for t in meta.tensors}

    if "linear_num_value_heads" not in cfg:
        # ssm_a / ssm_dt.bias have shape (num_v_heads,) in the GGUF file.
        # GGUFReader tensor.shape is the GGUF ne array (reversed from numpy).
        # For a 1-D tensor, ne = [num_v_heads], so shape[0] = num_v_heads.
        ssm_a_shape = shapes.get(f"blk.{block_idx}.ssm_a") or shapes.get(f"blk.{block_idx}.ssm_dt.bias")
        if ssm_a_shape:
            cfg["linear_num_value_heads"] = int(ssm_a_shape[0])

    if "linear_value_head_dim" not in cfg:
        # ssm_norm.weight has shape (head_v_dim,).
        ssm_norm_shape = shapes.get(f"blk.{block_idx}.ssm_norm.weight")
        if ssm_norm_shape:
            cfg["linear_value_head_dim"] = int(ssm_norm_shape[0])

    if "linear_conv_kernel_dim" not in cfg or "linear_key_head_dim" not in cfg:
        # ssm_conv1d.weight: GGUFReader reverses GGUF ne, so shape = [kernel_size, conv_dim].
        # (Verified: shape[0]=kernel_size, shape[1]=conv_dim for Qwen3.5 GGUF files.)
        ssm_conv1d_shape = shapes.get(f"blk.{block_idx}.ssm_conv1d.weight")
        if ssm_conv1d_shape and len(ssm_conv1d_shape) >= 2:
            kernel_size = int(ssm_conv1d_shape[0])  # shape[0] = kernel_size (typically 4)
            conv_dim = int(ssm_conv1d_shape[1])  # shape[1] = conv_dim = key_dim*2 + value_dim
            if "linear_conv_kernel_dim" not in cfg:
                cfg["linear_conv_kernel_dim"] = kernel_size
            if "linear_num_key_heads" not in cfg or "linear_key_head_dim" not in cfg:
                num_v_heads = cfg.get("linear_num_value_heads", 32)
                head_v_dim = cfg.get("linear_value_head_dim", 128)
                value_dim = num_v_heads * head_v_dim
                key_dim = (conv_dim - value_dim) // 2
                # Assume key_head_dim == value_head_dim (standard in Qwen3.5).
                head_k_dim = head_v_dim
                if "linear_key_head_dim" not in cfg:
                    cfg["linear_key_head_dim"] = head_k_dim
                if "linear_num_key_heads" not in cfg and head_k_dim > 0:
                    cfg["linear_num_key_heads"] = key_dim // head_k_dim

    # ---------- layer_types list ----------------------------------------
    n_layers = meta.block_count
    cfg["layer_types"] = ["linear_attention" if i in linear_blocks else "full_attention" for i in range(n_layers)]

    return cfg


# ---------------------------------------------------------------------------
# GGUF metadata reader
# ---------------------------------------------------------------------------


class GGUFMetadata:
    """Thin wrapper around :class:`gguf.GGUFReader` exposing typed metadata."""

    def __init__(self, path: str):
        try:
            from gguf import GGUFReader
        except ImportError:
            raise ImportError("Install 'gguf': pip install gguf")
        self._reader = GGUFReader(path)
        self._fields = {f.name: f for f in self._reader.fields.values()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_field(self, key: str):
        return self._fields.get(key)

    def _scalar(self, key: str, default=None):
        """Return the Python scalar value of a GGUF metadata field."""
        field = self._get_field(key)
        if field is None or not field.data:
            return default
        # field.data is a list of part indices; the last one holds the value
        raw = field.parts[field.data[-1]]
        val = raw.tolist()
        if isinstance(val, list):
            return val[0] if len(val) == 1 else val
        return val

    def _str(self, key: str, default: str = "") -> str:
        """Return a string metadata field decoded from UTF-8 bytes."""
        field = self._get_field(key)
        if field is None or not field.data:
            return default
        raw = field.parts[field.data[-1]]
        return bytes(raw).decode("utf-8", errors="replace")

    def _str_list(self, key: str) -> list[str]:
        """Return a list-of-strings metadata field."""
        field = self._get_field(key)
        if field is None:
            return []
        return [bytes(field.parts[idx]).decode("utf-8", errors="replace") for idx in field.data]

    def _a(self, template: str) -> str:
        """Substitute ``{arch}`` in a metadata key template."""
        return template.format(arch=self.arch)

    # ------------------------------------------------------------------
    # Architecture
    # ------------------------------------------------------------------

    @property
    def arch(self) -> str:
        return self._str("general.architecture", "llama")

    @property
    def model_name(self) -> str:
        return self._str("general.name", "")

    # ------------------------------------------------------------------
    # Model hyperparameters
    # ------------------------------------------------------------------

    @property
    def block_count(self) -> int:
        return int(self._scalar(self._a("{arch}.block_count"), 0))

    @property
    def context_length(self) -> int:
        return int(self._scalar(self._a("{arch}.context_length"), 2048))

    @property
    def embedding_length(self) -> int:
        return int(self._scalar(self._a("{arch}.embedding_length"), 512))

    @property
    def feed_forward_length(self) -> int:
        return int(self._scalar(self._a("{arch}.feed_forward_length"), 0))

    @property
    def expert_feed_forward_length(self) -> int:
        return int(self._scalar(self._a("{arch}.expert_feed_forward_length"), 0))

    @property
    def head_count(self) -> int:
        return int(self._scalar(self._a("{arch}.attention.head_count"), 8))

    @property
    def head_count_kv(self) -> Optional[int]:
        v = self._scalar(self._a("{arch}.attention.head_count_kv"), None)
        return int(v) if v is not None else None

    @property
    def head_dim(self) -> Optional[int]:
        v = self._scalar(self._a("{arch}.attention.key_length"), None)
        return int(v) if v is not None else None

    @property
    def norm_rms_eps(self) -> float:
        return float(self._scalar(self._a("{arch}.attention.layer_norm_rms_epsilon"), 1e-5))

    @property
    def norm_eps(self) -> float:
        return float(self._scalar(self._a("{arch}.attention.layer_norm_epsilon"), 1e-5))

    @property
    def vocab_size(self) -> int:
        v = self._scalar(self._a("{arch}.vocab_size"), None)
        if v is not None:
            return int(v)
        return max(len(self.tokens), 32000)

    @property
    def rope_freq_base(self) -> float:
        return float(self._scalar(self._a("{arch}.rope.freq_base"), 10000.0))

    @property
    def rope_dim_count(self) -> Optional[int]:
        v = self._scalar(self._a("{arch}.rope.dimension_count"), None)
        return int(v) if v is not None else None

    @property
    def rope_dim_sections(self) -> Optional[list]:
        """MRoPE section sizes (`{arch}.rope.dimension_sections`).

        Present in Qwen3.5-VL and similar multi-modal models that apply
        different RoPE frequencies to each section of the head dimension.
        Maps to ``xdrope_section`` in the EOLE ``rope_config``.

        GGUF arrays are sometimes padded with trailing zeros (e.g. the
        Qwen3.5-9B GGUF stores ``[11, 11, 10, 0]`` instead of
        ``[11, 11, 10]``).  Trailing zeros are stripped so that
        ``xdrope_section`` matches the three-element T/H/W layout expected
        by :func:`~eole.modules.rope.apply_rotary_pos_emb_xdrope`.

        **Implementation note**: for GGUF ARRAY fields, gguf-python stores one
        index per element in ``field.data``, each pointing to a separate 1-element
        numpy array in ``field.parts``.  Reading only ``field.parts[field.data[-1]]``
        would give just the *last* element (the trailing zero), so we must iterate
        over *all* indices in ``field.data``.
        """
        field = self._get_field(self._a("{arch}.rope.dimension_sections"))
        if field is None:
            return None
        try:
            # Collect all element values: each field.data[i] is an index into
            # field.parts pointing to a 1-element numpy array for that element.
            sections = [pv for idx in field.data for pv in field.parts[idx].tolist()]
            # Strip trailing zeros (GGUF null-padding artefact).
            while sections and sections[-1] == 0:
                sections.pop()
            return sections if sections else None
        except Exception:
            return None

    @property
    def rope_scaling_type(self) -> Optional[str]:
        return self._str(self._a("{arch}.rope.scaling.type"), "") or None

    @property
    def rope_scaling_factor(self) -> Optional[float]:
        v = self._scalar(self._a("{arch}.rope.scaling.factor"), None)
        return float(v) if v is not None else None

    @property
    def rope_scaling_original_ctx_len(self) -> Optional[int]:
        v = self._scalar(self._a("{arch}.rope.scaling.original_context_length"), None)
        return int(v) if v is not None else None

    @property
    def expert_count(self) -> int:
        return int(self._scalar(self._a("{arch}.expert_count"), 0))

    @property
    def expert_used_count(self) -> int:
        return int(self._scalar(self._a("{arch}.expert_used_count"), 0))

    @property
    def use_parallel_residual(self) -> bool:
        v = self._scalar(self._a("{arch}.use_parallel_residual"), None)
        return bool(v) if v is not None else False

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------

    @property
    def tokenizer_model(self) -> str:
        return self._str("tokenizer.ggml.model", "llama")

    @property
    def tokens(self) -> list[str]:
        return self._str_list("tokenizer.ggml.tokens")

    @property
    def token_types(self) -> list:
        field = self._get_field("tokenizer.ggml.token_type")
        if field is None:
            return []
        return field.parts[field.data[-1]].tolist() if field.data else []

    @property
    def merges(self) -> list[str]:
        return self._str_list("tokenizer.ggml.merges")

    @property
    def bos_token_id(self) -> Optional[int]:
        v = self._scalar("tokenizer.ggml.bos_token_id", None)
        return int(v) if v is not None else None

    @property
    def eos_token_id(self) -> Optional[int]:
        v = self._scalar("tokenizer.ggml.eos_token_id", None)
        return int(v) if v is not None else None

    @property
    def unk_token_id(self) -> Optional[int]:
        v = self._scalar("tokenizer.ggml.unknown_token_id", None)
        return int(v) if v is not None else None

    @property
    def pad_token_id(self) -> Optional[int]:
        v = self._scalar("tokenizer.ggml.padding_token_id", None)
        return int(v) if v is not None else None

    @property
    def add_bos_token(self) -> bool:
        v = self._scalar("tokenizer.ggml.add_bos_token", None)
        return bool(v) if v is not None else True

    @property
    def chat_template(self) -> Optional[str]:
        raw = self._str("tokenizer.chat_template", "")
        return raw if raw else None

    @property
    def hf_tokenizer_json(self) -> Optional[str]:
        """Embedded HuggingFace tokenizer.json (if present in the GGUF)."""
        raw = self._str("tokenizer.huggingface.json", "")
        return raw if raw else None

    # ------------------------------------------------------------------
    # Tensors
    # ------------------------------------------------------------------

    @property
    def tensors(self):
        return self._reader.tensors


# ---------------------------------------------------------------------------
# Config building
# ---------------------------------------------------------------------------

_RMS_NORM_ARCHS = frozenset(
    "llama mistral mixtral qwen2 qwen2moe qwen3 qwen3moe qwen35 phi3 deepseek2 "
    "falcon gemma gemma2 gemma3 llama4 deci grok starcoder2".split()
)
_SWIGLU_ARCHS = frozenset(
    "llama mistral mixtral qwen2 qwen2moe qwen3 qwen3moe qwen35 phi3 deepseek2 "
    "gemma gemma2 gemma3 llama4 deci grok starcoder2".split()
)
# Architectures that apply per-head RMS normalisations to Q and K before
# (or after) RoPE.  These require query_norm=True / key_norm=True in the
# decoder config so that MultiHeadedAttention creates the norm modules and
# the stored q_norm / k_norm weights can be loaded.
_QK_NORM_ARCHS = frozenset("qwen3 qwen3moe qwen35 gemma3 deepseek2".split())
# Architectures whose attention uses gated queries (Qwen3.5 style).
# The linear_query projection has doubled output size: first half is the
# actual query, second half is the sigmoid gate.  Without q_gating=True the
# GGUFLinear forward pass crashes because it tries to reshape (out=4096)
# a dequantized buffer that has 8192*in_features elements.
_Q_GATING_ARCHS = frozenset("qwen35".split())
# Architectures that use Gemma-style RMS norm (scaled by 1 + weight, not just
# weight).  Must match HF_mappings.LN_TABLE.
_GEMMA_RMS_ARCHS = frozenset("gemma2 gemma3 qwen35 qwen35moe".split())
# Architectures that use MRoPE (multi-dimensional rotary positional encoding)
# with interleaved rotation and therefore require rotary_interleave=True in
# the EOLE rope_config.
#
# qwen35 / qwen35moe are vision-language models whose text decoder always uses
# the interleaved MRoPE defined in Qwen3_5TextRotaryEmbedding.apply_interleaved_mrope
# (see transformers/models/qwen3_5/modeling_qwen3_5.py).  Even in text-only
# inference, position_ids are expanded to 3 dimensions and the interleaved
# layout is applied — so rotary_interleave=True must be set for all qwen35 GGUF
# files, regardless of whether rope.dimension_sections metadata is present.
_MROPE_INTERLEAVE_ARCHS = frozenset("qwen35 qwen35moe qwen2vl qwen3vl qwen3vlmoe".split())


def build_model_config(meta: GGUFMetadata, linear_blocks: frozenset = frozenset()) -> dict:
    """Build an EOLE ``model_config`` dict from GGUF metadata.

    Args:
        meta: Parsed GGUF metadata.
        linear_blocks: Set of block indices detected as linear-attention (SSM/GatedDeltaNet)
            blocks.  When non-empty, ``layer_types`` and GatedDeltaNet hyper-parameters
            are added to the returned config.
    """
    arch = meta.arch.lower()
    hidden_size = meta.embedding_length
    layers = meta.block_count
    heads = meta.head_count
    heads_kv = meta.head_count_kv
    head_dim = meta.head_dim
    ff_length = meta.feed_forward_length
    rope_base = meta.rope_freq_base
    rope_dim = meta.rope_dim_count
    rope_dim_sections = meta.rope_dim_sections
    rope_scaling_type = meta.rope_scaling_type
    rope_scaling_factor = meta.rope_scaling_factor
    rope_scaling_orig_ctx = meta.rope_scaling_original_ctx_len
    num_experts = meta.expert_count
    num_experts_per_tok = meta.expert_used_count
    expert_ff = meta.expert_feed_forward_length

    use_rms = arch in _RMS_NORM_ARCHS
    norm_eps = meta.norm_rms_eps if use_rms else meta.norm_eps
    # Gemma2/3 and Qwen3.5 use a scaled variant (weight + 1) of RMS norm.
    if arch in _GEMMA_RMS_ARCHS:
        layer_norm = "gemma-rms"
    elif use_rms:
        layer_norm = "rms"
    else:
        layer_norm = "standard"
    mlp_activation_fn = "gated-silu" if arch in _SWIGLU_ARCHS else "gelu"

    model_config: dict = {
        "layers": layers,
        "hidden_size": hidden_size,
        "heads": heads,
        "transformer_ff": ff_length,
        "mlp_activation_fn": mlp_activation_fn,
        "layer_norm": layer_norm,
        "norm_eps": norm_eps,
        "add_qkvbias": False,
        "add_final_linear_bias": False,
        "add_ffnbias": False,
        "shared_layer_norm": False,
        "generator_bias": False,
        "rope_config": {
            "rotary_interleave": False,
            "rotary_theta": rope_base,
        },
        "max_position_embeddings": meta.context_length,
        "embeddings": {
            "position_encoding_type": PositionEncodingType.Rotary,
            "n_positions": 0,
            # Embedding vectors must match the model's hidden dimension.
            # EmbeddingsConfig.tgt_word_vec_size defaults to 512, so we must
            # set it explicitly here or the decoder will process 512-dim
            # tensors regardless of the actual hidden_size.
            "src_word_vec_size": hidden_size,
            "tgt_word_vec_size": hidden_size,
        },
    }

    if heads_kv is not None:
        model_config["heads_kv"] = heads_kv

    if head_dim is not None:
        model_config["head_dim"] = head_dim

    # Rotary dimension
    effective_head_dim = head_dim if head_dim is not None else hidden_size // heads
    if rope_dim is not None:
        model_config["rope_config"]["rotary_dim"] = rope_dim
    else:
        model_config["rope_config"]["rotary_dim"] = effective_head_dim

    # MRoPE section sizes (xdrope_section in EOLE): present in multimodal models
    # like Qwen3.5-VL that partition the head dimension into independent RoPE sections.
    if rope_dim_sections is not None:
        model_config["rope_config"]["xdrope_section"] = rope_dim_sections

    # Architectures that use interleaved MRoPE (mrope_interleaved=True in HF config).
    if arch in _MROPE_INTERLEAVE_ARCHS:
        model_config["rope_config"]["rotary_interleave"] = True

    # Rope scaling (e.g. yarn, linear) from GGUF rope.scaling.* metadata.
    if rope_scaling_type is not None:
        model_config["rope_config"]["scaling_type"] = rope_scaling_type
        if rope_scaling_factor is not None:
            model_config["rope_config"]["scaling_factor"] = rope_scaling_factor
        if rope_scaling_orig_ctx is not None:
            model_config["rope_config"]["original_max_position_embeddings"] = rope_scaling_orig_ctx

    if num_experts > 0:
        model_config["num_experts"] = num_experts
        model_config["num_experts_per_tok"] = num_experts_per_tok or 2
        if expert_ff:
            model_config["moe_transformer_ff"] = expert_ff

    # Architecture-specific overrides
    if arch in ("phi2",):
        model_config.update(
            parallel_residual=True,
            shared_layer_norm=True,
            add_qkvbias=True,
            add_final_linear_bias=True,
            add_ffnbias=True,
        )
    # Qwen2 has Q/K/V projection biases in the decoder; Qwen3 and Qwen3.5 VL
    # do not.  The vision encoder's add_qkvbias is set separately in
    # build_vision_encoder_config and is always True (the encoder has biases).
    if arch in ("qwen2",):
        model_config.update(add_qkvbias=True, add_final_linear_bias=False)
    if meta.use_parallel_residual:
        model_config["parallel_residual"] = True

    if arch in ("gemma2", "gemma3"):
        model_config["ffn_layernorm"] = True

    # Architectures that apply per-head RMS norms to Q and K projections.
    # These must go in the decoder sub-dict (TransformerConfig fields are
    # shared between encoder and decoder; putting them at model level would
    # incorrectly propagate to the vision encoder in VLM models).
    if arch in _QK_NORM_ARCHS:
        model_config.setdefault("decoder", {}).update(
            query_norm=True,
            key_norm=True,
        )

    # Architectures with gated query attention (Qwen3.5 style).
    # linear_query has 2× output size; first half is query, second half is
    # sigmoid gate.  Must go in the decoder sub-dict (same reason as QK norms).
    if arch in _Q_GATING_ARCHS:
        model_config.setdefault("decoder", {}).update(q_gating=True)

    # Hybrid models: add layer_types and GatedDeltaNet hyper-parameters.
    # These are decoder-specific fields (TransformerDecoderConfig), so they must
    # go under the "decoder" sub-dict rather than the top-level model config.
    if linear_blocks:
        lin_cfg = _infer_linear_attn_config(meta, linear_blocks)
        model_config.setdefault("decoder", {}).update(lin_cfg)

    return model_config


# ---------------------------------------------------------------------------
# Tensor-name → EOLE leaf module name (for quant_layers detection)
# ---------------------------------------------------------------------------

# These are the leaf module names (last path component) that correspond to
# linear layers inside a transformer block.
_QUANTIZABLE_SUFFIXES = frozenset(
    "attn_q attn_k attn_v attn_output attn_qkv "
    "ffn_gate ffn_up ffn_down ffn_gate_up "
    "ffn_gate_exps ffn_up_exps ffn_down_exps".split()
)

# Map suffix → EOLE leaf module name  (only the first component of the mapped path)
_SUFFIX_TO_EOLE_LEAF: dict[str, str] = {
    "attn_q": "linear_query",
    "attn_k": "linear_keys",
    "attn_v": "linear_values",
    "attn_output": "final_linear",
    "attn_qkv": "qkv_proj",
    "ffn_gate": "gate_up_proj",
    "ffn_up": "up_proj",
    "ffn_down": "down_proj",
    "ffn_gate_up": "gate_up_proj",
    "ffn_gate_exps": "experts_gate_up",
    "ffn_up_exps": "experts_up",
    "ffn_down_exps": "experts_down",
    "ffn_gate_inp": "gate",
}


def _detect_quant_layers(tensors) -> tuple[list[str], str]:
    """Scan tensors and return (quant_layers, dominant_qtype_name).

    ``quant_layers`` contains the EOLE leaf-module names of all layers that
    carry at least one quantised tensor.
    ``dominant_qtype_name`` is the name of the most-common quantisation type
    across those tensors (e.g. ``"Q4_K"``).
    """
    from collections import Counter

    quant_leaf_set: set[str] = set()
    qtype_counter: Counter = Counter()

    for tensor in tensors:
        m = re.fullmatch(r"blk\.\d+\.(.*?)(?:\.weight|\.bias)?$", tensor.name)
        if not m:
            continue
        raw_suffix = m.group(1)
        # Strip any ".weight" / ".bias" that might still be in raw_suffix
        base_suffix = raw_suffix.rsplit(".", 1)[0] if "." in raw_suffix else raw_suffix
        if base_suffix not in _QUANTIZABLE_SUFFIXES:
            continue
        if _is_float_type(tensor.tensor_type.name):
            continue
        leaf = _SUFFIX_TO_EOLE_LEAF.get(base_suffix)
        if leaf:
            quant_leaf_set.add(leaf)
        qtype_counter[tensor.tensor_type.name] += 1

    dominant_qtype = qtype_counter.most_common(1)[0][0] if qtype_counter else ""
    return sorted(quant_leaf_set), dominant_qtype


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

_TOKEN_TYPE_CONTROL = 3  # ggml token type: control (BOS/EOS/…)


def _extract_vocab_from_gguf(meta: GGUFMetadata, vocabs: dict, output_dir: str) -> tuple:
    """Extract vocab from GGUF and write tokenizer artefacts.

    Returns ``(src_vocab, tokenizer_basename, gpt2_pretok)``.
    """
    if pyonmttok is None:
        raise ImportError("Install pyonmttok: pip install pyonmttok")

    tokens = meta.tokens
    merges = meta.merges
    if not tokens:
        raise ValueError("GGUF file contains no tokenizer.ggml.tokens")

    def _tok(tid):
        return tokens[tid] if tid is not None and 0 <= tid < len(tokens) else None

    bos_str = _tok(meta.bos_token_id)
    eos_str = _tok(meta.eos_token_id)
    unk_str = _tok(meta.unk_token_id)
    pad_str = _tok(meta.pad_token_id)

    if bos_str:
        vocabs["specials"]["bos_token"] = bos_str
    if eos_str:
        vocabs["specials"]["eos_token"] = eos_str
    if unk_str:
        vocabs["specials"]["unk_token"] = unk_str
    if pad_str:
        vocabs["specials"]["pad_token"] = pad_str
    else:
        vocabs["specials"].setdefault("pad_token", DefaultTokens.PAD)

    gpt2_pretok = bool(merges)
    if merges:
        tokenizer_basename = "bpe.model"
        with open(os.path.join(output_dir, tokenizer_basename), "w", encoding="utf-8") as f:
            f.write("v3;false;false;false;Ġ;Ġ\n")
            for merge in merges:
                f.write(merge + "\n")
    else:
        tokenizer_basename = "sentencepiece.model"
        sp_path = os.path.join(output_dir, tokenizer_basename)
        if not os.path.exists(sp_path):
            open(sp_path, "wb").close()  # placeholder

    vocab_list = list(tokens)
    declared = meta.vocab_size
    while len(vocab_list) < declared:
        vocab_list.append(f"{DefaultTokens.VOCAB_PAD}{len(vocab_list)}")

    if gpt2_pretok:
        vocab_list = [DefaultTokens.PAD if t == "Ā" else t for t in vocab_list]
        if DefaultTokens.PAD in vocab_list:
            vocabs["specials"]["pad_token"] = DefaultTokens.PAD

    src_vocab = pyonmttok.build_vocab_from_tokens(vocab_list)
    return src_vocab, tokenizer_basename, gpt2_pretok


# ---------------------------------------------------------------------------
# Tensor conversion
# ---------------------------------------------------------------------------


def _tensor_to_torch(tensor, target_dtype: torch.dtype) -> torch.Tensor:
    """Convert a single GGUF ReaderTensor to a torch.Tensor.

    For quantised types (Q4_K, Q8_0, …) the raw ``uint8`` block data is
    stored as-is.  The companion ``gguf_qtype`` tensor carries the
    :class:`gguf.GGMLQuantizationType` integer value so that
    :class:`~eole.modules.gguf_linear.GGUFLinear` can dequantize correctly
    at inference time.

    For float types (F16, BF16, F32) the tensor is cast to *target_dtype*.

    **BF16 special case**: gguf-python returns BF16 tensors as raw ``uint8``
    bytes (block_size=1, type_size=2) rather than a native float dtype,
    because NumPy has no bfloat16 type.  The resulting tensor has shape
    ``(out_features, in_features * 2)`` with dtype ``uint8``.  We reinterpret
    every consecutive pair of bytes as a ``bfloat16`` value (identical to how
    vLLM's ``gguf_quant_weights_iterator`` handles this case) and then cast to
    *target_dtype*.  Without this fix the raw bytes would be stored as-is and
    loaded back as garbage float values.

    Returns ``(tensor, qtype_tensor_or_None)``.
    ``qtype_tensor_or_None`` is a ``torch.int32`` scalar when the tensor is
    quantised; ``None`` for float tensors.
    """
    import numpy as np

    tname = tensor.tensor_type.name
    data = tensor.data  # numpy ndarray

    if _is_float_type(tname):
        t = torch.from_numpy(data.copy())
        if not t.is_floating_point() and tname == "BF16":
            # gguf-python returns BF16 data as raw uint8 bytes.
            # Reinterpret each consecutive pair of bytes as bfloat16, then
            # cast to the requested target dtype.  Matches vLLM's handling in
            # gguf_quant_weights_iterator.
            t = t.contiguous().view(torch.bfloat16)
        if t.is_floating_point():
            t = t.to(target_dtype)
        return t, None
    else:
        # Quantized: keep raw uint8 bytes; data.shape is already
        # (out_features, bytes_per_row) for 2-D weights.
        raw = data.copy()
        if raw.dtype != np.uint8:
            raw = raw.view(np.uint8)
        t = torch.from_numpy(raw)  # dtype=torch.uint8
        qtype_t = torch.tensor([tensor.tensor_type.value], dtype=torch.int32)
        return t, qtype_t


# ---------------------------------------------------------------------------
# Shard builder
# ---------------------------------------------------------------------------


def build_safetensors(
    meta: GGUFMetadata,
    output_dir: str,
    target_dtype: torch.dtype,
    linear_blocks: frozenset = frozenset(),
    extra_tensors: Optional[dict] = None,
    has_vision_encoder: bool = False,
) -> tuple[dict, list[str]]:
    """Convert all tensors from the GGUF and write ``model.00.safetensors``.

    Args:
        meta: Parsed GGUF metadata / tensor source.
        output_dir: Directory where ``model.00.safetensors`` will be written.
        target_dtype: Target dtype for non-quantised (float) tensors.
        linear_blocks: Set of block indices that are linear-attention (SSM)
            blocks, used to route tensor names correctly.
        extra_tensors: Optional dict of additional tensors (already converted
            to torch) to merge into the safetensors shard.  Used when an
            mmproj GGUF has been separately converted.
        has_vision_encoder: When True (VLM model with a separate vision
            encoder), ``replace_gguf_linear`` at load time targets only
            ``self.decoder`` and does not reach ``self.generator``.  In that
            case ``generator.weight`` must be dequantised to float so that it
            can be loaded by the plain ``nn.Linear`` generator.
            When False (plain text model), ``replace_gguf_linear`` is applied
            to the whole model (``quant_target = self``) and *will* replace
            ``self.generator`` with a :class:`~eole.modules.gguf_linear.GGUFLinear`,
            so ``generator.weight`` can safely remain as quantised ``uint8``.

    Returns ``(written_tensor_dict, quant_layers)`` where *quant_layers* is the
    list of EOLE leaf-module names (e.g. ``"linear_query"``) that carry
    quantised weights and therefore need to be replaced with
    :class:`~eole.modules.gguf_linear.GGUFLinear`.
    """
    if safetensors_save_file is None:
        raise ImportError("Install safetensors: pip install safetensors")

    # Choose the correct must-dequantize set based on the model topology.
    # For VLM models, replace_gguf_linear targets self.decoder at load time
    # (not self.generator), so generator.weight must be a plain float tensor.
    # For plain text models, replace_gguf_linear targets self (the whole
    # model), which does include self.generator, so it can stay quantized.
    must_deq = _MUST_DEQUANTIZE_NAMES_WITH_GENERATOR if has_vision_encoder else _MUST_DEQUANTIZE_NAMES_BASE

    # Compute V-head reordering parameters for linear-attention blocks.
    # llama.cpp's _LinearAttentionVReorderBase reorders V heads from grouped
    # (HF) to tiled order when writing GGUF files (only when num_k_heads !=
    # num_v_heads, e.g. Qwen3.5-14B has num_k_heads=32, num_v_heads=64).
    # We must apply the inverse transformation here.
    _lin_cfg = _infer_linear_attn_config(meta, linear_blocks) if linear_blocks else {}
    _num_k_heads = _lin_cfg.get("linear_num_key_heads", 0)
    _num_v_heads = _lin_cfg.get("linear_num_value_heads", 0)
    _head_v_dim = _lin_cfg.get("linear_value_head_dim", 0)
    _head_k_dim = _lin_cfg.get("linear_key_head_dim", 0)
    _needs_v_reorder = bool(linear_blocks and _num_k_heads > 0 and _num_v_heads > 0 and _num_k_heads != _num_v_heads)
    _num_v_per_k = (_num_v_heads // _num_k_heads) if _num_k_heads > 0 else 1

    written: dict[str, torch.Tensor] = {}

    # Per-input-tensor accounting for the verbose summary.
    n_input = 0  # total GGUF tensors read (excluding explicitly-None mapped ones)
    n_filtered = 0  # eole_name is None  (rope_freqs, etc.)
    skipped: list[str] = []  # eole_name == "" (unmapped)
    n_quant_weight = 0  # stored as uint8 (GGUFLinear)
    n_dequant_weight = 0  # quantized in GGUF but dequantized to float (embeddings)
    n_float_tensor = 0  # already-float (norms, biases, …)

    quant_layers, dominant_qtype = _detect_quant_layers(meta.tensors)

    quant_bytes = 0
    dequant_overhead_bytes = 0
    float_bytes = 0

    for tensor in meta.tensors:
        gguf_name = tensor.name
        eole_name = _gguf_to_eole_name(gguf_name, linear_blocks)

        if eole_name is None:
            # Explicitly skipped (e.g. rope_freqs)
            n_filtered += 1
            continue
        n_input += 1
        if eole_name == "":
            skipped.append(gguf_name)
            continue

        t, qtype_t = _tensor_to_torch(tensor, target_dtype)

        # Some EOLE modules cannot hold uint8 quantized data (nn.Embedding,
        # generator/lm-head nn.Linear when not covered by replace_gguf_linear).
        # Dequantize these to target_dtype so they can be loaded normally.
        if qtype_t is not None and eole_name in must_deq:
            orig_bytes = tensor.data.nbytes
            try:
                import numpy as _np
                from gguf.quants import dequantize as _gguf_dequantize

                dq = _gguf_dequantize(tensor.data, tensor.tensor_type)
                t = torch.from_numpy(_np.array(dq, copy=True)).to(target_dtype)
                qtype_t = None  # no companion gguf_qtype – it's plain float now
                dequant_overhead_bytes += t.nbytes - orig_bytes
                n_dequant_weight += 1
            except Exception as exc:
                raise RuntimeError(
                    f"Tensor {gguf_name!r} (→ {eole_name!r}) is quantized "
                    f"({tensor.tensor_type.name}) and could not be dequantized: {exc}"
                ) from exc

        # -------------------------------------------------------------------
        # Inverse V-head reordering for linear-attention blocks.
        #
        # llama.cpp's _LinearAttentionVReorderBase swaps V-head order from
        # grouped (HF) to tiled when writing GGUF (only when num_k_heads !=
        # num_v_heads).  We must undo that here.
        #
        # This works on both float tensors and uint8 quantized tensors:
        # – Row reordering: simply reindex rows (or uint8 bytes-per-row).
        # – Column reordering (out_proj): reindex column groups; requires the
        #   columns to be divisible by num_v_heads (holds when head_v_dim%32==0,
        #   which is always the case for standard Qwen3.5 heads).
        # -------------------------------------------------------------------
        if _needs_v_reorder:
            m_blk = re.fullmatch(r"blk\.(\d+)\.(.*)", gguf_name)
            if m_blk and int(m_blk.group(1)) in linear_blocks:
                _blk_suffix = m_blk.group(2)
                _qk_rows = _num_k_heads * _head_k_dim * 2

                if _blk_suffix in ("ssm_a", "ssm_a.weight"):
                    # A_log: 1D float tensor (num_v_heads,)
                    if t.dim() == 1:
                        t = _inverse_v_head_reorder_1d(t, _num_k_heads, _num_v_per_k)

                elif _blk_suffix == "ssm_dt.bias":
                    # dt_bias: 1D float tensor (num_v_heads,)
                    if t.dim() == 1:
                        t = _inverse_v_head_reorder_1d(t, _num_k_heads, _num_v_per_k)

                elif _blk_suffix in ("ssm_alpha.weight", "ssm_beta.weight"):
                    # in_proj_a / in_proj_b: (num_v_heads, hidden), head_dim=1
                    if t.dim() == 2:
                        t = _inverse_v_head_reorder_rows(t, _num_k_heads, _num_v_per_k, 1)

                elif _blk_suffix == "attn_gate.weight":
                    # in_proj_z: (num_v_heads * head_v_dim, hidden)
                    if t.dim() == 2:
                        t = _inverse_v_head_reorder_rows(t, _num_k_heads, _num_v_per_k, _head_v_dim)

                elif _blk_suffix == "attn_qkv.weight":
                    # in_proj_qkv: rows = Q+K (grouped, no reorder) + V (tiled)
                    # Only the V-part rows need to be reordered.
                    if t.dim() == 2 and t.shape[0] > _qk_rows:
                        qk_part = t[:_qk_rows]
                        v_part = t[_qk_rows:]
                        v_part = _inverse_v_head_reorder_rows(v_part, _num_k_heads, _num_v_per_k, _head_v_dim)
                        t = torch.cat([qk_part, v_part], dim=0)

                elif _blk_suffix == "ssm_out.weight":
                    if qtype_t is not None and t.dim() == 2:
                        import numpy as _np2
                        from gguf.quants import dequantize as _gguf_dq2, quantize as _gguf_q2
                        from gguf import GGMLQuantizationType

                        # Dequantize original Q5_K → float32
                        dq = _gguf_dq2(tensor.data, tensor.tensor_type)
                        t_f32 = torch.from_numpy(_np2.array(dq, copy=True)).float()
                        t_f32 = t_f32.reshape(t.shape[0], -1)  # (out, in)

                        # Undo column reorder on float
                        t_f32 = _inverse_v_head_reorder_cols(t_f32, _num_k_heads, _num_v_per_k)

                        # Re-quantize to Q8_0 (quantize_blocks is implemented, higher quality)
                        reqs = _gguf_q2(t_f32.numpy(), GGMLQuantizationType.Q8_0)
                        t = torch.from_numpy(_np2.array(reqs, copy=True))

                        # Update qtype_t to Q8_0 so GGUFLinear gets the right gguf_qtype scalar
                        qtype_t = torch.tensor([GGMLQuantizationType.Q8_0.value], dtype=torch.int32)
                    elif t.dim() == 2:
                        t = _inverse_v_head_reorder_cols(t, _num_k_heads, _num_v_per_k)

                elif _blk_suffix == "ssm_conv1d.weight":
                    # conv1d: (conv_dim, kernel_size) – only V channels need reorder.
                    # Note: the unsqueeze(1) happens AFTER this block.
                    if t.dim() == 2 and t.shape[0] > _qk_rows:
                        qk_ch = t[:_qk_rows]
                        v_ch = t[_qk_rows:]
                        v_ch = _inverse_v_head_reorder_rows(v_ch, _num_k_heads, _num_v_per_k, _head_v_dim)
                        t = torch.cat([qk_ch, v_ch], dim=0)

        # A_log inverse transform: llama.cpp's Qwen3NextModel.modify_tensors converts
        # the HF A_log (log-scale) to -exp(A_log) before writing ssm_a to GGUF.
        # EOLE's GatedDeltaNet.forward() computes A = -exp(self.A_log), so it expects
        # A_log in log-scale.  Restore: A_log = log(-ssm_a).
        if eole_name.endswith(".A_log") and t.dim() == 1:
            t = torch.log(-t)

        # GemmaRMSNorm layernorm1p correction: for architectures in _GEMMA_RMS_ARCHS
        # (gemma2, gemma3, qwen35, qwen35moe), llama.cpp adds +1 to all norm weights
        # during HF→GGUF conversion so ggml's standard RMSNorm produces the same
        # result as HF's layernorm1p (which stores weight as deviation from 1 and
        # computes (1+weight)*x/rms(x)).  EOLE's GemmaRMSNorm also uses (1+weight),
        # so the +1 would be applied twice.  Subtract 1 here to restore the
        # "deviation-from-1" convention expected by GemmaRMSNorm.
        # Exception: linear_attn.norm.weight is a plain RMSNormGated (not GemmaRMSNorm)
        # and was explicitly excluded from the +1 by llama.cpp's Qwen3NextModel.
        if (
            meta.arch in _GEMMA_RMS_ARCHS
            and qtype_t is None  # float tensors only; norm weights are never quantized
            and eole_name.endswith("norm.weight")
            and not eole_name.endswith("linear_attn.norm.weight")
        ):
            t = t - 1.0

        # Conv1d weights: the gguf reader reverses GGUF ne, so data arrives as
        # (conv_dim, kernel_size) – i.e. shape[0]=kernel_size, shape[1]=conv_dim
        # in tensor.shape, but data.shape=(conv_dim, kernel_size) in NumPy order.
        # PyTorch nn.Conv1d expects (out_channels, in_channels/groups, kernel_size)
        # = (conv_dim, 1, kernel_size).  Insert the groups dim without transposing.
        if eole_name.endswith("conv1d.weight") and t.dim() == 2:
            t = t.unsqueeze(1).contiguous()

        written[eole_name] = t

        # For quantised linear weights, also store the per-tensor qtype so
        # GGUFLinear.forward() knows how to dequantize.
        if qtype_t is not None:
            # Derive the companion gguf_qtype key: replace ".weight" → ".gguf_qtype"
            qtype_key = eole_name.rsplit(".weight", 1)[0] + ".gguf_qtype"
            written[qtype_key] = qtype_t
            quant_bytes += t.nbytes
            n_quant_weight += 1
        else:
            float_bytes += t.nbytes
            # n_dequant_weight was already incremented above for tensors that were
            # quantized in the GGUF but forced to float.  Only count here tensors
            # that were native float types (F32/F16/BF16) in the GGUF.
            if _is_float_type(tensor.tensor_type.name):
                n_float_tensor += 1
            # (else: quantized → dequantized → already counted in n_dequant_weight)

    n_mapped = n_input - len(skipped)  # tensors that produced an eole key

    # Merge in any extra tensors from the vision encoder (mmproj)
    n_vision = len(extra_tensors) if extra_tensors else 0
    if extra_tensors:
        written.update(extra_tensors)

    shard_path = os.path.join(output_dir, "model.00.safetensors")
    safetensors_save_file(written, shard_path)

    shard_size_mb = os.path.getsize(shard_path) / 1024 / 1024
    # Identity check: n_mapped == n_quant_weight + n_dequant_weight + n_float_tensor
    # Safetensors entries from decoder == n_mapped + n_quant_weight (the gguf_qtype scalars)
    n_written_total = len(written)
    expected_decoder_entries = n_mapped + n_quant_weight

    print(f"Decoder GGUF  ({len(meta.tensors)} input tensors):")
    print(f"  {n_filtered} filtered (rope_freqs, etc.) → not written")
    print(f"  {len(skipped)} unmapped (no EOLE key) → not written")
    if skipped:
        print(f"    {skipped[:5]}{'...' if len(skipped) > 5 else ''}")
    print(f"  {n_quant_weight} quantized weights → stored as uint8 ({quant_bytes / 1024**2:.0f} MB)")
    if n_dequant_weight:
        print(
            f"  {n_dequant_weight} quantized embeddings → dequantized to float16 "
            f"(+{dequant_overhead_bytes / 1024**2:.0f} MB overhead vs packed)"
        )
    print(f"  {n_float_tensor} native float tensors (norms, biases, …) ({float_bytes / 1024**2:.0f} MB)")
    print(
        f"  = {n_mapped} tensor EOLE keys + {n_quant_weight} gguf_qtype scalars "
        f"= {expected_decoder_entries} decoder safetensors entries"
    )
    if n_vision:
        print(f"  + {n_vision} vision encoder entries (from mmproj)")
    print(f"Shard: {shard_path}")
    print(f"  {n_written_total} total entries  ({shard_size_mb:.0f} MB)")

    return written, quant_layers


# ---------------------------------------------------------------------------
# Vision encoder (mmproj) GGUF support
# ---------------------------------------------------------------------------


class GGUFClipMetadata(GGUFMetadata):
    """GGUFMetadata specialised for ``clip`` architecture mmproj files.

    The clip GGUF uses ``clip.vision.*`` keys rather than the standard
    ``{arch}.*`` keys that :class:`GGUFMetadata` reads.
    """

    @property
    def arch(self) -> str:
        return self._str("general.architecture", "clip")

    # Vision-encoder specific fields
    @property
    def vision_block_count(self) -> int:
        return int(self._scalar("clip.vision.block_count", 27))

    @property
    def vision_embedding_length(self) -> int:
        return int(self._scalar("clip.vision.embedding_length", 1152))

    @property
    def vision_feed_forward_length(self) -> int:
        return int(self._scalar("clip.vision.feed_forward_length", 4304))

    @property
    def vision_head_count(self) -> int:
        return int(self._scalar("clip.vision.attention.head_count", 16))

    @property
    def vision_image_size(self) -> int:
        return int(self._scalar("clip.vision.image_size", 768))

    @property
    def vision_patch_size(self) -> int:
        return int(self._scalar("clip.vision.patch_size", 16))

    @property
    def vision_spatial_merge_size(self) -> int:
        return int(self._scalar("clip.vision.spatial_merge_size", 2))

    @property
    def vision_projection_dim(self) -> int:
        return int(self._scalar("clip.vision.projection_dim", 4096))

    @property
    def vision_norm_eps(self) -> float:
        return float(self._scalar("clip.vision.attention.layer_norm_epsilon", 1e-6))

    @property
    def projector_type(self) -> str:
        return self._str("clip.projector_type", "qwen3vl_merger")


def _mmproj_to_eole_tensors(
    clip_meta: "GGUFClipMetadata",
    target_dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Convert all tensors from a clip mmproj GGUF to EOLE names.

    Returns ``(eole_tensor_dict, skipped_names)`` where the dict maps EOLE
    tensor names to already-converted torch tensors.

    The Qwen3.5/Qwen3VL mmproj GGUF uses the following tensor naming scheme:

    * ``v.blk.{i}.attn_qkv.weight``  – fused QKV; split into separate Q/K/V
    * ``v.blk.{i}.attn_out.weight``  – attention output projection
    * ``v.blk.{i}.ffn_up.weight``    – MLP up projection → ``mlp.gate_up_proj``
    * ``v.blk.{i}.ffn_down.weight``  – MLP down projection → ``mlp.down_proj``
    * ``v.blk.{i}.ln1.*``            – pre-attention layer norm
    * ``v.blk.{i}.ln2.*``            – post-attention layer norm
    * ``v.patch_embd.weight``        – patch convolution weights (Conv2d or
      first temporal slice of Conv3d)
    * ``v.patch_embd.weight.1``      – second temporal slice of Conv3d
      (Qwen3.5 VL); stacked with ``weight`` to form a 5-D Conv3d weight
    * ``v.position_embd.weight`` or ``v.position_embed.weight`` – absolute
      position embedding table (name varies across llama.cpp versions)
    * ``v.post_ln.weight/bias``      – post-encoder layer norm (part of merger)
    * ``mm.0.weight/bias``           – merger linear_fc1
    * ``mm.2.weight/bias``           – merger linear_fc2
    """

    written: dict[str, torch.Tensor] = {}
    skipped: list[str] = []

    # Temporal patch conv: Qwen3.5 VL stores the Conv3d weight split across
    # two tensors (one per temporal step).  Collect them here; they are
    # combined after the main loop.
    _patch_t0: Optional[torch.Tensor] = None
    _patch_t1: Optional[torch.Tensor] = None

    for tensor in clip_meta.tensors:
        name = tensor.name
        t, qtype_t = _tensor_to_torch(tensor, target_dtype)

        # Vision encoder uses plain nn.Linear (not GGUFLinear), so every tensor
        # must be floating-point.  gguf-python handles F16/F32/F64 natively but
        # returns BF16 as raw uint8 bytes (BF16 is not in its explicit float list,
        # block_size=1 / type_size=2 → falls to the generic uint8 path).  The
        # result: t.is_floating_point() is False even though it is a "float" type
        # in _FLOAT_TYPE_NAMES, so qtype_t is None and the old check missed it.
        # We therefore gate on t.is_floating_point() and handle two sub-cases:
        #   1. BF16 (or any other raw-bytes float type in _FLOAT_TYPE_NAMES):
        #      reinterpret the uint8 storage as bfloat16, then cast.
        #   2. Truly quantized (Q8_0, Q4_K, …): use gguf.quants.dequantize.
        if not t.is_floating_point():
            if tensor.tensor_type.name in _FLOAT_TYPE_NAMES:
                # Raw bytes for a float type (currently BF16 only): view as bf16.
                t = t.contiguous().view(torch.bfloat16).to(target_dtype)
            else:
                try:
                    from gguf.quants import dequantize as _gguf_dequantize
                    import numpy as _np

                    dq = _gguf_dequantize(tensor.data, tensor.tensor_type)
                    t = torch.from_numpy(_np.array(dq, copy=True)).to(target_dtype)
                except Exception as exc:
                    raise RuntimeError(
                        f"Vision encoder tensor {name!r} is quantized "
                        f"({tensor.tensor_type.name}) and could not be dequantized: {exc}. "
                        "Use a BF16/F16 mmproj GGUF file instead."
                    ) from exc

        # --- Global vision-encoder tensors -----------------------------------
        if name == "v.patch_embd.weight":
            # First temporal slice (or the only slice for Conv2d models).
            # Shape after GGUF ne-reversal: (out_ch, in_ch, kH, kW).
            _patch_t0 = t
            continue

        if name == "v.patch_embd.weight.1":
            # Second temporal slice for Qwen3.5 VL Conv3d.
            _patch_t1 = t
            continue

        if name == "v.patch_embd.bias":
            written["encoder.patch_conv.bias"] = t
            continue

        if name in ("v.position_embd.weight", "v.position_embed.weight"):
            written["encoder.pos_embed.weight"] = t
            continue

        # v.post_ln is the norm inside the Qwen3VL merger (adapter.norm)
        if name == "v.post_ln.weight":
            written["adapter.norm.weight"] = t
            continue
        if name == "v.post_ln.bias":
            written["adapter.norm.bias"] = t
            continue

        # --- Merger (adapter) linear layers ----------------------------------
        # mm.0 → linear_fc1,  mm.2 → linear_fc2
        if name == "mm.0.weight":
            written["adapter.linear_fc1.weight"] = t
            continue
        if name == "mm.0.bias":
            written["adapter.linear_fc1.bias"] = t
            continue
        if name == "mm.2.weight":
            written["adapter.linear_fc2.weight"] = t
            continue
        if name == "mm.2.bias":
            written["adapter.linear_fc2.bias"] = t
            continue

        # --- Per-block vision encoder tensors --------------------------------
        m = re.fullmatch(r"v\.blk\.(\d+)\.(.*)", name)
        if not m:
            skipped.append(name)
            continue

        blk = int(m.group(1))
        suffix = m.group(2)
        pfx = f"encoder.transformer_layers.{blk}"

        if suffix in ("attn_qkv.weight", "attn_qkv.bias"):
            # Fused QKV tensor: split into separate Q / K / V along dim-0.
            # The Q/K/V projection size may differ from hidden_size (e.g. when
            # head_dim * num_heads != hidden_size), so compute the per-component
            # size dynamically from the tensor rather than assuming hidden_size.
            qkv_size = t.shape[0] // 3
            q = t[:qkv_size]
            k = t[qkv_size : 2 * qkv_size]
            v = t[2 * qkv_size :]
            param = ".weight" if suffix.endswith(".weight") else ".bias"
            written[f"{pfx}.self_attn.linear_query{param}"] = q.contiguous()
            written[f"{pfx}.self_attn.linear_keys{param}"] = k.contiguous()
            written[f"{pfx}.self_attn.linear_values{param}"] = v.contiguous()

        elif suffix == "attn_out.weight":
            written[f"{pfx}.self_attn.final_linear.weight"] = t
        elif suffix == "attn_out.bias":
            written[f"{pfx}.self_attn.final_linear.bias"] = t

        elif suffix == "ffn_up.weight":
            # Non-gated MLP → gate_up_proj (single up-projection)
            written[f"{pfx}.mlp.gate_up_proj.weight"] = t
        elif suffix == "ffn_up.bias":
            written[f"{pfx}.mlp.gate_up_proj.bias"] = t
        elif suffix == "ffn_down.weight":
            written[f"{pfx}.mlp.down_proj.weight"] = t
        elif suffix == "ffn_down.bias":
            written[f"{pfx}.mlp.down_proj.bias"] = t

        elif suffix in ("ln1.weight", "ln.weight"):
            written[f"{pfx}.input_layernorm.weight"] = t
        elif suffix in ("ln1.bias", "ln.bias"):
            written[f"{pfx}.input_layernorm.bias"] = t
        elif suffix == "ln2.weight":
            written[f"{pfx}.post_attention_layernorm.weight"] = t
        elif suffix == "ln2.bias":
            written[f"{pfx}.post_attention_layernorm.bias"] = t

        else:
            skipped.append(name)

    # --- Post-loop: assemble temporal patch conv weight ----------------------
    if _patch_t0 is not None:
        if _patch_t1 is not None:
            # Qwen3.5 VL Conv3d: both temporal slices present.
            # Each slice has shape (out_ch, in_ch, kH, kW).
            # Stack along dim=2 → (out_ch, in_ch, 2, kH, kW) for Conv3d.
            written["encoder.patch_conv.weight"] = torch.stack([_patch_t0, _patch_t1], dim=2)
        else:
            # Standard Conv2d: single (out_ch, in_ch, kH, kW) tensor.
            written["encoder.patch_conv.weight"] = _patch_t0

    return written, skipped


def build_vision_encoder_config(
    clip_meta: "GGUFClipMetadata",
    meta: "GGUFMetadata | None" = None,
) -> dict:
    """Build the EOLE ``encoder`` sub-config dict from clip metadata.

    The returned dict is suitable for passing as the ``encoder`` key inside
    a :class:`~eole.config.models.VisionTransformerLMModelConfig` dict.

    Args:
        clip_meta: Parsed GGUF clip/vision encoder metadata.
        meta: Parsed main GGUF metadata (text decoder), used to look up the
            image placeholder token ID from the tokeniser vocabulary.
    """
    hidden = clip_meta.vision_embedding_length
    heads = clip_meta.vision_head_count
    ff = clip_meta.vision_feed_forward_length
    img_size = clip_meta.vision_image_size
    patch_size = clip_meta.vision_patch_size
    norm_eps = clip_meta.vision_norm_eps

    # Number of patches along each spatial dimension (used for position table).
    # The position embedding table has one entry per patch: spatial_size^2.
    # (Note: this is NOT 2×spatial^2 – the tensor shape in GGUF is
    # [hidden_size, spatial^2] with ne-reversal giving (spatial^2, hidden_size).)
    spatial_size = img_size // patch_size
    num_pos = spatial_size * spatial_size

    # Image placeholder token ID.
    # For Qwen3.5 VL the placeholder is <|image_pad|> (token 151655 in the
    # standard Qwen3.5 vocabulary).  Search the GGUF token list first so the
    # correct ID is used if the vocabulary differs; fall back to the known
    # Qwen3.5 default if meta is unavailable or the token is not found.
    _QWEN35VL_IMAGE_TOKEN_ID = 151655
    image_token_id = _QWEN35VL_IMAGE_TOKEN_ID
    if meta is not None:
        _IMAGE_PAD_TOKENS = {"<|image_pad|>", "<image_pad>"}
        for tok_id, tok_str in enumerate(meta.tokens):
            if tok_str in _IMAGE_PAD_TOKENS:
                image_token_id = tok_id
                break

    return {
        "layers": clip_meta.vision_block_count,
        "hidden_size": hidden,
        "heads": heads,
        "heads_kv": heads,
        # Explicitly set so the encoder never inherits the text decoder's head_dim
        # (e.g. Qwen3.5 VL text decoder uses head_dim=256 while the vision encoder
        # uses head_dim = hidden_size // heads = 1152 // 16 = 72).
        "head_dim": hidden // heads,
        "transformer_ff": ff,
        "add_qkvbias": True,
        "add_ffnbias": True,
        "add_final_linear_bias": True,
        "layer_norm": "standard",
        "norm_eps": norm_eps,
        "mlp_activation_fn": "gelu-tanh",
        "image_size": img_size,
        "patch_size": patch_size,
        "layernorm_pre": False,
        "layernorm_post": False,
        "temporal_patch_size": 2,  # Qwen3.5 VL uses Conv3d with kernel (2,patch,patch)
        "patch_conv_bias": True,
        "num_position_embeddings": num_pos,
        # Vision encoder uses standard 2D RoPE (pixtral-style), NOT the decoder's
        # multi-dimensional MRoPE.  rotary_interleave must be False and theta=10000
        # (matches convert_HF.py for Qwen3_5ForConditionalGeneration).
        "position_encoding_type": PositionEncodingType.Rotary,
        "rope_config": {
            "rotary_interleave": False,
            "rotary_theta": 10000,
        },
        # Image placeholder token ID: embed_vision_language_features uses this
        # to separate image tokens from text tokens in the combined sequence.
        "image_token_id": image_token_id,
    }


# ---------------------------------------------------------------------------
# EOLE config builder
# ---------------------------------------------------------------------------


def build_eole_config(
    meta: GGUFMetadata,
    vocabs: dict,
    src_vocab,
    model_config_dict: dict,
    training_config_dict: dict,
    transforms: list,
    transforms_configs: dict,
    optional_eos: list,
) -> dict:
    """Assemble and return the EOLE ``config.json`` dictionary.

    When *model_config_dict* contains an ``"encoder"`` key the function uses
    :class:`~eole.config.models.VisionTransformerLMModelConfig` (multimodal
    model) instead of :class:`~eole.config.models.TransformerLMModelConfig`.
    """
    if "encoder" in model_config_dict:
        from eole.config.models import VisionTransformerLMModelConfig

        model_cls = VisionTransformerLMModelConfig
    else:
        from eole.config.models import TransformerLMModelConfig

        model_cls = TransformerLMModelConfig

    config = TrainConfig(
        data=None,
        skip_empty_level="silent",
        save_data=None,
        n_sample=0,
        src_vocab=None,
        tgt_vocab=None,
        share_vocab=True,
        src_vocab_size=meta.vocab_size,
        tgt_vocab_size=meta.vocab_size,
        vocab_size_multiple=8,
        decoder_start_token=vocabs.get("decoder_start_token", ""),
        **{k: v for k, v in vocabs.get("specials", {}).items()},
        transforms=transforms,
        transforms_configs=transforms_configs,
        model=model_cls(**model_config_dict),
        training=TrainingConfig(**{k: v for k, v in training_config_dict.items() if k not in ("compute_dtype",)}),
    )

    config_dict = recursive_model_fields_set(config)
    config_dict["inference"] = {"optional_eos": optional_eos}
    if meta.chat_template:
        config_dict["inference"]["chat_template"] = meta.chat_template
    return config_dict


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


@register_bin(name="GGUF")
class GGUFConverter(BaseBin):
    """Convert a GGUF model file to EOLE format.

    Quantised weights are kept in their native compact binary form and stored as
    ``uint8`` tensors in the output safetensors shard – no loss of quantisation
    precision.  At inference time :class:`~eole.modules.gguf_linear.GGUFLinear`
    dequantizes them on-the-fly exactly as the GPTQ/Marlin backend does for
    its packed int4 weights.
    """

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--gguf_path",
            type=str,
            required=True,
            help="Path to the input GGUF file.",
        )
        parser.add_argument(
            "--output",
            type=str,
            required=True,
            help="Path to the output directory (created if absent).",
        )
        parser.add_argument(
            "--dtype",
            type=str,
            default="fp16",
            choices=list(TORCH_DTYPES.keys()),
            help="Dtype to use for non-quantised (float) tensors (default: fp16).",
        )
        parser.add_argument(
            "--tokenizer",
            type=str,
            default="hf",
            choices=["hf", "onmt"],
            help="Tokenizer transform to embed in the EOLE config (default: hf).",
        )
        parser.add_argument(
            "--hf_tokenizer",
            type=str,
            default=None,
            help=(
                "Optional path to a HuggingFace tokenizer.json file (or directory "
                "containing one).  When provided the HF tokenizer is used directly "
                "instead of extracting it from the GGUF metadata."
            ),
        )
        parser.add_argument(
            "--mmproj",
            type=str,
            default=None,
            help=(
                "Optional path to a separate clip/mmproj GGUF file containing the "
                "vision encoder (e.g. mmproj-BF16.gguf for Qwen3.5 VL).  When "
                "provided the vision tensors are merged into the output safetensors "
                "shard and the config uses VisionTransformerLMModelConfig."
            ),
        )

    @classmethod
    def run(cls, args):
        if safetensors_save_file is None:
            raise ImportError("Install safetensors: pip install safetensors")
        try:
            import gguf  # noqa: F401
        except ImportError:
            raise ImportError("Install gguf: pip install gguf")

        os.makedirs(args.output, exist_ok=True)
        target_dtype = TORCH_DTYPES.get(args.dtype, torch.float16)

        print(f"Reading GGUF metadata from {args.gguf_path} …")
        meta = GGUFMetadata(args.gguf_path)
        print(f"  Architecture : {meta.arch}")
        print(f"  Layers       : {meta.block_count}")
        print(f"  Hidden size  : {meta.embedding_length}")
        print(f"  Heads        : {meta.head_count}")
        if meta.head_count_kv:
            print(f"  KV heads     : {meta.head_count_kv}")
        if meta.expert_count:
            print(f"  Experts      : {meta.expert_count} (used: {meta.expert_used_count})")
        print(f"  Tensors in GGUF : {len(meta.tensors)}")

        # ------------------------------------------------------------------
        # Optional vision encoder (mmproj GGUF)
        # ------------------------------------------------------------------
        clip_meta = None
        mmproj_tensors: dict = {}
        if getattr(args, "mmproj", None):
            print(f"Reading mmproj (vision encoder) from {args.mmproj} …")
            clip_meta = GGUFClipMetadata(args.mmproj)
            print(f"  Projector    : {clip_meta.projector_type}")
            print(f"  Enc layers   : {clip_meta.vision_block_count}")
            print(f"  Enc hidden   : {clip_meta.vision_embedding_length}")
            print(f"  Image size   : {clip_meta.vision_image_size}")
            print(f"  Tensors in GGUF : {len(clip_meta.tensors)}")
            mmproj_tensors, mmproj_skipped = _mmproj_to_eole_tensors(clip_meta, target_dtype)
            print(f"  Converted {len(mmproj_tensors)} vision tensors → EOLE keys")
            if mmproj_skipped:
                print(f"  Skipped {len(mmproj_skipped)}: {mmproj_skipped[:5]}")

        # Detect whether the output.weight tensor is missing (tied embeddings)
        tensor_names = {t.name for t in meta.tensors}
        share_decoder_embeddings = "output.weight" not in tensor_names
        print(f"  Shared emb   : {share_decoder_embeddings}")

        # ------------------------------------------------------------------
        # Tokenizer
        # ------------------------------------------------------------------
        vocabs: dict = {"specials": {}}
        optional_eos: list = []
        transforms_configs: dict = {}
        transforms: list = []

        hf_tok_path = args.hf_tokenizer
        if hf_tok_path is not None and os.path.isdir(hf_tok_path):
            candidate = os.path.join(hf_tok_path, "tokenizer.json")
            if os.path.exists(candidate):
                hf_tok_path = candidate

        # Helper: build vocab from HF tokenizer.json dict
        def _vocab_from_hf_tok(tok_data: dict, vocabs: dict) -> tuple:
            vlist = list(tok_data.get("model", {}).get("vocab", {}).keys())
            if isinstance(vlist, dict):
                vlist = list(vlist.keys())
            declared = meta.vocab_size
            while len(vlist) < declared:
                vlist.append(f"{DefaultTokens.VOCAB_PAD}{len(vlist)}")
            for tok in tok_data.get("added_tokens", []):
                idx = tok["id"]
                if 0 <= idx < len(vlist):
                    vlist[idx] = tok["content"]
            for field in ["bos_token", "eos_token", "unk_token", "pad_token"]:
                val = tok_data.get(field)
                if isinstance(val, str):
                    vocabs["specials"][field] = val
                elif isinstance(val, dict):
                    vocabs["specials"][field] = val.get("content", "")
            src_v = pyonmttok.build_vocab_from_tokens(vlist) if pyonmttok else None
            return src_v, "tokenizer.json"

        if hf_tok_path is not None and os.path.isfile(hf_tok_path):
            import shutil

            dest_tok = os.path.join(args.output, "tokenizer.json")
            if os.path.normpath(hf_tok_path) != os.path.normpath(dest_tok):
                shutil.copy2(hf_tok_path, dest_tok)
            with open(dest_tok, encoding="utf-8") as fh:
                tok_data = json.load(fh)
            src_vocab, tok_basename = _vocab_from_hf_tok(tok_data, vocabs)
            transforms = ["huggingface_tokenize"]
            transforms_configs["huggingface_tokenize"] = {"path": dest_tok}

        elif meta.hf_tokenizer_json is not None:
            dest_tok = os.path.join(args.output, "tokenizer.json")
            tok_data = json.loads(meta.hf_tokenizer_json)
            with open(dest_tok, "w", encoding="utf-8") as fh:
                json.dump(tok_data, fh, indent=2, ensure_ascii=False)
            print(f"  Extracted embedded tokenizer.json → {dest_tok}")
            src_vocab, tok_basename = _vocab_from_hf_tok(tok_data, vocabs)
            transforms = ["huggingface_tokenize"]
            transforms_configs["huggingface_tokenize"] = {"path": dest_tok}

        else:
            print("  Extracting tokenizer from GGUF …")
            src_vocab, tok_basename, gpt2_pretok = _extract_vocab_from_gguf(meta, vocabs, args.output)
            if args.tokenizer == "hf":
                tok_path = os.path.join(args.output, "tokenizer.json")
                if os.path.exists(tok_path):
                    transforms = ["huggingface_tokenize"]
                    transforms_configs["huggingface_tokenize"] = {"path": tok_path}
                else:
                    subword_type = "bpe" if gpt2_pretok else "sentencepiece"
                    transforms = ["onmt_tokenize", "filtertoolong"]
                    transforms_configs = {
                        "filtertoolong": {"src_seq_length": 512, "tgt_seq_length": 512},
                        "onmt_tokenize": {
                            "src_subword_type": subword_type,
                            "src_subword_model": os.path.join("${MODEL_PATH}", tok_basename),
                            "gpt2_pretok": gpt2_pretok,
                        },
                    }
            else:
                subword_type = "bpe" if gpt2_pretok else "sentencepiece"
                transforms = ["onmt_tokenize", "filtertoolong"]
                transforms_configs = {
                    "filtertoolong": {"src_seq_length": 512, "tgt_seq_length": 512},
                    "onmt_tokenize": {
                        "src_subword_type": subword_type,
                        "src_subword_model": os.path.join("${MODEL_PATH}", tok_basename),
                        "gpt2_pretok": gpt2_pretok,
                    },
                }

        # Optional EOS tokens
        eos_id = meta.eos_token_id
        tokens = meta.tokens
        for tid, ttype in enumerate(meta.token_types):
            if ttype == _TOKEN_TYPE_CONTROL and tid != eos_id and tid < len(tokens):
                optional_eos.append(tokens[tid])

        # Fill any special tokens still missing from GGUF token-ID metadata.
        # This is needed when the tokenizer came from a HF tokenizer.json that
        # stores specials only in tokenizer_config.json (not in tokenizer.json).
        _special_id_props = {
            "bos_token": meta.bos_token_id,
            "eos_token": meta.eos_token_id,
            "unk_token": meta.unk_token_id,
            "pad_token": meta.pad_token_id,
        }
        for field, tok_id in _special_id_props.items():
            if field not in vocabs["specials"] and tok_id is not None and tok_id < len(tokens):
                vocabs["specials"][field] = tokens[tok_id]

        # Decoder start token
        if meta.add_bos_token and "bos_token" in vocabs["specials"]:
            vocabs["decoder_start_token"] = vocabs["specials"]["bos_token"]
        else:
            vocabs["decoder_start_token"] = ""

        # Write vocab files
        if src_vocab is not None:
            vocabs["src"] = src_vocab
            vocabs["tgt"] = src_vocab
            vocab_dict = vocabs_to_dict(vocabs)
            with open(os.path.join(args.output, "vocab.json"), "w", encoding="utf-8") as fh:
                json.dump(vocab_dict, fh, indent=2, ensure_ascii=False)
            with open(os.path.join(args.output, "vocab.txt"), "w", encoding="utf-8") as fh:
                for tok in vocab_dict["src"]:
                    fh.write(tok.replace("\n", DefaultTokens.SEP) + "\n")

        # ------------------------------------------------------------------
        # Detect hybrid-model block layout (linear vs. full attention)
        # ------------------------------------------------------------------
        linear_blocks = _detect_linear_attention_blocks(meta.tensors)
        if linear_blocks:
            shown = sorted(linear_blocks)[:5]
            suffix = "…" if len(linear_blocks) > 5 else ""
            print(f"  Linear-attention blocks: {shown}{suffix}  ({len(linear_blocks)} total)")

        # ------------------------------------------------------------------
        # Convert tensors (keep quantised as uint8, cast floats to target_dtype)
        # ------------------------------------------------------------------
        print("Converting tensors …")
        written, quant_layers = build_safetensors(
            meta,
            args.output,
            target_dtype,
            linear_blocks,
            extra_tensors=mmproj_tensors or None,
            has_vision_encoder=(clip_meta is not None),
        )

        # ------------------------------------------------------------------
        # Build EOLE config
        # ------------------------------------------------------------------
        model_config_dict = build_model_config(meta, linear_blocks)
        model_config_dict["share_decoder_embeddings"] = share_decoder_embeddings

        # When a vision encoder was provided, inject the encoder config and
        # adapter type so that VisionTransformerLMModelConfig is produced.
        if clip_meta is not None:
            model_config_dict["encoder"] = build_vision_encoder_config(clip_meta, meta)
            model_config_dict["adapter"] = "qwen3_5vl"
            # spatial_merge_size controls how many vision patches are grouped
            # before projection.  Qwen3.5 VL uses spatial_merge_size=2, giving
            # merged_size = hidden_size * 4.  ModelConfig defaults to 1 which
            # would create linear_fc1 of shape (hidden_size, hidden_size) instead
            # of the correct (4*hidden_size, 4*hidden_size), causing a shape
            # mismatch when loading the adapter weights.
            model_config_dict["spatial_merge_size"] = clip_meta.vision_spatial_merge_size

        quant_layers_all = [
            "gate_up_proj",
            "down_proj",
            "up_proj",
            "linear_values",
            "linear_query",
            "linear_keys",
            "final_linear",
        ]
        if linear_blocks:
            # GatedDeltaNet linear layers that may carry quantised weights.
            quant_layers_all += ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]
        if clip_meta is None:
            # Plain text model: replace_gguf_linear targets the whole model
            # (quant_target = self), so self.generator is reachable and can be
            # replaced with GGUFLinear.  Include it in quant_layers so the
            # quantised generator.weight is loaded correctly without
            # dequantising to float (which would inflate the file size by ~3x
            # for models with large vocabularies quantised to Q4_K).
            quant_layers_all += ["generator"]

        training_config_dict: dict = {
            "quant_type": "gguf" if quant_layers else "",
            "quant_layers": quant_layers_all if quant_layers else [],
            "w_bit": 0,
            "group_size": 0,
            "compute_dtype": args.dtype,
        }

        print("Building config.json …")
        config_dict = build_eole_config(
            meta=meta,
            vocabs=vocabs,
            src_vocab=src_vocab,
            model_config_dict=model_config_dict,
            training_config_dict=training_config_dict,
            transforms=transforms,
            transforms_configs=transforms_configs,
            optional_eos=optional_eos,
        )

        config_path = os.path.join(args.output, "config.json")
        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump(config_dict, fh, indent=2, ensure_ascii=False)

        # Safetensors entry breakdown: weight + gguf_qtype scalars + vision
        n_qtype_scalars = sum(1 for k in written if k.endswith(".gguf_qtype"))
        n_vision_entries = len(mmproj_tensors) if mmproj_tensors else 0
        n_decoder_entries = len(written) - n_vision_entries
        print(f"\nConversion complete → {args.output}")
        print("  config.json")
        if src_vocab is not None:
            print("  vocab.json / vocab.txt")
        print(f"  model.00.safetensors  ({len(written)} total entries):")
        print(
            f"    {n_decoder_entries} decoder entries "
            f"(includes {n_qtype_scalars} gguf_qtype scalars for quantized layers)"
        )
        if n_vision_entries:
            print(f"    {n_vision_entries} vision encoder entries")
        if quant_layers:
            print(f"  Quantized layers ({len(quant_layers)}): {', '.join(quant_layers)}")
