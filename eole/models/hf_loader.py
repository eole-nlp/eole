"""In-memory HuggingFace model loader for direct inference without pre-conversion.

This module provides :class:`HFLoader` which performs the same tensor remapping
and config translation as ``eole_convert HF``, but entirely in memory so that
inference can be run directly from a HuggingFace model ID or local HF directory
without a separate conversion step.

:func:`build_train_config` is the shared config-and-vocab building function used by
both the in-memory inference path (:class:`HFLoader`) and the disk-based conversion
tool (:class:`~eole.bin.convert.convert_HF.LlamaHFConverter`).

The public entry point for the inference pipeline is :func:`load_hf_model`.
"""

import os
from types import SimpleNamespace

import safetensors
import torch

from eole.bin.convert.convert_HF import (
    HuggingfaceFiles,
    build_config_dict,
    check_tokenizer_config,
    check_special_tokens,
    check_sentencepiece_tokenizer,
    check_bpe_tokenizer,
    check_generation_config,
    get_weight,
    get_huggingface_model,
    get_shards_map,
)
from eole.bin.convert.HF_mappings import KEY_MAPS, ARCH_TABLE, TOK_TABLE
from eole.config.run import TrainConfig
from eole.config.training import TrainingConfig
from eole.inputters.inputter import vocabs_to_dict, dict_to_vocabs
from eole.utils.logging import logger


class _InMemoryTensorStore:
    """Mimics the ``safetensors.safe_open`` interface but keeps tensors in RAM.

    An instance of this class is passed to
    :meth:`~eole.models.model.BaseModel._load_parameters_by_module` in place of
    a real safetensors file handle so that the existing weight-loading machinery
    can be reused without modification.
    """

    def __init__(self):
        self._tensors = {}

    def __setitem__(self, name: str, tensor: torch.Tensor):
        self._tensors[name] = tensor.contiguous()

    def __contains__(self, name: str) -> bool:
        return name in self._tensors

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._tensors[name]

    def keys(self):
        return self._tensors.keys()


def _hf_cache_dir(model_id: str) -> str:
    """Return a stable cache directory for a given HF model ID.

    Downloaded tokenizer / config artifacts (e.g. BPE vocab files) are written
    here so they survive across Python sessions without re-downloading.
    """
    slug = model_id.replace("/", "--").replace("\\", "--")
    return os.path.join(os.path.expanduser("~"), ".cache", "eole", "hf", slug)


def build_train_config(hf, output_dir, tokenizer_pref="hf", dtype=None, use_env_path=False):
    """Build the EOLE :class:`~eole.config.run.TrainConfig` and vocabulary from
    a :class:`~eole.bin.convert.convert_HF.HuggingfaceFiles` instance.

    This is the shared config-and-vocab building logic used by both the
    in-memory inference path (:class:`HFLoader`) and the disk-based conversion
    tool (:class:`~eole.bin.convert.convert_HF.LlamaHFConverter`).

    Args:
        hf: :class:`~eole.bin.convert.convert_HF.HuggingfaceFiles` instance.
        output_dir: Directory where tokenizer artifacts (e.g. BPE model file)
            are written.
        tokenizer_pref: ``"hf"`` to use the HuggingFace tokenizer transform;
            ``"onmt"`` to use the ONMT tokenizer transform.
        dtype: Optional compute-dtype string (e.g. ``"fp16"``). Falls back to
            the model's ``torch_dtype`` field and then to ``"fp16"``.
        use_env_path: When ``True``, the ONMT tokenizer model path in the
            produced transforms config uses the ``${MODEL_PATH}`` environment
            variable prefix — suitable for on-disk ``config.json`` files that
            must be portable.  When ``False`` (the default) an absolute path
            rooted at *output_dir* is used instead.

    Returns:
        tuple: ``(train_config, vocabs, params, model_config_dict, inference_dict)``

        * *train_config* – :class:`~eole.config.run.TrainConfig`.
        * *vocabs* – ``dict`` with ``src``, ``tgt``, ``specials``, and
          ``decoder_start_token`` entries.  ``vocabs["src"]`` and
          ``vocabs["tgt"]`` are the pyonmttok vocabulary objects.
        * *params* – list of HF parameter suffixes to extract
          (``["weight", "bias"]`` or quantised equivalents).
        * *model_config* – raw ``dict`` produced by
          :func:`~eole.bin.convert.convert_HF.build_config_dict`; needed
          by :func:`~eole.bin.convert.convert_HF.build_shards` and by
          :meth:`HFLoader._build_tensors`.
        * *inference_dict* – ``dict`` of inference-related settings
          (``optional_eos``, ``chat_template``, generation params).
    """
    model_config, training_config, params = build_config_dict(hf)

    compute_dtype = dtype or hf.config.get("torch_dtype") or "fp16"
    gpt2_pretok = False

    (add_bos_token, chat_template, optional_eos, mapped_tokens) = check_tokenizer_config(hf)
    vocabs = check_special_tokens(hf)

    if hf.tokenizer_model is not None:
        src_subword_type = "sentencepiece"
        src_vocab, tokenizer_basename = check_sentencepiece_tokenizer(hf)
    else:
        src_subword_type = "bpe"
        src_vocab, tokenizer_basename, vocabs, gpt2_pretok = check_bpe_tokenizer(hf, vocabs, output_dir)

    # Set decoder start token
    if hf.config.get("decoder_start_token_id") is not None:
        vocabs["decoder_start_token"] = src_vocab.ids_to_tokens[hf.config["decoder_start_token_id"]]
    elif add_bos_token:
        vocabs["decoder_start_token"] = vocabs["specials"].get("bos_token", "")
    else:
        vocabs["decoder_start_token"] = ""

    vocabs["src"] = src_vocab
    vocabs["tgt"] = src_vocab

    # Configure transforms
    if use_env_path:
        tokenizer_model_path = os.path.join("${MODEL_PATH}", tokenizer_basename)
    else:
        tokenizer_model_path = os.path.join(output_dir, tokenizer_basename)

    match TOK_TABLE[hf.arch], tokenizer_pref:
        case "huggingface_tokenize", "hf":
            transforms = ["huggingface_tokenize"]
            transforms_configs = {
                TOK_TABLE[hf.arch]: {"path": hf.tokenizer_json},
            }
        case _:
            transforms = ["onmt_tokenize", "filtertoolong"]
            transforms_configs = {
                "filtertoolong": {"src_seq_length": 512, "tgt_seq_length": 512},
                "onmt_tokenize": {
                    "src_subword_type": src_subword_type,
                    "src_subword_model": tokenizer_model_path,
                    "gpt2_pretok": gpt2_pretok,
                    "mapped_tokens": mapped_tokens,
                },
            }

    generation_config_dict = check_generation_config(hf)

    # Build an args-like namespace so get_huggingface_model() can read model_dir
    hf_args = SimpleNamespace(model_dir=hf.model_dir)

    train_config = TrainConfig(
        data=None,
        skip_empty_level="silent",
        save_data=None,
        n_sample=0,
        src_vocab=None,
        tgt_vocab=None,
        share_vocab=True,
        src_vocab_size=hf.vocab_size,
        tgt_vocab_size=hf.vocab_size,
        vocab_size_multiple=8,
        decoder_start_token=vocabs["decoder_start_token"],
        **vocabs["specials"],
        transforms=transforms,
        transforms_configs=transforms_configs,
        model=ARCH_TABLE[hf.arch](
            **model_config,
            huggingface_model=get_huggingface_model(hf_args),
        ),
        training=TrainingConfig(
            compute_dtype=compute_dtype,
            batch_size=896,
            batch_size_multiple=1,
            batch_type="tokens",
            normalization="tokens",
            accum_count=[32],
            accum_steps=[0],
            valid_batch_size=256,
            **training_config,
        ),
    )

    inference_dict = {
        "optional_eos": optional_eos,
        **generation_config_dict,
        **chat_template,
    }

    return train_config, vocabs, params, model_config, inference_dict


class HFLoader:
    """Convert a HuggingFace model into EOLE in-memory data structures.

    This is the in-memory equivalent of the disk-based ``eole_convert HF``
    pipeline. It fetches (or reuses cached) HF artifacts, translates the model
    config, builds the vocabulary, and remaps all weight tensors — all in RAM.

    Usage::

        train_config, vocabs_dict, f, keys_shard = HFLoader.load(model_id, token=token)

    The returned ``f`` and ``keys_shard`` have the same interface as those
    produced by :meth:`~eole.models.model.BaseModel._load_safetensors_shards`
    and can be passed directly to
    :meth:`~eole.models.model.BaseModel._load_parameters_by_module`.
    """

    def __init__(self, model_id: str, token: str | None = None):
        self.model_id = model_id
        self.token = token

    @classmethod
    def load(cls, model_id: str, token: str | None = None):
        """Load a HF model into EOLE in-memory structures.

        Args:
            model_id: HuggingFace model ID (e.g. ``"meta-llama/Llama-3.1-8B"``
                or a local directory containing HF-format files.
            token: Optional HuggingFace access token for gated models.

        Returns:
            tuple: ``(train_config, vocabs_dict, f, keys_shard)`` where

            * *train_config* – :class:`~eole.config.run.TrainConfig` with model
              and training settings derived from the HF config.
            * *vocabs_dict* – vocabulary dict (same format as ``vocab.json``).
            * *f* – ``[_InMemoryTensorStore]`` — a one-element list whose sole
              item holds all remapped weight tensors.
            * *keys_shard* – ``{tensor_key: 0}`` mapping every key to shard
              index 0 (the single in-memory store).
        """
        return cls(model_id, token=token)._load()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_fetch_args(self, cache_dir: str) -> SimpleNamespace:
        """Build the minimal ``args`` namespace expected by :func:`HuggingfaceFiles.fetch`."""
        return SimpleNamespace(
            model_dir=self.model_id,
            token=self.token,
            output=cache_dir,
        )

    def _load(self):
        cache_dir = _hf_cache_dir(self.model_id)
        os.makedirs(cache_dir, exist_ok=True)
        args = self._make_fetch_args(cache_dir)
        hf = HuggingfaceFiles.fetch(args)
        train_config, vocabs, params, model_config, inference_dict = build_train_config(hf, cache_dir)

        # Attach inference settings so callers can propagate them if needed
        train_config.__dict__["_hf_inference_dict"] = inference_dict

        vocabs_dict = vocabs_to_dict(vocabs)

        # Build in-memory tensor store with remapped weights
        store = _InMemoryTensorStore()
        self._build_tensors(hf, model_config, params, store)

        keys_shard = {key: 0 for key in store.keys()}
        return train_config, vocabs_dict, [store], keys_shard

    def _build_tensors(self, hf, model_config, params, store: _InMemoryTensorStore):
        """Remap all HF checkpoint tensors and write them into *store*.

        This mirrors the logic in :func:`~eole.bin.convert.convert_HF.build_shards`
        but targets an in-memory store instead of disk-based safetensors shards.
        All layers are treated as a single shard (``nshards=1``).
        """
        # Use a single logical shard — all layers mapped together
        shard_checkpoints, shard_layer_ranges = get_shards_map(model_config, hf, 1)

        # ---- Global tensors (embeddings, layer-norm, generator) ---------------
        for target in KEY_MAPS[hf.arch].keys():
            if model_config.get("share_decoder_embeddings") and target == "generator.weight":
                continue
            source = KEY_MAPS[hf.arch][target]
            srckey, srcmap = source if isinstance(source, tuple) else (source, None)
            if not isinstance(srckey, str):
                continue

            if hf.wmap_path:
                weight_map = hf.wmap["weight_map"]
                if srckey in weight_map:
                    checkpoint = hf.get_load_ckpt(hf.base_dir, weight_map[srckey])
                else:
                    checkpoint = None
            else:
                checkpoint = hf.get_load_ckpt(*os.path.split(hf.model_path))

            if checkpoint is None:
                continue

            w = get_weight(checkpoint, srckey)
            if w is not None:
                context = {
                    "hidden_size": model_config["hidden_size"],
                    "transformer_ff": model_config["transformer_ff"],
                }
                if srcmap is not None:
                    w = eval("w" + srcmap, {"w": w, **context}).contiguous()
                if target == "encoder.class_embedding.weight":
                    store[target] = w.unsqueeze(0)
                else:
                    store[target] = w

        # ---- Per-layer tensors ------------------------------------------------
        for ckpt in shard_checkpoints[0]:
            checkpoint = hf.checkpoint(ckpt)
            if isinstance(checkpoint, dict):
                ckpt_keys = set(checkpoint.keys())
            else:
                with safetensors.safe_open(checkpoint, framework="pt", device="cpu") as f:
                    ckpt_keys = set(f.keys())

            for i in shard_layer_ranges[0]:
                _layer_cache: dict = {}
                prefix_mapping = (
                    ("encoder", hf.encoder_layer_prefix, "encoder.transformer_layers."),
                    ("encoder.sam", hf.encoder_sam_layer_prefix, "encoder.sam.blocks."),
                    ("decoder", hf.decoder_layer_prefix, "decoder.transformer_layers."),
                )
                for section, hf_prefix, eole_prefix in prefix_mapping:
                    if hf_prefix is None:
                        continue
                    key_map = KEY_MAPS[hf.arch].get(section, {})
                    for target, source in key_map.items():
                        for param in params:
                            if not isinstance(source, (str, tuple)):
                                continue
                            srckey, srcmap = source if isinstance(source, tuple) else (source, None)
                            if srckey.endswith("."):
                                srckey = srckey + param
                            full_srckey = hf_prefix + str(i) + srckey
                            if full_srckey not in _layer_cache:
                                if full_srckey not in ckpt_keys:
                                    _layer_cache[full_srckey] = None
                                else:
                                    _layer_cache[full_srckey] = get_weight(checkpoint, full_srckey)
                            w = _layer_cache[full_srckey]
                            if w is None:
                                continue

                            if srcmap is not None:
                                hidden_size = (
                                    model_config["hidden_size"]
                                    if section.startswith("decoder")
                                    else model_config["encoder"]["hidden_size"]
                                )
                                context = {
                                    "hidden_size": hidden_size,
                                    "transformer_ff": model_config["transformer_ff"],
                                    "moe_transformer_ff": model_config.get("decoder", {}).get(
                                        "moe_transformer_ff",
                                        model_config.get("moe_transformer_ff", 0),
                                    ),
                                }
                                w = eval("w" + srcmap, {"w": w, **context}).contiguous()

                            target1 = target
                            if target.endswith("."):
                                target1 = target + param
                            eole_key = eole_prefix + str(i) + target1
                            if eole_key not in store:
                                store[eole_key] = w

                _layer_cache.clear()

        # Post-process: Gemma4-specific weight fixups.
        if hf.arch in ("Gemma4ForCausalLM", "Gemma4ForConditionalGeneration"):
            text_config = hf.config.get("text_config", hf.config)

            # For full_attention layers with attention_k_eq_v=True, v_proj is None in HF
            # (value states reuse key states), so tie linear_values to linear_keys.
            if text_config.get("attention_k_eq_v", False):
                layer_types = model_config.get("decoder", {}).get("layer_types", [])
                for i, lt in enumerate(layer_types):
                    if lt == "full_attention":
                        for param in params:
                            k_key = f"decoder.transformer_layers.{i}.self_attn.linear_keys.{param}"
                            v_key = f"decoder.transformer_layers.{i}.self_attn.linear_values.{param}"
                            if k_key in store and v_key not in store:
                                store[v_key] = store.get_tensor(k_key)

            # Gemma4MultiModalProjector uses RMSNormNoScale (no learnable weight):
            # do NOT inject any adapter.norm.weight — the module has no such parameter.


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_hf_model(running_config, device_id: int = 0):
    """Build and return an EOLE model loaded directly from a HuggingFace source.

    This is the high-level entry point called from
    :func:`~eole.predict.build_predictor` when the model path is detected as a
    HuggingFace model ID rather than a local EOLE model directory.

    Args:
        running_config: :class:`~eole.config.run.PredictConfig` whose
            ``model_path[0]`` is either a HuggingFace Hub model ID
            (``"org/model"``) or a local directory containing HF-format files.
        device_id: GPU/CPU device index passed to
            :meth:`~eole.models.model.BaseModel._setup_device_and_offset`.

    Returns:
        tuple: ``(model, vocabs, model_config)`` — same shape as the tuple
        returned by :meth:`~eole.models.model.BaseModel.for_inference`.
    """
    from eole.models.model import get_model_class
    from eole.config.data import NestedAllTransformsConfig

    model_id = running_config.model_path[0]
    token = getattr(running_config, "hf_token", None)

    logger.info(f"Loading HuggingFace model directly: {model_id}")

    train_config, vocabs_dict, f, keys_shard = HFLoader.load(model_id, token=token)
    model_config = train_config.model

    # ---- Propagate model and quantisation settings into running_config --------
    update_dict = {"model": model_config}
    for field in (
        "quant_type",
        "quant_layers",
        "quant_exclude_modules",
        "autoround_packing_format",
        "autoround_sym",
    ):
        if field not in running_config.model_fields_set:
            val = getattr(train_config.training, field, None)
            if val is not None:
                update_dict[field] = val

    # Propagate compute_dtype from HF config when not explicitly set
    if "compute_dtype" not in running_config.model_fields_set:
        update_dict["compute_dtype"] = train_config.training.compute_dtype

    # Propagate transforms when not explicitly set by the user
    if "transforms" not in running_config.model_fields_set:
        update_dict["transforms"] = list(train_config.transforms or [])
        update_dict["_all_transform"] = list(train_config.transforms or [])

    if "transforms_configs" not in running_config.model_fields_set:
        tc = train_config.transforms_configs
        if tc is not None:
            from eole.config import recursive_model_fields_set

            tc_dict = recursive_model_fields_set(tc)
        else:
            tc_dict = {}
        update_dict["transforms_configs"] = NestedAllTransformsConfig(**tc_dict)

    # Propagate inference settings from HF generation_config / tokenizer_config
    hf_inference = getattr(train_config, "_hf_inference_dict", {})
    for key, value in hf_inference.items():
        if key not in running_config.model_fields_set and hasattr(running_config, key):
            update_dict[key] = value

    running_config.update(**update_dict)

    # ---- Build model architecture --------------------------------------------
    vocabs = dict_to_vocabs(vocabs_dict)
    model_class = get_model_class(model_config)
    model = model_class.build(model_config, vocabs, running_config)

    # ---- Setup device --------------------------------------------------------
    device, tp_offset = model._setup_device_and_offset(running_config, device_id)

    # ---- Load weights from in-memory store -----------------------------------
    logger.info("Loading weights from HuggingFace model into EOLE model")
    dtype = getattr(running_config, "storage_dtype", torch.float32)
    if device == torch.device("cpu"):
        tp_offset = 0
    buf_list = [name for name, _ in model.named_buffers()]

    keyfound = model._load_parameters_by_module(
        f, keys_shard, {}, buf_list, device, dtype, tp_offset, True, running_config
    )
    model._report_extra_keys(keys_shard, keyfound, buf_list)
    model._reset_lora_to_fp32()
    model._reset_invfreq_to_fp32(buf_list)

    # ---- Inference-specific finalization ------------------------------------
    model._prepare_for_inference(running_config)

    return model, vocabs, model_config
