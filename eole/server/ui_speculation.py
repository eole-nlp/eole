"""Trusted model-side speculation options for workspace engine selection."""

import json
import os
from pathlib import Path

from eole.config.inference import InferenceConfig


def defaults(model):
    configured = getattr(model, "speculation_defaults", None)
    if configured is not None:
        return configured
    # Lightweight server integrations may not use Model directly.
    config = getattr(model, "pre_config", {})
    engine = getattr(getattr(model, "engine", None), "config", None)
    path = config.get("dflash2_model", getattr(engine, "dflash2_model", None))
    mtp = config.get("self_speculative_decoding", getattr(engine, "self_speculative_decoding", False))
    return dict(
        mode="dflash2" if path else "mtp" if mtp else "off",
        mtp=bool(mtp),
        path=os.path.expanduser(os.path.expandvars(path)) if path else None,
        quantization=config.get("dflash2_quantization", "int8"),
    )


def capabilities(model):
    configured = defaults(model)
    path = configured["path"]
    reason = "No DFlash2 checkpoint configured for this model."
    available, drafts = False, None
    if path:
        folder = Path(os.path.expandvars(path)).expanduser()
        try:
            metadata = json.loads((folder / "config.json").read_text())
            if not isinstance(metadata, dict) or not isinstance(metadata.get("dflash_config"), dict):
                raise ValueError("Unsupported drafter metadata")
            block = metadata["dflash_config"].get("block_size", 0)
            available = (
                metadata.get("architectures") == ["DFlash2DraftModel"]
                and isinstance(block, int)
                and block > 1
                and (folder / "model.safetensors").is_file()
            )
            drafts = block - 1 if available else None
            reason = "" if available else "The configured DFlash2 checkpoint is incomplete or unsupported."
        except (OSError, ValueError, TypeError):
            reason = "Download the configured DFlash2 checkpoint to enable this option."
    config = getattr(getattr(model, "engine", None), "config", None) if model.loaded else None
    settings = getattr(model, "pre_config", {})

    def setting(name):
        return getattr(config, name, settings.get(name, InferenceConfig.model_fields[name].default))

    dtype = str(getattr(config, "compute_dtype", settings.get("compute_dtype", "bf16")))
    if available and dtype not in ("bf16", "bfloat16", "torch.bfloat16", "auto"):
        available, reason = False, "DFlash2 requires BF16 compute."
    if available and getattr(config, "world_size", settings.get("world_size", 1)) != 1:
        available, reason = False, "DFlash2 requires single-process inference."
    if available and (setting("batch_size") != 1 or setting("beam_size") != 1):
        available, reason = False, "DFlash2 requires batch_size=1 and beam_size=1."
    if available and setting("top_k") != 1 and setting("temperature") != 0:
        available, reason = False, "DFlash2 requires greedy decoding (top_k=1 or temperature=0)."
    return dict(
        default="dflash2" if available else "mtp" if configured["mtp"] else "off",
        mtp_available=configured["mtp"],
        dflash2_available=available,
        dflash2_quantization=configured["quantization"],
        dflash2_drafts=drafts,
        dflash2_reason=reason,
    )


def selection(model, requested=None, *, eligible=True):
    options = capabilities(model)
    mode = requested or options["default"]
    if mode == "dflash2" and not options["dflash2_available"]:
        raise ValueError(options["dflash2_reason"])
    if mode == "mtp" and not options["mtp_available"]:
        raise ValueError("Native MTP is not configured for this model")
    return mode if eligible else "off"
