"""Describe selected inference implementations without importing optional packages."""

from collections import Counter
from importlib import metadata
import sys

import torch


def _version(distributions, module):
    for distribution in distributions:
        try:
            return metadata.version(distribution)
        except metadata.PackageNotFoundError:
            pass
    loaded = sys.modules.get(module)
    return getattr(loaded, "__version__", "unavailable")


def _kernel_name(fn):
    if fn is None:
        return "PyTorch conv1d"
    from eole.modules import gated_delta_net as gdn

    name = getattr(fn, "__name__", type(fn).__name__)
    module = getattr(fn, "__module__", "")
    if name.startswith("_torch_"):
        return f"PyTorch ({name})"
    if fn is gdn.fused_recurrent_gated_delta_rule:
        return "FLA fused_recurrent_gated_delta_rule (Triton JIT)"
    if fn in (gdn.causal_conv1d_fn, gdn.causal_conv1d_update):
        if module.startswith("eole."):
            return f"FLA {name} (Triton JIT)"
        return f"causal-conv1d {name} (extension)"
    return f"{module}.{name}" + (" (Triton JIT)" if module.startswith("fla.") else "")


def inference_backend_summary(model, compile_enabled, compile_mode, speculative=False, num_drafts=0):
    """Return selected paths; graph capture and SDPA dispatch are runtime decisions."""
    from eole.modules.gated_delta_net import GatedDeltaNet

    decoder = getattr(model, "decoder", None)
    if decoder is None:
        return []
    parameters = next(model.parameters(), None)
    device = str(parameters.device) if parameters is not None else "unknown"
    dtype = str(parameters.dtype) if parameters is not None else "unknown"
    compiled = compile_enabled and (
        hasattr(decoder, "_forward_compile")
        if compile_mode in ("0", "1")
        else any(hasattr(layer, "_forward_compile") for layer in getattr(decoder, "transformer_layers", []))
    )
    scope = "decoder" if compile_mode in ("0", "1") else "layers"
    verifier = "compiled" if compiled and compile_mode in ("0", "1") else "eager"
    lines = [
        f"Inference runtime: torch={torch.__version__}, CUDA build={torch.version.cuda}, "
        f"device={device}, dtype={dtype}",
        f"Inference compile: enabled={compile_enabled}, mode={compile_mode}, "
        f"{scope} path={'compiled' if compiled else 'eager'}, "
        f"CUDA graphs requested={compiled and compile_mode in ('0', '2')}; capture not verified",
        f"Inference speculation: enabled={speculative}, drafts={num_drafts if speculative else 0}, "
        f"verifier={verifier if speculative else 'unused'}",
    ]
    layers = list(getattr(decoder, "transformer_layers", []))
    flash_layers = [layer for layer in layers if hasattr(getattr(layer, "self_attn", None), "flash_attn_with_kvcache")]
    mtp_flash_layers = sum(
        hasattr(getattr(getattr(head, "layer", None), "self_attn", None), "flash_attn_with_kvcache")
        for head in getattr(model, "mtp_heads", [])
    )
    ops = sys.modules.get("eole.ops")
    flash_module = sys.modules.get("flash_attn")
    flash_fn = getattr(flash_module, "flash_attn_with_kvcache", None)
    provider = getattr(flash_fn, "__module__", "unavailable")
    flash_active = bool(flash_layers) and parameters is not None and parameters.is_cuda
    mtp_flash_selected = mtp_flash_layers if speculative and parameters is not None and parameters.is_cuda else 0
    lines.append(
        f"Inference attention: FlashAttention KV-cache selected={flash_active}, layers={len(flash_layers)}, "
        f"MTP layers selected={mtp_flash_selected}, "
        f"flash-attn={_version(('flash-attn', 'flash-attn-4'), 'flash_attn')}, provider={provider}; "
        "other attention calls use PyTorch SDPA/manual attention (SDPA kernel chosen at runtime)"
    )
    gdn_layers = [module for module in decoder.modules() if isinstance(module, GatedDeltaNet)]
    lines.append(
        f"Inference extensions: eole CUDA/C++ available={getattr(ops, '_CPP_OPS_AVAILABLE', False)}, "
        f"fla-core={_version(('fla-core', 'flash-linear-attention'), 'fla')}, "
        f"causal-conv1d={_version(('causal-conv1d',), 'causal_conv1d')}, "
        f"triton={_version(('triton',), 'triton')}"
    )
    for attr, phase in (
        ("_causal_conv1d_fn", "conv prefill"),
        ("_causal_conv1d_update", "conv decode"),
        ("_chunk_gated_delta_rule", "GDN prefill"),
        ("_recurrent_gated_delta_rule", "GDN decode/verify/replay"),
    ):
        if gdn_layers:
            selected = Counter(_kernel_name(getattr(layer, attr)) for layer in gdn_layers)
            lines.append(
                f"Inference {phase}: " + ", ".join(f"{name} [layers={count}]" for name, count in selected.items())
            )
    if gdn_layers:
        norms = Counter(
            type(getattr(layer, "norm", None)).__module__
            + "."
            + type(getattr(layer, "norm", None)).__name__
            + (
                " (FLA; inference compile adapter)"
                if getattr(getattr(layer, "norm", None), "_fla_backend", False)
                else ""
            )
            for layer in gdn_layers
        )
        lines.append(
            "Inference GDN norm/gate: " + ", ".join(f"{name} [layers={count}]" for name, count in norms.items())
        )
    if gdn_layers and speculative:
        stencil = "compiled PyTorch stencil + SiLU" if verifier == "compiled" else "PyTorch grouped conv1d + SiLU"
        lines.append(f"Inference conv verify: {stencil}")
    if speculative:
        mtp_recurrent = "compiled layers" if compiled and compile_mode in ("2", "3") else "eager"
        lines.append(
            f"Inference MTP: context refresh=eager, recurrent head={mtp_recurrent}, vocabulary projection=eager"
        )
    quantized = Counter(
        type(module).__name__
        for module in model.modules()
        if type(module).__name__.endswith("Linear") and type(module).__name__ != "Linear"
    )
    if quantized:
        lines.append(
            "Inference quantized projections: " + ", ".join(f"{name}={count}" for name, count in quantized.items())
        )
    return lines
