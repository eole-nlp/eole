"""Read symmetric grouped compressed-tensors INT4 as Eole's GPTQ layout.

Repacking changes only tensor layout, not quantized values or scales. No
compressed-tensors runtime dependency is needed for conversion.
"""

import torch


def validate_config(config):
    groups = list(config.get("config_groups", {}).values())
    if config.get("format") != "pack-quantized" or len(groups) != 1:
        raise ValueError("compressed-tensors requires one pack-quantized configuration group")
    group = groups[0]
    weights = group.get("weights") or {}
    if (
        group.get("format", "pack-quantized") != "pack-quantized"
        or group.get("targets") != ["Linear"]
        or group.get("input_activations") is not None
        or group.get("output_activations") is not None
        or weights.get("num_bits") != 4
        or weights.get("type") != "int"
        or weights.get("symmetric") is not True
        or weights.get("strategy") != "group"
        or weights.get("group_size") not in (32, 64, 128)
        or weights.get("block_structure") is not None
        or weights.get("dynamic", False)
        or weights.get("actorder") not in (None, "static")
        or config.get("transform_config")
        or config.get("sparsity_config")
    ):
        raise ValueError("Supported compressed-tensors scheme: symmetric static grouped W4A16 Linear weights")
    cache = config.get("kv_cache_scheme")
    if cache and not (cache.get("num_bits") == 8 and cache.get("type") == "float"):
        raise ValueError("Unsupported compressed-tensors KV-cache scheme")
    return weights["group_size"]


def repack_int4(packed, scales, shape, group_size):
    """Convert [N,K/8] packed weights to GPTQ [K/8,N], keeping offset-8 codes."""
    if shape.ndim != 1 or shape.numel() != 2 or shape.dtype not in (torch.int32, torch.int64):
        raise ValueError("weight_shape must contain two integer dimensions")
    if group_size not in (32, 64, 128):
        raise ValueError("Unsupported INT4 group size")
    n, k = shape.tolist()
    if n <= 0 or k <= 0 or k % group_size or n % 8:
        raise ValueError("INT4 dimensions must have complete groups and output channels divisible by 8")
    if packed.dtype != torch.int32 or tuple(packed.shape) != (n, k // 8):
        raise ValueError("Invalid compressed-tensors weight_packed shape or dtype")
    if not scales.is_floating_point() or tuple(scales.shape) != (n, k // group_size):
        raise ValueError("Invalid compressed-tensors weight_scale shape or dtype")
    # Source words already pack consecutive input coordinates. Only transpose
    # the word matrix; the nibble order and offset-8 representation agree.
    qweight = packed.t().contiguous()
    # GPTQ stores zero_point - 1, so symmetric INT4's offset 8 is encoded as 7.
    zeros = torch.full((k // group_size, n // 8), 0x77777777, dtype=torch.int32, device=packed.device)
    return {
        "qweight": qweight,
        "scales": scales.t().contiguous(),
        "qzeros": zeros,
        "g_idx": torch.arange(k, dtype=torch.int32, device=packed.device) // group_size,
    }


class PackedCheckpoint(dict):
    """Lazy checkpoint view exposing GPTQ names without loading a whole shard."""

    def __init__(self, keys, read_tensor, group_size):
        self.raw_keys = set(keys)
        self.read_tensor = read_tensor
        self.group_size = group_size
        self._prefix = None
        self._converted = None
        virtual = set()
        for key in self.raw_keys:
            if key.endswith(".weight_packed"):
                prefix = key.removesuffix("weight_packed")
                virtual.update(prefix + suffix for suffix in ("qweight", "qzeros", "scales", "g_idx"))
        super().__init__()
        for key in self.raw_keys | virtual:
            dict.__setitem__(self, key, None)

    def source_keys(self, key):
        prefix, _, suffix = key.rpartition(".")
        if suffix in ("qweight", "qzeros", "scales", "g_idx") and prefix + ".weight_packed" in self.raw_keys:
            return {prefix + "." + s for s in ("weight_packed", "weight_scale", "weight_shape")}
        return {key}

    def __getitem__(self, key):
        prefix, _, suffix = key.rpartition(".")
        if suffix in ("qweight", "qzeros", "scales", "g_idx") and prefix + ".weight_packed" in self.raw_keys:
            if self._prefix != prefix:
                if self.read_tensor(prefix + ".weight_g_idx") is not None:
                    raise ValueError("Activation-order weight_g_idx is not supported")
                tensors = [
                    self.read_tensor(prefix + "." + s) for s in ("weight_packed", "weight_scale", "weight_shape")
                ]
                if any(t is None for t in tensors):
                    raise ValueError(f"Incomplete packed weight: {prefix}")
                self._converted = repack_int4(*tensors, self.group_size)
                self._prefix = prefix
            return self._converted[suffix]
        return self.read_tensor(key)
