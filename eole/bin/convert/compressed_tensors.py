"""Read compressed-tensors INT4 and mixed NVFP4/FP8 checkpoints.

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


class CompressedCheckpoint(dict):
    """Lazy tensor view with raw checkpoint keys and source-key provenance."""


class PackedCheckpoint(CompressedCheckpoint):
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


def validate_float_config(config):
    """Validate the static NVFP4 + per-channel FP8 mixed export we execute."""
    if config.get("format") != "mixed-precision" or config.get("transform_config") or config.get("sparsity_config"):
        raise ValueError("Unsupported compressed floating-point export")
    formats = set()
    for group in config.get("config_groups", {}).values():
        w = group.get("weights") or {}
        a = group.get("input_activations") or {}
        fmt = group.get("format")
        common = w.get("type") == "float" and w.get("symmetric") and not w.get("dynamic", False)
        common = common and not w.get("block_structure") and w.get("actorder") in (None, "static")
        common = common and not a.get("block_structure")
        common = common and a.get("type") == "float" and a.get("symmetric") and not group.get("output_activations")
        if fmt == "nvfp4-pack-quantized":
            valid = (
                w.get("num_bits") == 4
                and w.get("group_size") == 16
                and w.get("strategy") == "tensor_group"
                and w.get("scale_dtype") == "torch.float8_e4m3fn"
                and a.get("num_bits") == 4
                and a.get("group_size") == 16
                and a.get("strategy") == "tensor_group"
                and a.get("dynamic") == "local"
                and a.get("scale_dtype", "torch.float8_e4m3fn") == "torch.float8_e4m3fn"
            )
        elif fmt == "float-quantized":
            valid = (
                w.get("num_bits") == 8
                and w.get("strategy") == "channel"
                and a.get("num_bits") == 8
                and a.get("strategy") == "token"
                and a.get("dynamic") is True
            )
        else:
            valid = False
        if not common or not valid:
            raise ValueError(f"Unsupported compressed floating-point group: {fmt}")
        formats.add(fmt)
    if "nvfp4-pack-quantized" not in formats:
        raise ValueError("Expected an NVFP4 configuration group")


class FloatCheckpoint(CompressedCheckpoint):
    """Alias compressed floating codes/scales without altering their bytes."""

    def __init__(self, keys, read_tensor, all_keys=None):
        self.raw_keys = set(keys)
        all_keys = set(all_keys) if all_keys is not None else self.raw_keys
        self.read_tensor = read_tensor
        self.aliases = {}
        hidden = set()
        prefixes = {
            key.rsplit(".", 1)[0]
            for key in all_keys
            if key.endswith((".weight_packed", ".weight_scale", ".weight_global_scale", ".input_global_scale"))
        }
        for prefix in prefixes:
            packed = prefix + ".weight_packed"
            weight = prefix + ".weight"
            scale = prefix + ".weight_scale"
            global_scales = [prefix + ".weight_global_scale", prefix + ".input_global_scale"]
            if packed in all_keys or any(key in all_keys for key in global_scales):
                if weight in all_keys:
                    raise ValueError(f"Packed and unpacked weight coexist for {prefix}")
                source, companions = packed, [scale, *global_scales]
            else:
                source, companions = weight, [scale]
            missing = {source, *companions} - all_keys
            if missing:
                raise ValueError("Incomplete compressed floating-point projection: " + ", ".join(sorted(missing)))
            hidden.update([source, *companions])
            if source in self.raw_keys:
                self.aliases[prefix + ".qweight"] = source
                self.aliases[prefix + ".scales"] = scale
                self.aliases.update({name: name for name in companions[1:]})
        dict.__init__(self, {key: None for key in (self.raw_keys - hidden) | self.aliases.keys()})

    def source_keys(self, key):
        return {self.aliases.get(key, key)}

    def __getitem__(self, key):
        return self.read_tensor(self.aliases.get(key, key))


def validate_modelopt_config(config):
    """Accept the scalar-global, group-16 ModelOpt NVFP4 Linear export."""
    groups = list(config.get("config_groups", {}).values())
    if (
        config.get("quant_method") != "modelopt"
        or config.get("quant_algo") != "NVFP4"
        or len(groups) != 1
        or config.get("transform_config")
        or config.get("sparsity_config")
    ):
        raise ValueError("Supported ModelOpt scheme: one NVFP4 Linear configuration group")
    group = groups[0]
    if group.get("targets") != ["Linear"] or group.get("output_activations"):
        raise ValueError("Unsupported ModelOpt NVFP4 targets or output quantization")
    for name in ("weights", "input_activations"):
        spec = group.get(name) or {}
        if (
            spec.get("type") != "float"
            or spec.get("num_bits") != 4
            or spec.get("group_size") != 16
            or spec.get("dynamic", False)
            or spec.get("block_structure")
            or spec.get("symmetric", True) is not True
            or spec.get("strategy") not in (None, "tensor_group")
        ):
            raise ValueError("ModelOpt NVFP4 requires static group-16 weights and calibrated activations")
    cache = config.get("kv_cache_scheme")
    if cache and not (cache.get("num_bits") == 8 and cache.get("type") == "float"):
        raise ValueError("Unsupported ModelOpt KV-cache scheme")
    return 16


class ModelOptCheckpoint(FloatCheckpoint):
    """Map ModelOpt NVFP4 to canonical codes/scales without requantizing.

    ModelOpt exports dequantization multipliers; the canonical floating view
    uses their reciprocals. Original source keys remain the provenance keys.
    """

    def __init__(self, keys, read_tensor, all_keys=None):
        raw_keys = set(keys)
        all_keys = set(all_keys) if all_keys is not None else raw_keys
        if any(key.endswith(".pre_quant_scale") for key in all_keys):
            raise ValueError("ModelOpt pre-quantization transforms are not supported")
        sources = {}
        prefixes = {
            key.rsplit(".", 1)[0]
            for key in all_keys
            if key.endswith((".weight_scale", ".weight_scale_2", ".input_scale"))
        }
        for prefix in prefixes:
            key = prefix + ".weight_scale"
            required = {prefix + suffix for suffix in (".weight", ".weight_scale", ".weight_scale_2", ".input_scale")}
            missing = required - all_keys
            if missing:
                raise ValueError("Incomplete ModelOpt NVFP4 projection: " + ", ".join(sorted(missing)))
            sources.update(
                {
                    prefix + ".weight_packed": prefix + ".weight",
                    prefix + ".weight_scale": key,
                    prefix + ".weight_global_scale": prefix + ".weight_scale_2",
                    prefix + ".input_global_scale": prefix + ".input_scale",
                }
            )
        normalized = {source: target for target, source in sources.items()}

        def read_normalized(key):
            value = read_tensor(sources.get(key, key))
            if value is not None and key.endswith(".weight") and value.dtype == torch.uint8:
                raise ValueError("Packed ModelOpt weight lacks NVFP4 scale metadata: " + key)
            if value is not None and key.endswith(".weight_packed"):
                if value.dtype != torch.uint8 or value.ndim != 2:
                    raise ValueError("ModelOpt NVFP4 requires output-major packed UINT8 weights")
            if value is not None and key.endswith(".weight_scale"):
                if value.dtype != torch.float8_e4m3fn or value.ndim != 2:
                    raise ValueError("ModelOpt NVFP4 requires group-16 E4M3 block scales")
            if value is not None and key.endswith((".weight_global_scale", ".input_global_scale")):
                if value.numel() != 1 or not bool(torch.isfinite(value).all() & (value > 0).all()):
                    raise ValueError("ModelOpt NVFP4 global scales must be positive finite scalars")
                value = value.float().reciprocal()
            return value

        super().__init__(
            {normalized.get(key, key) for key in raw_keys},
            read_normalized,
            {normalized.get(key, key) for key in all_keys},
        )
        self.raw_keys = raw_keys
        self._modelopt_sources = sources

    def source_keys(self, key):
        normalized = self.aliases.get(key, key)
        return {self._modelopt_sources.get(normalized, normalized)}
