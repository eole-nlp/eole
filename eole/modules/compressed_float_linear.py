"""Native NVFP4/FP8 Linear weights from compressed-tensors checkpoints.

Weight codes and checkpoint scales remain in their original precision. FlashInfer
is imported only for NVFP4 inference; FP8 uses PyTorch's scaled GEMM.
"""

import torch
from torch import nn

from eole.modules.quantization_selection import is_selected, is_excluded


@torch.library.custom_op("eole::nvfp4_linear", mutates_args=())
def nvfp4_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    input_global_scale: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    from flashinfer.fp4_quantization import fp4_quantize
    from flashinfer.gemm import mm_fp4

    packed, input_scales = fp4_quantize(x, input_global_scale, nvfp4_4over6=None)
    return mm_fp4(packed, weight.t(), input_scales, scales.t(), alpha, out_dtype=x.dtype, backend="b12x")


@nvfp4_linear.register_fake
def _nvfp4_linear_fake(x, weight, scales, input_global_scale, alpha):
    return x.new_empty((x.shape[0], weight.shape[0]))


class CompressedFloatLinear(nn.Module):
    """Mixed FP8 W8A8 or NVFP4 W4A4 projection, identified by loaded codes."""

    supports_projection_fusion = True
    checkpoint_weight_layout = "compressed_float"

    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("qweight", torch.empty(0, dtype=torch.uint8))
        self.register_buffer("scales", torch.empty(0, dtype=torch.float32))
        self.register_buffer("weight_global_scale", torch.empty(0, dtype=torch.float32))
        self.register_buffer("input_global_scale", torch.empty(0, dtype=torch.float32))
        self.register_buffer("swizzled_scales", None, persistent=False)
        self.register_buffer("alpha", None, persistent=False)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.is_fp4 = False

    def _apply(self, fn, recurse=True):
        # Module.to(dtype) must not cast FP8 weight codes or block scales. Probe
        # the requested device using an empty tensor, then move original buffers.
        buffers = self._buffers
        self._buffers = {}
        try:
            super()._apply(fn, recurse)
            moved = {
                name: None if value is None else value.to(device=fn(value.new_empty(0)).device)
                for name, value in buffers.items()
            }
        except Exception:
            self._buffers = buffers
            raise
        self._buffers = moved
        return self

    def post_init(self):
        if self.qweight.device.type != "cuda" or torch.cuda.get_device_capability(self.qweight.device) != (12, 0):
            raise ValueError("NVFP4/FP8 compressed inference currently requires an SM120 GPU")
        self.is_fp4 = self.qweight.dtype == torch.uint8
        expected = (self.out_features, self.in_features // 2 if self.is_fp4 else self.in_features)
        if tuple(self.qweight.shape) != expected:
            raise ValueError(f"Invalid compressed weight shape: {tuple(self.qweight.shape)}, expected {expected}")
        if self.is_fp4:
            from flashinfer.fp4_quantization import block_scale_interleave

            if self.in_features % 16 or self.scales.dtype != torch.float8_e4m3fn:
                raise ValueError("NVFP4 requires group-16 FP8 E4M3 block scales")
            if tuple(self.scales.shape) != (self.out_features, self.in_features // 16):
                raise ValueError("Invalid NVFP4 block-scale shape")
            for scale in (self.input_global_scale, self.weight_global_scale):
                if scale.numel() != 1 or not bool(torch.isfinite(scale).all() & (scale > 0).all()):
                    raise ValueError("NVFP4 global scales must be positive finite scalars")
            padded_n = (self.out_features + 127) // 128 * 128
            padded_groups = (self.in_features // 16 + 3) // 4 * 4
            self.swizzled_scales = block_scale_interleave(self.scales.view(torch.uint8)).reshape(
                padded_n, padded_groups
            )
            self.alpha = (self.input_global_scale.float() * self.weight_global_scale.float()).reciprocal().reshape(())
        else:
            if self.qweight.dtype != torch.float8_e4m3fn or self.scales.numel() != self.out_features:
                raise ValueError("FP8 requires E4M3 weights and one scale per output channel")
            self.scales = self.scales.float().reshape(-1, 1).contiguous()

    @classmethod
    def fuse(cls, projections):
        if not all(isinstance(module, cls) for module in projections):
            raise ValueError("Mixed projection backends")
        dtypes = {module.qweight.dtype for module in projections}
        if len(dtypes) != 1:
            raise ValueError("Mixed FP8 and NVFP4 projections")
        fp4 = projections[0].qweight.dtype == torch.uint8
        if fp4:
            for name in ("weight_global_scale", "input_global_scale"):
                scale = getattr(projections[0], name).float().reshape(-1)
                if any(not torch.equal(getattr(module, name).float().reshape(-1), scale) for module in projections[1:]):
                    # A single GEMM epilogue cannot preserve distinct global scales.
                    return None
        if len({module.in_features for module in projections}) != 1:
            raise ValueError("Mismatched projection input dimensions")
        fused = cls(projections[0].in_features, sum(module.out_features for module in projections))
        fused.qweight = torch.cat([module.qweight for module in projections], dim=0)
        fused.scales = torch.cat([module.scales for module in projections], dim=0)
        if fp4:
            fused.weight_global_scale = projections[0].weight_global_scale.clone()
            fused.input_global_scale = projections[0].input_global_scale.clone()
        if any(module.bias is not None for module in projections):
            bias = next(module.bias for module in projections if module.bias is not None)
            fused.bias = nn.Parameter(
                torch.cat(
                    [
                        module.bias if module.bias is not None else bias.new_zeros(module.out_features)
                        for module in projections
                    ]
                ),
                requires_grad=False,
            )
        fused.post_init()
        return fused

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, self.in_features).contiguous()
        if self.is_fp4:
            out = nvfp4_linear(x, self.qweight, self.swizzled_scales, self.input_global_scale, self.alpha)
        else:
            scale = x.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 448.0
            quantized = (x.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
            out = torch._scaled_mm(
                quantized, self.qweight.t(), scale_a=scale, scale_b=self.scales.t(), out_dtype=x.dtype
            )
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*shape[:-1], self.out_features)


def replace_compressed_float_linear(model, modules, exclusions=(), prefix=""):
    for name, module in list(model.named_children()):
        path = prefix + "." + name if prefix else name
        if is_excluded(path, exclusions):
            continue
        if isinstance(module, nn.Linear) and is_selected(path, modules, exclusions):
            setattr(
                model, name, CompressedFloatLinear(module.in_features, module.out_features, module.bias is not None)
            )
        else:
            replace_compressed_float_linear(module, modules, exclusions, path)
