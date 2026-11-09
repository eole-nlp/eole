"""CPU expert layouts and temporary Marlin backing for adaptive execution.

GPTQ4 codes/scales are preserved. Floating-point sources are quantized once to
per-output-channel INT8, with identical codes and BF16 scales on both devices.
"""

import os
from pathlib import Path
import tempfile

import torch
import torch.nn.functional as F


PROJECTIONS = ("gate_up_proj", "up_proj", "down_proj")


class HostGPTQLinear:
    def __init__(self, linear):
        if (
            getattr(linear, "checkpoint_weight_layout", None) != "gptq"
            or getattr(linear, "bits", None) != 4
            or linear.group_size != 128
            or linear.desc_act
            or linear.bias is not None
            or linear.qweight.device.type != "cpu"
            or linear.scales.dtype != torch.bfloat16
        ):
            raise ValueError("Hybrid GPTQ requires CPU GPTQ4, group128, BF16 scales, no desc_act or bias")
        if (
            linear.qweight.dtype != torch.int32
            or linear.qweight.shape != (linear.in_features // 8, linear.out_features)
            or linear.scales.shape != (linear.in_features // 128, linear.out_features)
            or linear.scales.device.type != "cpu"
            or linear.qzeros.device.type != "cpu"
        ):
            raise ValueError("Hybrid GPTQ requires unrepacked CPU GPTQ weights and group128 scales")
        if not bool((linear.qzeros == -2004318072).all()):
            raise ValueError("Hybrid GPTQ requires direct symmetric zero point 8")
        shifts = torch.arange(8, dtype=torch.int64) * 4
        codes = (linear.qweight.detach().long().unsqueeze(1) >> shifts[None, :, None]) & 15
        codes = codes.reshape(linear.in_features, linear.out_features).t().int().contiguous()
        self.weight = torch.ops.aten._convert_weight_to_int4pack_for_cpu(codes, 1)
        scales = linear.scales.detach()
        self.scales_zeros = torch.stack([scales, torch.zeros_like(scales)], -1).contiguous()

    def __call__(self, x):
        return torch.ops.aten._weight_int4pack_mm_for_cpu(
            x.to(torch.bfloat16).contiguous(), self.weight, 128, self.scales_zeros
        )


class HostInt8Linear:
    """One RTN quantization shared by CPU INT8 and GPU Marlin execution."""

    def __init__(self, linear):
        if (
            type(linear) is not torch.nn.Linear
            or linear.weight.device.type != "cpu"
            or not linear.weight.is_floating_point()
            or linear.bias is not None
            or linear.in_features % 128
            or linear.out_features % 64
        ):
            raise ValueError("Adaptive INT8 requires aligned, bias-free floating-point CPU Linear weights")
        weight = linear.weight.detach().float()
        self.scales = (weight.abs().amax(1) / 127).clamp_min(torch.finfo(torch.bfloat16).tiny).bfloat16()
        # Round scales before quantizing so CPU and GPU represent the same weights.
        self.weight = (weight / self.scales.float()[:, None]).round().clamp(-127, 127).to(torch.int8).contiguous()

    def __call__(self, x):
        return torch.ops.aten._weight_int8pack_mm(x.to(torch.bfloat16).contiguous(), self.weight, self.scales)

    def marlin_projection(self, device):
        from eole.modules.marlin_linear import MarlinQuantLinear

        n, k = self.weight.shape
        unsigned = self.weight.t().long() + 128
        shifts = torch.arange(4) * 8
        packed = (unsigned.reshape(k // 4, 4, n) << shifts[None, :, None]).sum(1).to(torch.int32)
        projection = MarlinQuantLinear(8, 128, False, True, k, n).bfloat16()
        with torch.no_grad():
            projection.qweight.copy_(packed)
            # The MoE kernel supports group128. Repeating a per-channel scale
            # preserves exactly the CPU weights without another quantization.
            projection.scales.copy_(self.scales[None, :].expand(k // 128, -1))
            projection.qzeros.zero_()
            projection.g_idx.copy_(torch.arange(k, dtype=torch.int32) // 128)
        return projection.to(device)


class HostExpert:
    def __init__(self, expert, projection_type):
        if expert.activation is not F.silu:
            raise ValueError("Adaptive experts require gated SiLU")
        self.gate, self.up, self.down = (projection_type(getattr(expert, name)) for name in PROJECTIONS)

    def __call__(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MarlinExpertStore:
    """Temporary mmap backing avoids a second anonymous-RAM weight bank.

    Packing uses at most one GPU expert at a time. Files belong to this model
    runtime, are not reusable artifacts, and are removed on close/failure.
    Budget disk space for the expert weights at their execution precision.
    CPU native layouts remain in RAM.
    """

    def __init__(self, experts, device, directory=None, source_format="auto"):
        from eole.modules.moe_quant_utils import MARLIN_UINT4B8_TYPE_ID, MARLIN_UINT8B128_TYPE_ID

        if source_format not in ("auto", "gptq"):
            raise ValueError("Unknown adaptive expert source format")
        floating = type(experts[0].gate_up_proj) is torch.nn.Linear
        if source_format == "gptq" and floating:
            raise ValueError("hybrid_gptq requires GPTQ INT4 experts; use adaptive for floating-point sources")
        self.bits = 8 if floating else 4
        self.scalar_type_id = MARLIN_UINT8B128_TYPE_ID if floating else MARLIN_UINT4B8_TYPE_ID
        projection_type = HostInt8Linear if floating else HostGPTQLinear
        if directory is not None:
            Path(directory).mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix="eole-experts-", dir=directory)
        self.tensors = []
        self.cpu_experts = []
        try:
            for index, expert in enumerate(experts):
                host = HostExpert(expert, projection_type)
                self.cpu_experts.append(host)
                projections = []
                try:
                    for name, cpu in zip(PROJECTIONS, (host.gate, host.up, host.down)):
                        projection = cpu.marlin_projection(device) if floating else getattr(expert, name).to(device)
                        projections.append(projection)
                        projection.post_init()
                    gate, up, down = projections
                    values = (
                        torch.cat([gate.qweight, up.qweight], -1),
                        torch.cat([gate.scales, up.scales], -1),
                        down.qweight,
                        down.scales,
                    )
                    if not self.tensors:
                        self.hidden = gate.in_features
                        self.intermediate = down.in_features
                        for n, value in enumerate(values):
                            shape = (len(experts), *value.shape)
                            size = value.numel() * len(experts)
                            path = Path(self.directory.name) / str(n)
                            # Preallocate the file: fail here on insufficient disk,
                            # rather than risking a SIGBUS on a later mmap store.
                            with path.open("wb") as output:
                                os.posix_fallocate(output.fileno(), 0, size * value.element_size())
                            tensor = torch.from_file(str(path), shared=True, size=size, dtype=value.dtype).view(shape)
                            self.tensors.append(tensor)
                    for target, value in zip(self.tensors, values):
                        target[index].copy_(value.cpu())
                    del values
                finally:
                    # MLP.forward holds a bound-method cycle; explicit release
                    # prevents obsolete CUDA weights surviving until Python GC.
                    for projection in projections:
                        for attr in ("qweight", "scales", "qzeros", "g_idx", "g_idx_sort_indices", "workspace"):
                            setattr(projection, attr, None)
                    if floating:
                        # Break the MLP forward-method cycle's ownership of FP weights.
                        for name in PROJECTIONS:
                            getattr(expert, name).weight = None
                experts[index] = torch.nn.Identity()
        except BaseException:
            self.close()
            raise

    def close(self):
        self.tensors.clear()
        self.cpu_experts.clear()
        self.directory.cleanup()
