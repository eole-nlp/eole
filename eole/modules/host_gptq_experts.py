"""CPU INT4 execution and disk-backed Marlin layouts for hybrid GPTQ experts.

Keep the checkpoint's codes/scales: this is layout conversion, not a second
quantization. The initial supported format is symmetric GPTQ4, group 128,
BF16 scales and direct zero point 8, without activation ordering or bias.
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


class HostGPTQExpert:
    def __init__(self, expert):
        if expert.activation is not F.silu:
            raise ValueError("Hybrid GPTQ requires gated SiLU experts")
        self.gate, self.up, self.down = (HostGPTQLinear(getattr(expert, name)) for name in PROJECTIONS)

    def __call__(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MarlinExpertStore:
    """Temporary mmap backing avoids a second anonymous-RAM weight bank.

    Packing uses at most one GPU expert at a time. Files belong to this model
    runtime, are not reusable artifacts, and are removed on close/failure.
    Budget disk space for one extra INT4 model. CPU native layouts remain in RAM.
    """

    def __init__(self, experts, device, directory=None):
        if directory is not None:
            Path(directory).mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix="eole-experts-", dir=directory)
        self.tensors = []
        self.cpu_experts = []
        try:
            for index, expert in enumerate(experts):
                self.cpu_experts.append(HostGPTQExpert(expert))
                try:
                    expert.to(device)
                    for name in PROJECTIONS:
                        getattr(expert, name).post_init()
                    values = (
                        torch.cat([expert.gate_up_proj.qweight, expert.up_proj.qweight], -1),
                        torch.cat([expert.gate_up_proj.scales, expert.up_proj.scales], -1),
                        expert.down_proj.qweight,
                        expert.down_proj.scales,
                    )
                    if not self.tensors:
                        self.hidden = expert.gate_up_proj.in_features
                        self.intermediate = expert.down_proj.in_features
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
                    for name in PROJECTIONS:
                        projection = getattr(expert, name)
                        for attr in ("qweight", "scales", "qzeros", "g_idx", "g_idx_sort_indices", "workspace"):
                            setattr(projection, attr, None)
                experts[index] = torch.nn.Identity()
        except BaseException:
            self.close()
            raise

    def close(self):
        self.tensors.clear()
        self.cpu_experts.clear()
        self.directory.cleanup()
