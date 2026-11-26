"""Resident experts with block-scaled floating-point checkpoint weights."""

import torch
from torch import nn
from torch.nn import functional as F


class BlockFloatExpertWeight(nn.Module):
    """Keep released expert weights packed, dequantizing only selected experts to BF16."""

    preserve_checkpoint_dtype = True

    def __init__(self, shape, weight_format):
        super().__init__()
        experts, rows, cols = shape
        self.weight_format = weight_format
        self.nvfp4 = weight_format == "nvfp4"
        self.block_size = 16 if self.nvfp4 else 32
        if cols % self.block_size:
            raise ValueError(f"Expert input width {cols} must be divisible by {self.block_size}")
        packed_cols = cols // 2 if self.nvfp4 else cols
        weight_dtype = torch.uint8 if self.nvfp4 else torch.float8_e4m3fn
        scale_dtype = torch.float8_e4m3fn if self.nvfp4 else torch.uint8
        self.register_buffer("weight", torch.empty((experts, rows, packed_cols), dtype=weight_dtype))
        self.register_buffer("scales", torch.empty((experts, rows, cols // self.block_size), dtype=scale_dtype))
        if self.nvfp4:
            self.register_buffer("amax", torch.empty(experts, dtype=torch.float32))
            self.register_buffer("scale_denominator", torch.empty(1, dtype=torch.float32))

    def prepare(self):
        expected = (torch.uint8, torch.float8_e4m3fn) if self.nvfp4 else (torch.float8_e4m3fn, torch.uint8)
        if (self.weight.dtype, self.scales.dtype) != expected:
            raise ValueError(f"Invalid packed weights or block scales for {self.weight_format} experts")
        if self.nvfp4:
            self.outer_scale = self.amax / self.scale_denominator
            self.fp4_values = self.amax.new_tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])

    def forward(self, indices, dtype):
        packed = self.weight.view(torch.uint8)[indices]
        scales = self.scales.view(torch.uint8)[indices]
        if self.nvfp4:
            codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
            magnitude = self.fp4_values.index_select(0, (codes & 7).flatten().int()).view_as(codes)
            values = torch.where(codes < 8, magnitude, -magnitude)
            block_scales = scales.view(torch.float8_e4m3fn).float()
        else:
            values = packed.view(torch.float8_e4m3fn).float()
            block_scales = torch.ldexp(torch.ones_like(scales, dtype=torch.float32), scales.int() - 127)

        dequantized = (values.unflatten(-1, (-1, self.block_size)) * block_scales.unsqueeze(-1)).flatten(-2)
        if self.nvfp4:
            dequantized = dequantized * self.outer_scale[indices][..., None, None]
        return dequantized.to(dtype)


class BlockFloatExperts(nn.Module):
    def __init__(self, cfg, weight_format):
        super().__init__()
        experts, hidden, intermediate = cfg.num_experts, cfg.hidden_size, cfg.moe_transformer_ff
        self._grouped_prefill = False
        self.gate_softcap = cfg.layer_loop.gate_softcap
        self.up_softcap = cfg.layer_loop.up_softcap
        self.gate_up = BlockFloatExpertWeight((experts, 2 * intermediate, hidden), weight_format)
        self.down = BlockFloatExpertWeight((experts, hidden, intermediate), weight_format)

    def prepare(self):
        self.gate_up.prepare()
        self.down.prepare()

        from eole.ops import _CPP_OPS_AVAILABLE

        device = self.gate_up.weight.device
        self._grouped_prefill = (
            _CPP_OPS_AVAILABLE
            and device.type == "cuda"
            and torch.version.cuda is not None
            and torch.cuda.get_device_capability(device) >= (8, 9)
            and self.gate_up.weight.size(0) <= 1024
        )

    def _activate(self, x):
        # The original checkpoint interleaves the gate and up rows.
        gate, up = x.float().unflatten(-1, (-1, 2)).unbind(-1)
        gate_factor = self.gate_softcap * (gate / self.gate_softcap).tanh()
        up_factor = self.up_softcap * (up / self.up_softcap).tanh()
        return (gate_factor * gate.sigmoid() * up_factor).to(x.dtype)

    def forward(self, flat, weights, indices, *, fast_decode=False):
        dtype = flat.dtype
        if len(flat) == 1:
            selected = indices[0]
            if fast_decode:
                from eole.triton.block_float_moe import expert_gemv

                gate_up = expert_gemv(self.gate_up, flat, selected, shared_input=True)
                from eole.triton.looped_decoder import situ, sum_routes

                activated = situ(gate_up, self.gate_softcap, self.up_softcap)
                routes = expert_gemv(self.down, activated, selected, shared_input=False)
                return sum_routes(routes, weights[0], flat.shape)
            else:
                gate_up = (self.gate_up(selected, dtype) @ flat.T).squeeze(-1)
                routes = (self.down(selected, dtype) @ self._activate(gate_up).unsqueeze(-1)).squeeze(-1)
            return (routes.float() * weights[0, :, None]).sum(dim=0).to(dtype).view_as(flat)

        if self._grouped_prefill and dtype == torch.bfloat16:
            from eole.triton.block_float_moe import grouped_experts

            return grouped_experts(self, flat, weights, indices)

        result = torch.zeros_like(flat, dtype=torch.float32)
        # Group prompt tokens by expert to reuse weights and bound dequantization scratch space.
        for expert in indices.unique().tolist():
            token, route = torch.where(indices == expert)
            gate_up = F.linear(flat[token], self.gate_up(expert, dtype))
            routes = F.linear(self._activate(gate_up), self.down(expert, dtype))
            result.index_add_(0, token, routes.float() * weights[token, route, None])
        return result.to(dtype).view_as(flat)
