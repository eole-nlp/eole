"""Optional Blackwell kernel adapter for vLLM's Inkling implementation."""

import torch
from torch import nn


class BlackwellKernels:
    """Bind compatible kernels before graph capture, retaining Eole's decoder and caches."""

    def __init__(self):
        try:
            from vllm.models.inkling2.nvidia.ops import moe, moe_sm120, activation, trtllm_bmm, ll_gemm_sm120
        except ImportError as exc:
            raise ValueError("The vllm Inkling backend requires a vLLM build with models.inkling2 support") from exc
        self.routing = moe
        self.experts = moe_sm120
        self.activation = activation
        self.plan = trtllm_bmm.expert_plan
        self.dense = ll_gemm_sm120

    def linear(self, x, weight):
        if self.dense.use_sm12x_bf16_ll_gemm(x, weight):
            return self.dense.sm12x_bf16_ll_gemm(x, weight)
        return torch.nn.functional.linear(x, weight)

    def projections(self, attention, q, k, v):
        weight = attention.input_weight
        if q is k and q is v:
            output = self.linear(q, weight)
        else:
            output = self.dense.qkv_bmm(
                q.flatten(0, 1),
                k.flatten(0, 1),
                v.flatten(0, 1),
                weight,
                attention.query_rows,
                attention.query_rows + attention.projection_sizes[-2],
            ).view(*q.shape[:-1], -1)
        return output.split(attention.projection_sizes, dim=-1)


class RoutedBlackwellExperts(nn.Module):
    def __init__(self, moe, kernels):
        super().__init__()
        self.kernels = kernels
        self.register_buffer("router", moe.gate.weight.to(torch.bfloat16))
        device = self.router.device
        self.register_buffer("bounds", torch.tensor([0, self.router.size(0)], device=device, dtype=torch.int32))
        self.register_buffer("counters", torch.zeros(kernels.routing.NUM_COUNTERS, device=device, dtype=torch.int32))
        self.register_buffer("counts", torch.zeros(self.router.size(0), device=device, dtype=torch.int32))
        self.register_buffer("ones", torch.ones(self.router.size(0), device=device))
        gate, down = moe.experts.gate_up, moe.experts.down
        self.plan = kernels.plan(
            1,
            moe.num_experts_per_tok,
            self.router.size(0),
            gate.nvfp4,
            (
                (gate.weight.size(1), self.router.size(1)),
                (self.router.size(1), down.weight.size(-1) * (2 if down.nvfp4 else 1)),
            ),
        )

    @staticmethod
    def scales(projection):
        return (projection.amax, projection.scale_denominator) if projection.nvfp4 else (None, None)

    def forward(self, moe, x):
        kernels = self.kernels
        experts = moe.experts
        gate, down = experts.gate_up, experts.down
        routed, quantized = kernels.routing.route(
            x,
            self.router,
            moe.score_correction,
            self.counters,
            self.counts,
            moe.num_experts_per_tok,
            moe.route_scale,
            self.bounds,
            self.plan,
            *self.scales(gate),
        )
        source, weights, rows, tiles = routed
        hidden = kernels.experts.bmm(
            *quantized,
            gate.weight,
            gate.scales,
            self.ones,
            tiles,
            self.plan,
            0,
        )
        count = self.plan.max_ctas
        quantized = kernels.activation.swiglu_quantize(
            hidden,
            weights,
            experts.gate_softcap,
            experts.up_softcap,
            (rows, tiles[2 * count + 2 :], tiles[:count]),
            self.plan.tile,
            *self.scales(down),
        )
        output = kernels.experts.bmm(
            *quantized,
            down.weight,
            down.scales,
            self.ones,
            tiles,
            self.plan,
            1,
        )
        return kernels.routing.combine(output, source, moe.num_experts_per_tok, x.dtype)
