"""MoE mixture of experts"."""

import torch
import torch.nn as nn
from eole.modules.transformer_mlp import MLP
from torch.distributed import all_reduce


class MoE(nn.Module):
    def __init__(
        self,
        model_config,
        running_config,
    ):
        super().__init__()
        self.experts = nn.ModuleList(
            [
                MLP(
                    model_config,
                    running_config,
                    moe_transformer_ff=model_config.moe_transformer_ff,
                )
                for i in range(model_config.num_experts)
            ]
        )
        self.gate = nn.Linear(model_config.hidden_size, model_config.num_experts, bias=False)
        self.num_experts_per_tok = model_config.num_experts_per_tok
        self.parallel_gpu = running_config.parallel_gpu
        if model_config.num_shared_experts > 0:
            self.shared_experts = MLP(
                model_config,
                running_config=running_config,
                moe_transformer_ff=model_config.moe_transformer_ff * model_config.num_shared_experts,
            )
        else:
            self.shared_experts = None
        self.moe_softmax_after = model_config.moe_softmax_after

    def forward(self, x):
        z = x
        orig_shape = x.shape
        x = x.view(-1, x.shape[-1])

        scores = self.gate(x)
        if self.parallel_gpu > 1:
            all_reduce(scores)
        if not self.moe_softmax_after:
            scores = scores.softmax(dim=-1)
        expert_weights, expert_indices = torch.topk(scores, self.num_experts_per_tok, dim=-1, sorted=False)
        if self.moe_softmax_after:
            expert_weights = expert_weights.softmax(dim=-1)
        flat_expert_indices = expert_indices.view(-1)

        x = x.repeat_interleave(self.num_experts_per_tok, dim=0)
        y = torch.empty_like(x)
        for i, expert in enumerate(self.experts):
            if torch.any(flat_expert_indices == i):
                y[flat_expert_indices == i] = expert(x[flat_expert_indices == i].unsqueeze(0)).to(dtype=y.dtype)
        y = (y.view(*expert_weights.shape, -1) * expert_weights.unsqueeze(-1)).sum(dim=1)
        if self.shared_experts is not None:
            y = y + self.shared_experts(z)
        return y.view(*orig_shape)
