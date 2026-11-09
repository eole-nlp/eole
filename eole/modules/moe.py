"""MoE mixture of experts"."""

import torch
import torch.nn as nn
from eole.modules.transformer_mlp import MLP
from torch.distributed import all_reduce

from eole.modules.expert_execution import ResidentExpertExecutor

# Retain the historical helper imports for external callers.
from eole.modules.expert_execution import GroupedExperts as GroupedExperts
from eole.modules.expert_execution import naive_moe as naive_moe
from eole.modules.expert_execution import vectorized_moe as vectorized_moe


__all__ = ["MoE", "GroupedExperts", "naive_moe", "vectorized_moe"]


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
            if model_config.shared_expert_gate:
                self.shared_expert_gate = nn.Linear(model_config.hidden_size, 1, bias=False)
            else:
                self.shared_expert_gate = None
        else:
            self.shared_experts = None
            self.shared_expert_gate = None
        self.moe_softmax_after = model_config.moe_softmax_after
        self.moe_renormalize = model_config.moe_renormalize
        self._gates_fused = False

        self.activation_function = next(
            (a for a in ("gelu", "relu", "silu") if a in model_config.mlp_activation_fn), None
        )

        self._executor = ResidentExpertExecutor(self.experts, self.activation_function)

    def _maybe_fuse_gates(self):
        """Fuse all expert gates, executed only once."""
        if self._gates_fused:
            return

        for expert in self.experts:
            if hasattr(expert, "_fuse_gate") and getattr(expert, "up_proj", None) is not None:
                expert._fuse_gate()

        if self.shared_experts is not None and getattr(self.shared_experts, "up_proj", None) is not None:
            self.shared_experts._fuse_gate()

        self._gates_fused = True

    def _maybe_init_quant_weights(self, device, dtype):
        self._executor._maybe_init_quant_weights(device, dtype)

    def forward(self, x):

        if self.training:
            self._maybe_fuse_gates()
        else:
            self._maybe_init_quant_weights(x.device, x.dtype)

        B, T, C = x.shape

        # flatten only token dimension, but keep track
        x_flat = x.view(-1, C)

        expert_weights, expert_indices = self.route(x_flat)
        y = self._executor.execute(x_flat, expert_weights, expert_indices, training=self.training).view(B, T, C)
        return self.combine(x, y)

    def route(self, x_flat):
        """GPU routing shared by resident and hybrid execution."""
        K = self.num_experts_per_tok
        # ── Gate ──────────────────────────────────────────────────────────────
        scores = self.gate(x_flat)  # (BT, num_experts)

        if self.parallel_gpu > 1:
            all_reduce(scores)

        if not self.moe_softmax_after:
            scores = scores.softmax(dim=-1)

        expert_weights, expert_indices = torch.topk(scores, K, dim=-1, sorted=False)  # both are (BT, K)

        if self.moe_softmax_after:
            # Mixtral-style: softmax is applied *after* topk selection.
            expert_weights = expert_weights.softmax(dim=-1)
        elif self.moe_renormalize:
            # Qwen-style (default): softmax is applied over all experts before topk.
            # After topk selection the remaining weights no longer sum to 1, so we
            # re-normalize to restore a proper probability distribution.
            # clamp prevents a theoretical zero-sum when all selected logits are tiny.
            expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        return expert_weights, expert_indices

    def combine(self, x, y):
        """Combine routed results with the optional shared expert."""
        B, T, _ = x.shape
        # ── Optional shared experts ───────────────────────────────────────────
        if self.shared_experts is not None:
            shared_out = self.shared_experts(x)
            if self.shared_expert_gate is not None:
                x_flat_for_gate = x.view(-1, x.shape[-1])
                gate_val = torch.sigmoid(self.shared_expert_gate(x_flat_for_gate)).view(B, T, 1)
                y = y + gate_val * shared_out
            else:
                y = y + shared_out

        return y
