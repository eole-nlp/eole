"""MLP network from "Attention is All You Need"."""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.nn.utils import skip_init
from torch.distributed import all_reduce
from eole.constants import ACTIVATION_FUNCTIONS


class MLP(nn.Module):
    """A two/three-layer Feed-Forward-Network.

    Args:
        model_config: eole.config.models.ModelConfig object
        running_config: TrainingConfig or InferenceConfig derived from RunningConfig
    """

    def __init__(
        self,
        model_config,
        running_config=None,
    ):
        self.parallel_gpu = getattr(running_config, "parallel_gpu", 1)
        super(MLP, self).__init__()
        assert (
            model_config.transformer_ff % self.parallel_gpu == 0
        ), "Model intermediate ffn size must be divisible by the number of partitions"
        self.model_config = model_config
        self.gate_up_proj = skip_init(
            nn.Linear,
            in_features=model_config.hidden_size,
            out_features=model_config.transformer_ff // self.parallel_gpu,
            bias=model_config.add_ffnbias,
        )
        self.down_proj = skip_init(
            nn.Linear,
            in_features=model_config.transformer_ff // self.parallel_gpu,
            out_features=model_config.hidden_size,
            bias=model_config.add_ffnbias,
        )

        self.dropout_p = getattr(running_config, "dropout", [0.0])[0]
        self.dropout_1 = nn.Dropout(self.dropout_p)
        self.dropout_2 = nn.Dropout(self.dropout_p)
        self.activation = ACTIVATION_FUNCTIONS[model_config.mlp_activation_fn]
        if model_config.mlp_activation_fn in ["gated-silu", "gated-gelu", "gated-gelu-tanh"]:
            self.up_proj = skip_init(
                nn.Linear,
                in_features=model_config.hidden_size,
                out_features=model_config.transformer_ff // self.parallel_gpu,
                bias=model_config.add_ffnbias,
            )
            self.forward = self.gated_forward
        else:
            self.up_proj = None
            self.forward = self.simple_forward

        self.maybe_ckpt = checkpoint if "ffn" in getattr(running_config, "use_ckpting", []) else lambda f, x: f(x)

    def _fuse_gate(self) -> None:
        if hasattr(self.gate_up_proj, "weight"):
            new_gate = skip_init(
                nn.Linear,
                in_features=self.model_config.hidden_size,
                out_features=self.model_config.transformer_ff // self.parallel_gpu * 2,
                bias=self.model_config.add_ffnbias,
            )
            new_gate.weight = nn.Parameter(torch.cat([self.gate_up_proj.weight, self.up_proj.weight]))
            if self.model_config.add_ffnbias:
                new_gate.bias = nn.Parameter(torch.cat([self.gate_up_proj.bias, self.up_proj.bias]))

        elif hasattr(self.gate_up_proj, "qweight") and hasattr(self.up_proj, "qweight"):
            w_bit = self.gate_up_proj.w_bit
            group_size = self.gate_up_proj.group_size

            awq_cls = self.gate_up_proj.__class__
            dim = 1 if awq_cls.__name__ == "WQLinear_GEMM" else 0
            new_gate = awq_cls(
                w_bit=w_bit,
                group_size=group_size,
                in_features=self.model_config.hidden_size,
                out_features=self.model_config.transformer_ff // self.parallel_gpu * 2,
                bias=self.model_config.add_ffnbias,
                dev=self.gate_up_proj.qweight.device,
            )
            new_gate.qweight = torch.cat([self.gate_up_proj.qweight, self.up_proj.qweight], dim)
            new_gate.qzeros = torch.cat([self.gate_up_proj.qzeros, self.up_proj.qzeros], dim)
            new_gate.scales = torch.cat([self.gate_up_proj.scales, self.up_proj.scales], dim)
        del self.up_proj
        self.gate_up_proj = new_gate
        self.forward = self.simple_forward
        self.activation = ACTIVATION_FUNCTIONS["fused-" + self.model_config.mlp_activation_fn]

    def gated_forward(self, x):
        """Layer definition. Legacy Gated operations. No fusion.

        Args:
            x: ``(batch_size, input_len, model_dim)``

        Returns:
            (FloatTensor): Output ``(batch_size, input_len, model_dim)``.
        """
        mlp_out = self.maybe_ckpt(self.gate_up_proj, x)
        mlp_out = self.activation(mlp_out)
        mlp_out.mul_(self.maybe_ckpt(self.up_proj, x))
        if self.dropout_p > 0:
            mlp_out = self.dropout_1(mlp_out)
        mlp_out = self.maybe_ckpt(self.down_proj, mlp_out)
        if self.dropout_p > 0:
            mlp_out = self.dropout_2(mlp_out)

        if self.parallel_gpu > 1:
            # all_reduce is an inplace op - not easily backprop
            mlp_out1 = mlp_out.detach().clone()
            all_reduce(mlp_out1)
            mlp_out.copy_(mlp_out1 + (mlp_out - mlp_out.detach()))

        return mlp_out

    def simple_forward(self, x):
        """Layer definition.

        Args:
            x: ``(batch_size, input_len, model_dim)``

        Returns:
            (FloatTensor): Output ``(batch_size, input_len, model_dim)``.
        """
        mlp_out = self.maybe_ckpt(self.gate_up_proj, x)
        mlp_out = self.activation(mlp_out)
        if self.dropout_p > 0:
            mlp_out = self.dropout_1(mlp_out)
        mlp_out = self.maybe_ckpt(self.down_proj, mlp_out)

        if self.dropout_p > 0:
            mlp_out = self.dropout_2(mlp_out)

        if self.parallel_gpu > 1:
            # all_reduce is an inplace op - not easily backprop
            mlp_out1 = mlp_out.detach().clone()
            all_reduce(mlp_out1)
            mlp_out.copy_(mlp_out1 + (mlp_out - mlp_out.detach()))

        return mlp_out

    def update_dropout(self, dropout):
        self.dropout_1.p = dropout
        self.dropout_2.p = dropout
