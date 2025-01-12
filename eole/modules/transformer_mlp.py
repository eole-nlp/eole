"""MLP network from "Attention is All You Need"."""

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
        self.parallel_gpu = running_config.parallel_gpu
        super(MLP, self).__init__()
        assert (
            model_config.transformer_ff % self.parallel_gpu == 0
        ), "Model intermediate ffn size must be divisible by the number of partitions"
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
        if model_config.mlp_activation_fn == "gated-silu" or model_config.mlp_activation_fn == "gated-gelu":
            self.up_proj = skip_init(
                nn.Linear,
                in_features=model_config.hidden_size,
                out_features=model_config.transformer_ff // self.parallel_gpu,
                bias=model_config.add_ffnbias,
            )
        else:
            self.up_proj = None
        self.maybe_ckpt = checkpoint if "ffn" in getattr(running_config, "use_ckpting", []) else lambda f, x: f(x)

    def forward(self, x):
        """Layer definition.

        Args:
            x: ``(batch_size, input_len, model_dim)``

        Returns:
            (FloatTensor): Output ``(batch_size, input_len, model_dim)``.
        """
        mlp_out = self.maybe_ckpt(self.gate_up_proj, x)
        mlp_out = self.activation(mlp_out)
        if self.up_proj is not None:
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

    def update_dropout(self, dropout):
        self.dropout_1.p = dropout
        self.dropout_2.p = dropout
