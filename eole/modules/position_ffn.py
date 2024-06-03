"""Position feed-forward network from "Attention is All You Need"."""

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from eole.modules.rmsnorm import RMSNorm
from torch.nn.utils import skip_init
from torch.distributed import all_reduce
from enum import Enum


class ActivationFunction(str, Enum):
    relu = "relu"
    gelu = "gelu"
    silu = "silu"
    gated_gelu = "gated-gelu"


# for silu, see: https://arxiv.org/pdf/2002.05202.pdf
ACTIVATION_FUNCTIONS = {
    ActivationFunction.relu: F.relu,
    ActivationFunction.gelu: F.gelu,
    ActivationFunction.silu: F.silu,
    ActivationFunction.gated_gelu: F.gelu,
}


class PositionwiseFeedForward(nn.Module):
    """A two-layer Feed-Forward-Network with residual layer norm.

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
        super(PositionwiseFeedForward, self).__init__()
        assert (
            model_config.transformer_ff % self.parallel_gpu == 0
        ), "Model intermediate ffn size must be divisible by the number of partitions"
        self.w_1 = skip_init(
            nn.Linear,
            in_features=model_config.hidden_size,
            out_features=model_config.transformer_ff // self.parallel_gpu,
            bias=model_config.add_ffnbias,
        )
        self.w_2 = skip_init(
            nn.Linear,
            in_features=model_config.transformer_ff // self.parallel_gpu,
            out_features=model_config.hidden_size,
            bias=model_config.add_ffnbias,
        )
        if model_config.layer_norm == "standard" and not model_config.parallel_residual:
            self.layer_norm = nn.LayerNorm(
                model_config.hidden_size, eps=model_config.norm_eps
            )
        elif model_config.layer_norm == "rms" and not model_config.parallel_residual:
            self.layer_norm = RMSNorm(
                model_config.hidden_size, eps=model_config.norm_eps
            )
        elif not model_config.parallel_residual:
            raise ValueError(
                f"{model_config.layer_norm} layer norm type is not supported"
            )
        self.parallel_residual = model_config.parallel_residual
        self.dropout_p = getattr(running_config, "dropout", [0.0])[0]
        self.dropout_1 = nn.Dropout(self.dropout_p)
        self.dropout_2 = nn.Dropout(self.dropout_p)
        self.activation = ACTIVATION_FUNCTIONS[model_config.pos_ffn_activation_fn]
        if (
            model_config.pos_ffn_activation_fn == "silu"
            or model_config.pos_ffn_activation_fn == "gated-gelu"
        ):
            self.w_3 = skip_init(
                nn.Linear,
                in_features=model_config.hidden_size,
                out_features=model_config.transformer_ff // self.parallel_gpu,
                bias=model_config.add_ffnbias,
            )
        else:
            self.w_3 = None
        self.maybe_ckpt = (
            checkpoint
            if "ffn" in getattr(running_config, "use_ckpting", [])
            else lambda f, x: f(x)
        )

    def forward(self, x):
        """Layer definition.

        Args:
            x: ``(batch_size, input_len, model_dim)``

        Returns:
            (FloatTensor): Output ``(batch_size, input_len, model_dim)``.
        """
        if not self.parallel_residual:
            norm_x = self.layer_norm(x)
        else:
            norm_x = x.clone()
        inter = self.maybe_ckpt(self.w_1, norm_x)
        inter = self.activation(inter)
        if self.w_3 is not None:
            inter.mul_(self.maybe_ckpt(self.w_3, norm_x))
        if self.dropout_p > 0:
            inter = self.dropout_1(inter)
        inter = self.maybe_ckpt(self.w_2, inter)
        if self.dropout_p > 0:
            inter = self.dropout_2(inter)

        if self.parallel_gpu > 1:
            all_reduce(inter)

        return inter + x

    def update_dropout(self, dropout):
        self.dropout_1.p = dropout
        self.dropout_2.p = dropout
