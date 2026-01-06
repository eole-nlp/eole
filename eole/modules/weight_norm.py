"""  Weights normalization modules  """

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_var_maybe_avg(namespace, var_name, training, polyak_decay):
    """utility for retrieving polyak averaged params
    Update average
    """
    v = getattr(namespace, var_name)
    v_avg = getattr(namespace, var_name + "_avg")
    v_avg -= (1 - polyak_decay) * (v_avg - v.data)

    if training:
        return v
    else:
        return v_avg


def get_vars_maybe_avg(namespace, var_names, training, polyak_decay):
    """utility for retrieving polyak averaged params"""
    vars = []
    for vn in var_names:
        vars.append(get_var_maybe_avg(namespace, vn, training, polyak_decay))
    return vars


class WeightNormConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        init_scale=1.0,
        polyak_decay=0.9995,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=True,
        )

        self.V = self.weight
        self.g = nn.Parameter(torch.Tensor(out_channels))
        self.b = self.bias

        self.register_buffer("V_avg", torch.zeros_like(self.V))
        self.register_buffer("g_avg", torch.zeros(out_channels))
        self.register_buffer("b_avg", torch.zeros(out_channels))

        self.init_scale = init_scale
        self.polyak_decay = polyak_decay

    def forward(self, x):
        v, g, b = get_vars_maybe_avg(self, ["V", "g", "b"], self.training, self.polyak_decay)

        v_norm = torch.norm(v.view(self.out_channels, -1), dim=1)
        w = (g / v_norm).view(-1, 1, 1) * v

        return F.conv1d(
            x,
            w,
            b,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
