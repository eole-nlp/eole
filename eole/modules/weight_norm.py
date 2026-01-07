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
    """1D convolution layer with weight normalization and Polyak averaging.
    This module re-parameterizes the convolutional weight ``w`` as
    ``w = g * V / ||V||`` where ``V`` is an unconstrained weight tensor and
    ``g`` is a learned per-output-channel scale. The bias term ``b`` is kept
    separate. During the forward pass, the effective weight is reconstructed
    from ``V`` and ``g`` before applying a standard 1D convolution.
    In addition, the layer maintains Polyak-averaged copies of the parameters
    (``V_avg``, ``g_avg``, ``b_avg``) controlled by ``polyak_decay``. The
    helper :func:`get_vars_maybe_avg` returns either the current parameters
    (in training mode) or their moving averages (in evaluation mode), which
    can improve stability at test time.
    Args:
        in_channels: Number of channels in the input signal.
        out_channels: Number of channels produced by the convolution.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        padding: Zero-padding added to both sides of the input.
        dilation: Spacing between kernel elements.
        groups: Number of blocked connections from input channels to output channels.
        init_scale: Initial scaling factor applied via ``g`` (kept for API completeness).
        polyak_decay: Decay factor (close to 1) for Polyak averaging of parameters.
    """

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
        nn.init.ones_(self.g)
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
