"""
Implementation of "Convolutional Sequence to Sequence Learning"
"""

import torch
import torch.nn as nn

from eole.modules.weight_norm import WeightNormConv1d

SCALE_WEIGHT = 0.5**0.5


class GatedConv(nn.Module):
    def __init__(self, hidden_size, kernel_width=3, dropout=0.2):
        super().__init__()
        self.conv = WeightNormConv1d(
            hidden_size,
            2 * hidden_size,
            kernel_size=kernel_width,
            padding=kernel_width // 2,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, hidden, src_len)
        x = self.dropout(x)
        x = self.conv(x)
        out, gate = x.chunk(2, dim=1)
        return out * torch.sigmoid(gate)


class StackedCNN(nn.Module):
    def __init__(self, num_layers, hidden_size, kernel_width=3, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList(GatedConv(hidden_size, kernel_width, dropout) for _ in range(num_layers))

    def forward(self, x):
        # x: (batch, hidden, src_len)
        for conv in self.layers:
            x = SCALE_WEIGHT * (x + conv(x))
        return x
