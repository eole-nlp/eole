"""
Implementation of "Convolutional Sequence to Sequence Learning"
"""

import torch
import torch.nn as nn

from eole.modules.weight_norm import WeightNormConv1d

SCALE_WEIGHT = 0.5**0.5


class GatedConv(nn.Module):
    """
    One-dimensional gated convolutional block using a Gated Linear Unit (GLU).
    This module applies a weight-normalized 1D convolution that doubles the number
    of channels and splits the result into two halves: a content tensor and a
    gate tensor. The output is the element-wise product of the content and the
    sigmoid-activated gate, following the GLU formulation used in
    convolutional sequence-to-sequence architectures.
    Args:
        hidden_size: Number of input and output channels for the block
        kernel_width: Width of the temporal convolution kernel
        dropout: Dropout probability applied to the input before convolution
        nopad: If True, no padding is added (padding handled externally for causal masking).
               If False, symmetric padding is added to preserve sequence length.
    """

    def __init__(self, hidden_size, kernel_width=3, dropout=0.2, nopad: bool = False):
        super().__init__()
        # When nopad=True (decoder), padding=0 because decoder handles causal padding manually
        # When nopad=False (encoder), padding=kernel_width//2 for symmetric padding
        padding = 0 if nopad else kernel_width // 2

        self.conv = WeightNormConv1d(
            hidden_size,
            2 * hidden_size,
            kernel_size=kernel_width,
            padding=padding,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, hidden_size, seq_len)

        Returns:
            Gated output tensor (batch, hidden_size, seq_len)
        """
        x = self.dropout(x)
        x = self.conv(x)
        # Split into content and gate
        out, gate = x.chunk(2, dim=1)
        # Apply gating: content * sigmoid(gate)
        return out * torch.sigmoid(gate)


class StackedCNN(nn.Module):
    """
    Stack of gated 1D convolutional layers with residual connections.

    This module applies `num_layers` instances of GatedConv in sequence
    to an input tensor of shape (batch, hidden, src_len), using a residual
    connection and scaling factor at each layer as in "Convolutional Sequence
    to Sequence Learning" (Gehring et al., 2017).

    Args:
        num_layers: Number of convolutional layers to stack
        hidden_size: Number of channels in the hidden representation
        kernel_width: Width of the 1D convolution kernel
        dropout: Dropout probability applied before each convolution
        nopad: If True, no padding in conv layers (used for decoder with manual padding).
               If False, symmetric padding is used (encoder).
    """

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        kernel_width: int = 3,
        dropout: float = 0.2,
        nopad: bool = False,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [GatedConv(hidden_size, kernel_width, dropout, nopad=nopad) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, hidden_size, seq_len)

        Returns:
            Output tensor (batch, hidden_size, seq_len)
        """
        for conv in self.layers:
            # Residual connection with variance scaling
            x = SCALE_WEIGHT * (x + conv(x))
        return x
