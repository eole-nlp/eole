"""Convolutional encoder implementation."""

from typing import Optional, Tuple
import torch
import torch.nn as nn

from eole.encoders.encoder import EncoderBase
from eole.utils.cnn_factory import StackedCNN


class CNNEncoder(EncoderBase):
    """
    Convolutional sequence-to-sequence encoder.

    Based on "Convolutional Sequence to Sequence Learning" (Gehring et al., 2017)
    Reference: https://arxiv.org/abs/1705.03122

    Args:
        encoder_config: Encoder configuration
        running_config: Runtime configuration (optional)
    """

    def __init__(self, encoder_config, running_config=None):
        super().__init__()

        # Input projection to hidden dimension
        input_size = encoder_config.hidden_size
        self.linear = nn.Linear(input_size, encoder_config.hidden_size)

        # Stacked CNN layers
        dropout = getattr(running_config, "dropout", [0.0])[0] if running_config else 0.0
        self.cnn = StackedCNN(
            encoder_config.layers,
            encoder_config.hidden_size,
            kernel_width=encoder_config.cnn_kernel_width,
            dropout=dropout,
        )

    def forward(
        self, emb: torch.Tensor, pad_mask: Optional[torch.Tensor] = None, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input embeddings through CNN layers.

        Args:
            emb: Input embeddings (batch, src_len, dim)
            pad_mask: Padding mask (optional, not used)
            **kwargs: Additional arguments

        Returns:
            Tuple of:
                - CNN output (batch, src_len, hidden_size)
                - Projected embeddings (batch, src_len, hidden_size)
        """
        projected = self.linear(emb)

        # Conv1d expects (batch, hidden, src_len)
        x = projected.transpose(1, 2)

        x = self.cnn(x)

        # Back to encoder format
        x = x.transpose(1, 2)

        return x, projected

    def update_dropout(self, dropout: float, attention_dropout: Optional[float] = None) -> None:
        """Update CNN dropout rate."""
        # StackedCNN does not expose a single dropout module; its layers (e.g. GatedConv)
        # each own their own dropout. Update all of them.
        if hasattr(self.cnn, "layers"):
            for layer in self.cnn.layers:
                if hasattr(layer, "dropout") and hasattr(layer.dropout, "p"):
                    layer.dropout.p = dropout
