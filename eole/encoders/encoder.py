"""Base class for encoders."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any
import torch
import torch.nn as nn


class EncoderBase(nn.Module, ABC):
    """
    Abstract base encoder class defining the interface for all encoders.

    Used by:
        - eole.Models.EncoderDecoderModel
        - eole.Models.EncoderModel
    """

    @abstractmethod
    def forward(
        self, emb: torch.Tensor, pad_mask: Optional[torch.Tensor] = None, **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode input embeddings.

        Args:
            emb: Input embeddings of shape (batch, src_len, dim)
            pad_mask: Padding mask of shape (batch, src_len).
                     False for actual values, True for padding.
            **kwargs: Additional encoder-specific arguments

        Returns:
            Tuple containing:
                - enc_out: Encoder output for attention (batch, src_len, hidden_size)
                - enc_final_hs: Final hidden state or None
                  For RNN: (num_layers * directions, batch, hidden_size)
                  For LSTM: tuple of (hidden, cell)
                  For Transformer/CNN: None
        """
        raise NotImplementedError

    def update_dropout(self, dropout: float, attention_dropout: Optional[float] = None) -> None:
        """
        Update dropout rates dynamically.

        Args:
            dropout: General dropout rate
            attention_dropout: Attention-specific dropout rate (if applicable)
        """
        # Default implementation - subclasses override as needed
        pass
