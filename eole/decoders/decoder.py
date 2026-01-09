"""Decoder base class."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, Callable
import torch
import torch.nn as nn


class DecoderBase(nn.Module, ABC):
    """Abstract base class for decoders.

    Args:
        attentional: Whether the decoder returns non-empty attention weights.
    """

    def __init__(self, attentional: bool = True) -> None:
        super().__init__()
        self.attentional = attentional
        self.state: Dict[str, Any] = {}

    @abstractmethod
    def init_state(self, **kwargs) -> None:
        """Initialize decoder state from encoder outputs.

        Args:
            **kwargs: Initialization data. Common arguments include:
                - enc_out: Encoder output tensor
                - enc_final_hs: Encoder final hidden states
                - src: Source input tensor
        """
        raise NotImplementedError

    @abstractmethod
    def map_state(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """Apply a function to all tensors in decoder state.

        Used for operations like moving to device or selecting batch indices.

        Args:
            fn: Function to apply to each tensor in the state.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(
        self, emb: torch.Tensor, enc_out: Optional[torch.Tensor] = None, step: Optional[int] = None, **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass

        Args:
            emb (Tensor):
                Target embeddings of shape
                ``(batch_size, tgt_len, hidden_size)``.

            enc_out (Tensor, optional):
                Encoder outputs of shape
                ``(batch_size, src_len, hidden_size)``.
                Required for encoder–decoder models, unused for language models.

            step (int, optional):
                Decoding step for incremental (autoregressive) decoding.
                ``None`` indicates full-sequence decoding.

            **kwargs:
                Decoder-specific arguments (e.g. masks, alignment flags).

        Returns:
            (Tensor, Dict[str, Tensor]):

            * dec_outs:
                Decoder outputs of shape
                ``(batch_size, tgt_len, hidden_size)``.

            * attns:
                 Dictionary of attention tensors.

                - ``attns["std"]``:
                  Standard attention of shape
                  ``(batch_size, tgt_len, src_len)``,
                  or ``None`` if the decoder has no attention.
        """

        raise NotImplementedError

    def _enable_cache(self, device, pad_mask):
        pass

    def _disable_cache(self):
        pass

    def update_dropout(self, dropout: float, attention_dropout: Optional[float] = None) -> None:
        """Update dropout rates dynamically.

        Args:
            dropout: New dropout rate for standard dropout layers.
            attention_dropout: New dropout rate for attention layers (if applicable).
        """
        # Default implementation does nothing - override in subclasses that support it
        pass
