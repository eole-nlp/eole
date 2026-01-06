"""Mean pooling encoder implementation."""

from typing import Optional, Tuple
import torch

from eole.encoders.encoder import EncoderBase


class MeanEncoder(EncoderBase):
    """
    Minimal encoder that applies mean pooling over the sequence.

    Returns the input embeddings unchanged as encoder output, and provides
    mean-pooled representations as the final hidden state.

    Args:
        encoder_config: Encoder configuration
        running_config: Runtime configuration (optional, unused)
    """

    def __init__(self, encoder_config, running_config=None):
        super().__init__()
        self.num_layers = encoder_config.layers

    def forward(
        self, emb: torch.Tensor, pad_mask: Optional[torch.Tensor] = None, **kwargs
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply mean pooling over sequence dimension.

        Args:
            emb: Input embeddings (batch, seq_len, emb_dim)
            pad_mask: Padding mask (batch, seq_len)
                     False for values, True for padding
            **kwargs: Additional arguments (ignored)

        Returns:
            Tuple of:
                - Encoder output: unchanged input embeddings (batch, seq_len, emb_dim)
                - Final hidden state: tuple of (mean, mean) where mean has shape
                  (num_layers, batch, emb_dim)
        """
        batch_size, seq_len, emb_dim = emb.size()

        # Compute mean pooling with or without padding mask
        if pad_mask is not None:
            mean = self._masked_mean_pool(emb, pad_mask)
        else:
            mean = emb.mean(dim=1)  # (batch, emb_dim)

        # Expand to match expected hidden state shape for compatibility
        # with decoders expecting (num_layers, batch, hidden_dim)
        mean_expanded = mean.unsqueeze(0).expand(self.num_layers, batch_size, emb_dim)

        enc_out = emb
        enc_final_hs = (mean_expanded, mean_expanded)

        return enc_out, enc_final_hs

    def _masked_mean_pool(self, emb: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute mean pooling while excluding padding positions.

        Args:
            emb: Input embeddings (batch, seq_len, emb_dim)
            pad_mask: Padding mask - can be (batch, seq_len) or (batch, 1, 1, seq_len)
                     False for values, True for padding

        Returns:
            Mean-pooled tensor (batch, emb_dim)
        """
        # Squeeze mask to (batch, seq_len) if it has extra dimensions
        if pad_mask.dim() > 2:
            pad_mask = pad_mask.squeeze(1).squeeze(1)  # (batch, seq_len)

        # Convert mask: True (pad) -> 0.0, False (value) -> 1.0
        mask = (~pad_mask).float().unsqueeze(2)  # (batch, seq_len, 1)

        # Compute weighted sum and normalize by count of non-padding positions
        masked_sum = (emb * mask).sum(dim=1)  # (batch, emb_dim)
        mask_count = mask.sum(dim=1).clamp(min=1.0)  # (batch, 1), avoid div by zero

        return masked_sum / mask_count
