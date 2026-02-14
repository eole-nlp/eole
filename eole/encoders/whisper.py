"""Whisper encoder for speech-to-text models."""

import torch
import torch.nn as nn

from eole.encoders.encoder import EncoderBase
from eole.encoders.transformer import TransformerEncoderLayer
from eole.constants import LayerNorm


class WhisperEncoder(EncoderBase):
    """
    Whisper encoder: Conv1d stem + learned positional embeddings + transformer layers.

    Processes mel spectrograms into encoder hidden states for cross-attention
    with the decoder.

    Input: mel spectrogram (batch, num_mels, time)
    Output: (batch, time // 2, hidden_size)

    Args:
        encoder_config: Whisper encoder configuration
        running_config: Runtime configuration (optional)
    """

    def __init__(self, encoder_config, running_config=None):
        super().__init__()

        num_mel_bins = encoder_config.num_mel_bins
        hidden_size = encoder_config.hidden_size
        max_source_positions = encoder_config.max_source_positions

        self.conv1 = nn.Conv1d(num_mel_bins, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(max_source_positions, hidden_size)

        self.transformer_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(encoder_config, running_config=running_config)
                for _ in range(encoder_config.layers)
            ]
        )

        self.layer_norm = LayerNorm[encoder_config.layer_norm](
            hidden_size, eps=encoder_config.norm_eps
        )

        self._attn_mask_cache = None
        self._attn_mask_key = None

    def forward(
        self, emb: torch.Tensor, pad_mask: torch.Tensor | None = None, **kwargs
    ) -> tuple[torch.Tensor, None]:
        """
        Encode mel spectrogram features.

        Args:
            emb: Mel spectrogram tensor (batch, num_mels, time)
            pad_mask: Not used for Whisper (fixed-length input)
            **kwargs: Additional arguments (ignored)

        Returns:
            Tuple of:
                - Encoded output (batch, time//2, hidden_size)
                - None (transformers don't return final state)
        """
        x = torch.nn.functional.gelu(self.conv1(emb))
        x = torch.nn.functional.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        positions = torch.arange(x.size(1), device=x.device)
        x = x + self.embed_positions(positions)

        for layer in self.transformer_layers:
            x = layer(x, attn_mask=self._get_attn_mask(x))

        x = self.layer_norm(x)
        return x, None

    def _get_attn_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Return a cached all-ones attention mask for the given input."""
        key = (x.size(0), x.size(1), x.device)
        if self._attn_mask_cache is None or self._attn_mask_key != key:
            self._attn_mask_cache = torch.ones(
                x.size(0), 1, 1, x.size(1), device=x.device, dtype=torch.bool
            )
            self._attn_mask_key = key
        return self._attn_mask_cache

    def update_dropout(self, dropout: float, attention_dropout: float) -> None:
        """Update dropout rates for all transformer layers."""
        for layer in self.transformer_layers:
            layer.update_dropout(dropout, attention_dropout)
