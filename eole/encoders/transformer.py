"""Implementation of Transformer Encoder from 'Attention is All You Need'"""

from typing import Optional, Tuple
import torch
import torch.nn as nn

from eole.encoders.encoder import EncoderBase
from eole.modules.multi_headed_attn import SelfMHA
from eole.modules.transformer_mlp import MLP
from eole.modules.rope import build_rope
from eole.constants import LayerNorm


class TransformerEncoderLayer(nn.Module):
    """
    Single transformer encoder layer with self-attention and feed-forward network.

    Args:
        encoder_config: Encoder configuration
        running_config: Runtime configuration (optional)
    """

    def __init__(self, encoder_config, running_config=None):
        super().__init__()

        self.parallel_residual = encoder_config.parallel_residual
        self.dropout_p = getattr(running_config, "dropout", [0.0])[0] if running_config else 0.0

        # Layer components
        self.input_layernorm = LayerNorm[encoder_config.layer_norm](
            encoder_config.hidden_size, eps=encoder_config.norm_eps
        )
        self.self_attn = SelfMHA(
            encoder_config,
            running_config=running_config,
            is_decoder=False,
        )
        self.dropout = nn.Dropout(self.dropout_p)
        self.post_attention_layernorm = LayerNorm[encoder_config.layer_norm](
            encoder_config.hidden_size, eps=encoder_config.norm_eps
        )
        self.mlp = MLP(encoder_config, running_config=running_config)

    def forward(
        self,
        layer_in: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        position_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through encoder layer.

        Args:
            layer_in: Input tensor (batch_size, src_len, model_dim)
            attn_mask: Attention mask (batch_size, 1, src_len)
            position_embeddings: Rotary position encodings (optional)

        Returns:
            Output tensor (batch_size, src_len, model_dim)
        """
        norm_layer_in = self.input_layernorm(layer_in)
        self_attn, _ = self.self_attn(
            norm_layer_in,
            attn_mask=attn_mask,
            position_embeddings=position_embeddings,
        )
        if self.dropout_p > 0:
            self_attn = self.dropout(self_attn)

        # apply residual
        self_attn.add_(layer_in)

        # prepare feed-forward input
        ff_in = self.post_attention_layernorm(self_attn) if not self.parallel_residual else norm_layer_in

        # add residual after mlp
        return self_attn + self.mlp(ff_in)

    def update_dropout(self, dropout: float, attention_dropout: float) -> None:
        """Update dropout rates across all sub-modules."""
        self.dropout_p = dropout
        self.dropout.p = dropout
        self.self_attn.update_dropout(attention_dropout)
        self.mlp.update_dropout(dropout)


class TransformerEncoder(EncoderBase):
    """
    Transformer encoder from 'Attention is All You Need'.

    Reference:
        Vaswani et al. (2017) https://arxiv.org/abs/1706.03762

    Args:
        encoder_config: Complete encoder configuration
        running_config: Runtime configuration (optional)
    """

    def __init__(self, encoder_config, running_config=None):
        super().__init__()

        self.rope = build_rope(encoder_config)
        self.transformer_layers = nn.ModuleList(
            [TransformerEncoderLayer(encoder_config, running_config) for _ in range(encoder_config.layers)]
        )
        self.layer_norm = LayerNorm[encoder_config.layer_norm](encoder_config.hidden_size, eps=encoder_config.norm_eps)

    def forward(
        self, emb: torch.Tensor, pad_mask: Optional[torch.Tensor] = None, **kwargs
    ) -> Tuple[torch.Tensor, None]:
        """
        Encode input embeddings.

        Args:
            emb: Input embeddings with positional encodings
                 Shape: (batch_size, src_len, model_dim)
            pad_mask: Padding mask (batch, src_len)
                     False for values, True for padding
            **kwargs: Additional arguments (ignored)

        Returns:
            Tuple of:
                - Encoded output (batch_size, src_len, model_dim)
                - None (transformers don't return final state)

        Raises:
            ValueError: If pad_mask is not provided
        """
        assert pad_mask is not None, "TransformerEncoder requires pad_mask"

        position_embeddings = self.rope.update(emb.size(1), step=None)
        # Expand to (batch, 1, 1, maxlen) for broadcasting in attention
        pad_mask = pad_mask.unsqueeze(1)
        attn_mask = ~pad_mask

        for layer in self.transformer_layers:
            emb = layer(emb, attn_mask, position_embeddings=position_embeddings)

        output = self.layer_norm(emb)
        return output, None

    def update_dropout(self, dropout: float, attention_dropout: float) -> None:
        """Update dropout rates for all transformer layers."""
        for layer in self.transformer_layers:
            layer.update_dropout(dropout, attention_dropout)
