"""Implementation of Transformer Encoder from 'Attention is All You Need'"""

from typing import Optional, Tuple, Union
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

        layer_style = getattr(encoder_config, "encoder_layer_style", "standard")
        self.postnorm_style = layer_style in {"postnorm", "roberta"}
        self.parallel_residual = encoder_config.parallel_residual
        self.ffn_layernorm = encoder_config.ffn_layernorm
        self.dropout_p = getattr(running_config, "dropout", [0.0])[0] if running_config else 0.0

        # Layer components
        if not self.postnorm_style:
            self.input_layernorm = LayerNorm[encoder_config.layer_norm](
                encoder_config.hidden_size, eps=encoder_config.norm_eps
            )
        self.self_attn = SelfMHA(
            encoder_config,
            running_config=running_config,
            is_decoder=False,
        )
        self.dropout = nn.Dropout(self.dropout_p)
        if self.postnorm_style:
            self.attention_layer_norm = LayerNorm[encoder_config.layer_norm](
                encoder_config.hidden_size, eps=encoder_config.norm_eps
            )
        else:
            self.post_attention_layernorm = LayerNorm[encoder_config.layer_norm](
                encoder_config.hidden_size, eps=encoder_config.norm_eps
            )
        if self.ffn_layernorm:
            self.pre_feedforward_layernorm = LayerNorm[encoder_config.layer_norm](
                encoder_config.hidden_size, eps=encoder_config.norm_eps
            )
            self.post_feedforward_layernorm = LayerNorm[encoder_config.layer_norm](
                encoder_config.hidden_size, eps=encoder_config.norm_eps
            )
        elif self.postnorm_style:
            self.output_layer_norm = LayerNorm[encoder_config.layer_norm](
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
        if self.postnorm_style:
            attn_in = layer_in
        else:
            attn_in = self.input_layernorm(layer_in)
        self_attn, _ = self.self_attn(
            attn_in,
            attn_mask=attn_mask,
            position_embeddings=position_embeddings,
        )
        if self.dropout_p > 0:
            self_attn = self.dropout(self_attn)

        if self.postnorm_style and not self.ffn_layernorm:
            attn_out = self.attention_layer_norm(layer_in + self_attn)
            return self.output_layer_norm(attn_out + self.mlp(attn_out))

        if self.ffn_layernorm:
            # Gemma4-style: post_attention_layernorm applied to attn output BEFORE residual,
            # then separate pre/post feedforward layernorms wrap the MLP block.
            self_attn = self.post_attention_layernorm(self_attn)
            out = layer_in + self_attn
            ff_pre = self.pre_feedforward_layernorm(out)
            ff_out = self.post_feedforward_layernorm(self.mlp(ff_pre))
            return out + ff_out

        # apply residual
        self_attn.add_(layer_in)

        # prepare feed-forward input
        ff_in = self.post_attention_layernorm(self_attn) if not self.parallel_residual else attn_in

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
        if getattr(encoder_config, "final_encoder_layer_norm", True):
            self.layer_norm = LayerNorm[encoder_config.layer_norm](
                encoder_config.hidden_size, eps=encoder_config.norm_eps
            )
        else:
            self.layer_norm = None

    def forward(
        self, emb: torch.Tensor, pad_mask: Optional[torch.Tensor] = None, return_hidden_states: bool = False, **kwargs
    ) -> Union[Tuple[torch.Tensor, None], Tuple[torch.Tensor, None, list[torch.Tensor]]]:
        """
        Encode input embeddings.

        Args:
            emb: Input embeddings with positional encodings
                 Shape: (batch_size, src_len, model_dim)
            pad_mask: Padding mask (batch, src_len)
                     False for values, True for padding
            **kwargs: Additional arguments (ignored)

        Returns:
            If ``return_hidden_states`` is ``False``:
                - (encoded_output, None)
            If ``return_hidden_states`` is ``True``:
                - (encoded_output, None, hidden_states)

        Raises:
            ValueError: If pad_mask is not provided
        """
        assert pad_mask is not None, "TransformerEncoder requires pad_mask"

        if self.rope.cos_sin is not None:
            position_embeddings = self.rope.cos_sin[: emb.size(1)]
        else:
            position_embeddings = None
        # Expand to (batch, 1, 1, maxlen) for broadcasting in attention
        pad_mask = pad_mask.unsqueeze(1)
        attn_mask = ~pad_mask

        hidden_states = [emb] if return_hidden_states else None
        for layer in self.transformer_layers:
            emb = layer(emb, attn_mask, position_embeddings=position_embeddings)
            if hidden_states is not None:
                hidden_states.append(emb)

        output = self.layer_norm(emb) if self.layer_norm is not None else emb
        if hidden_states is not None:
            hidden_states[-1] = output
            return output, None, hidden_states
        return output, None

    def update_dropout(self, dropout: float, attention_dropout: float) -> None:
        """Update dropout rates for all transformer layers."""
        for layer in self.transformer_layers:
            layer.update_dropout(dropout, attention_dropout)
