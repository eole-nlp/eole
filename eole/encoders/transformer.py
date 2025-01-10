"""
Implementation of "Attention is All You Need" Transformer Encoder
"""

import torch.nn as nn

from eole.encoders.encoder import EncoderBase
from eole.modules.multi_headed_attn import SelfMHA
from eole.modules.transformer_mlp import MLP
from eole.constants import LayerNorm


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        model_config (eole.config.TransformerEncoderConfig): full encoder config
        running_config
    """

    def __init__(
        self,
        model_config,
        running_config=None,
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.parallel_residual = model_config.parallel_residual
        self.dropout_p = getattr(running_config, "dropout", [0.0])[0]

        # order of layers corresponds to forward flow of tensors
        self.input_layernorm = LayerNorm[model_config.layer_norm](model_config.hidden_size, eps=model_config.norm_eps)
        self.self_attn = SelfMHA(
            model_config,
            running_config=running_config,
            is_decoder=False,
        )
        self.dropout = nn.Dropout(self.dropout_p)
        self.post_attention_layernorm = LayerNorm[model_config.layer_norm](
            model_config.hidden_size, eps=model_config.norm_eps
        )
        self.mlp = MLP(
            model_config,
            running_config=running_config,
        )

    def forward(self, layer_in, pad_mask, position_embeddings=None):
        """
        Args:
            layer_in (FloatTensor): ``(batch_size, src_len, model_dim)``
            pad_mask (LongTensor): ``(batch_size, 1, src_len)``
            position_embeddings (FloatTensor): rotary position encodings, if any

        Returns:
            (FloatTensor):
            * layer_out ``(batch_size, src_len, model_dim)``
        """
        norm_layer_in = self.input_layernorm(layer_in)
        context, _ = self.self_attn(norm_layer_in, attn_mask=~pad_mask, position_embeddings=position_embeddings)
        if self.dropout_p > 0:
            context = self.dropout(context)
        if self.parallel_residual:
            ff_in = norm_layer_in
        else:
            ff_in = self.post_attention_layernorm(context + layer_in)
        # apply post attention norm and add residual after mlp
        layer_out = self.mlp(ff_in) + context + layer_in

        return layer_out

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.mlp.update_dropout(dropout)
        self.dropout.p = dropout


class TransformerEncoder(EncoderBase):
    """The Transformer encoder from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
        model_config (eole.config.TransformerEncoderConfig): full encoder config
        running_config (TrainingConfig / InferenceConfig)
    """

    def __init__(
        self,
        model_config,
        running_config=None,
    ):
        super(TransformerEncoder, self).__init__()
        self.transformer_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    model_config,
                    running_config=running_config,
                )
                for i in range(model_config.layers)
            ]
        )
        self.layer_norm = LayerNorm[model_config.layer_norm](model_config.hidden_size, eps=model_config.norm_eps)

    @classmethod
    def from_config(cls, model_config, running_config=None):
        """Alternate constructor."""
        return cls(
            model_config,
            running_config,
        )

    def forward(self, emb, **kwargs):
        """See :func:`EncoderBase.forward()`

        Args:
            emb (eole.modules.Embeddings):
                embeddings to use, should have positional encodings
            **kwargs
                pad_mask: ``(batch, maxlen)`` False when value, True when pad

        Returns:
            (torch.FloatTensor, torch.FloatTensor):
        * enc_out ``(batch_size, src_len, model_dim)``
        * encoder final state: None in the case of Transformer
        """
        pad_mask = kwargs.pop("pad_mask", None)
        assert pad_mask is not None, "TransformerEncoder requires a src pad mask"
        position_embeddings = kwargs.pop("position_embeddings", None)
        pad_mask = pad_mask.unsqueeze(1)  # batch x 1 x 1 x maxlen
        # dim 1 (heads) and 2 (src_len) will be broadcasted automatically in MHA

        for layer in self.transformer_layers:
            emb = layer(emb, pad_mask, position_embeddings=position_embeddings)
        emb = self.layer_norm(emb)
        return emb, None

    def update_dropout(self, dropout, attention_dropout):
        for layer in self.transformer_layers:
            layer.update_dropout(dropout, attention_dropout)
