"""
Implementation of "Attention is All You Need"
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
        self.input_layernorm = LayerNorm[model_config.layer_norm](
            model_config.hidden_size, eps=model_config.norm_eps
        )
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

    def forward(self, layer_in, mask):
        """
        Args:
            layer_in (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, 1, src_len)``

        Returns:
            (FloatTensor):
            * layer_out ``(batch_size, src_len, model_dim)``
        """
        norm_layer_in = self.input_layernorm(layer_in)
        context, _ = self.self_attn(norm_layer_in, mask=mask)
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
        embeddings (eole.modules.Embeddings):
          embeddings to use, should have positional encodings
        running_config (TrainingConfig / InferenceConfig)
    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * enc_out ``(batch_size, src_len, model_dim)``
        * encoder final state: None in the case of Transformer
        * src_len ``(batch_size)``
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
        # This is the Encoder out layer norm
        self.layer_norm = LayerNorm[model_config.layer_norm](
            model_config.hidden_size, eps=model_config.norm_eps
        )

    @classmethod
    def from_config(cls, model_config, running_config=None):
        """Alternate constructor."""
        return cls(
            model_config,  # TransformerEncoderConfig
            running_config,
        )

    def forward(self, emb, mask=None):
        """See :func:`EncoderBase.forward()`"""
        enc_out = emb
        mask = mask.unsqueeze(1).unsqueeze(1)
        # mask is now (batch x 1 x 1 x maxlen)
        mask = mask.expand(-1, -1, mask.size(3), -1)
        # Padding mask is now (batch x 1 x maxlen x maxlen)
        # 1 to be expanded to number of heads in MHA
        # Run the forward pass of every layer of the tranformer.

        for layer in self.transformer_layers:
            enc_out = layer(enc_out, mask)
        enc_out = self.layer_norm(enc_out)
        return enc_out, None

    def update_dropout(self, dropout, attention_dropout):
        for layer in self.transformer_layers:
            layer.update_dropout(dropout, attention_dropout)
