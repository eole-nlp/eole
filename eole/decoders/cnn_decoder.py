"""Implementation of the CNN Decoder part of
"Convolutional Sequence to Sequence Learning"
"""
import torch
import torch.nn as nn

from eole.modules.conv_multi_step_attention import ConvMultiStepAttention
from eole.utils.cnn_factory import shape_transform, GatedConv
from eole.decoders.decoder import DecoderBase

SCALE_WEIGHT = 0.5**0.5


class CNNDecoder(DecoderBase):
    """Decoder based on "Convolutional Sequence to Sequence Learning"
    :cite:`DBLP:journals/corr/GehringAGYD17`.

    Consists of residual convolutional layers, with ConvMultiStepAttention.
    """

    def __init__(
        self,
        model_config,
        running_config=None,
    ):
        super(CNNDecoder, self).__init__()

        self.cnn_kernel_width = model_config.cnn_kernel_width

        # Decoder State
        self.state = {}

        input_size = model_config.hidden_size  # we need embeddings.src_vec_size
        self.linear = nn.Linear(input_size, model_config.hidden_size)
        self.conv_layers = nn.ModuleList(
            [
                GatedConv(
                    model_config.hidden_size,
                    model_config.cnn_kernel_width,
                    getattr(running_config, "dropout", [0.0])[0],
                    True,
                )
                for i in range(model_config.layers)
            ]
        )
        self.attn_layers = nn.ModuleList(
            [
                ConvMultiStepAttention(model_config.hidden_size)
                for i in range(model_config.layers)
            ]
        )

    @classmethod
    def from_config(cls, model_config, running_config=None):
        """Alternate constructor."""
        return cls(
            model_config,
            running_config,
        )

    def init_state(self, **kwargs):
        """Init decoder state."""
        enc_out = kwargs.pop("enc_out", None)
        enc_final_hs = kwargs.pop("enc_final_hs", None)
        self.state["src"] = (enc_out + enc_final_hs) * SCALE_WEIGHT
        self.state["previous_input"] = None

    def map_state(self, fn):
        self.state["src"] = fn(self.state["src"], 0)
        if self.state["previous_input"] is not None:
            self.state["previous_input"] = fn(self.state["previous_input"], 0)

    def detach_state(self):
        self.state["previous_input"] = self.state["previous_input"].detach()

    def forward(self, emb, enc_out, step=None, **kwargs):
        """See :obj:`eole.modules.RNNDecoderBase.forward()`"""

        if self.state["previous_input"] is not None:
            emb = torch.cat([self.state["previous_input"], emb], 1)

        dec_outs = []
        attns = {"std": []}

        assert emb.dim() == 3  # batch x len x embedding_dim

        tgt_emb = emb
        # The output of CNNEncoder.
        enc_out_t = enc_out
        # The combination of output of CNNEncoder and source embeddings.
        enc_out_c = self.state["src"]

        emb_reshape = tgt_emb.view(tgt_emb.size(0) * tgt_emb.size(1), -1)
        linear_out = self.linear(emb_reshape)
        x = linear_out.view(tgt_emb.size(0), tgt_emb.size(1), -1)
        x = shape_transform(x)

        pad = torch.zeros(x.size(0), x.size(1), self.cnn_kernel_width - 1, 1)

        pad = pad.type_as(x)
        base_target_emb = x

        for conv, attention in zip(self.conv_layers, self.attn_layers):
            new_target_input = torch.cat([pad, x], 2)
            out = conv(new_target_input)
            c, attn = attention(base_target_emb, out, enc_out_t, enc_out_c)
            x = (x + (c + out) * SCALE_WEIGHT) * SCALE_WEIGHT

        dec_outs = x.squeeze(3).transpose(1, 2)

        # Process the result and update the attentions.
        if self.state["previous_input"] is not None:
            dec_outs = dec_outs[:, self.state["previous_input"].size(1) :, :]
            attn = attn[:, self.state["previous_input"].size(1) :].squeeze()
            attn = torch.stack([attn])
        attns["std"] = attn

        # Update the state.
        self.state["previous_input"] = emb
        # TODO change the way attns is returned dict => list or tuple (onnx)
        return dec_outs, attns

    def update_dropout(self, dropout, attention_dropout=None):
        for layer in self.conv_layers:
            layer.dropout.p = dropout
