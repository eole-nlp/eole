"""Implementation of the CNN Decoder part of
"Convolutional Sequence to Sequence Learning"
"""

import torch
import torch.nn as nn

from eole.modules.conv_multi_step_attention import ConvMultiStepAttention
from eole.utils.cnn_factory import GatedConv
from eole.decoders.decoder import DecoderBase

SCALE_WEIGHT = 0.5**0.5


class CNNDecoder(DecoderBase):
    """Convolutional CNN Decoder with multi-step attention."""

    def __init__(self, decoder_config, running_config=None, with_cross_attn=False):
        super().__init__()

        hidden_size = decoder_config.hidden_size
        dropout = getattr(running_config, "dropout", [0.0])[0] if running_config else 0.0

        # Input projection
        self.linear = nn.Linear(hidden_size, hidden_size)

        # Stacked CNN layers
        self.conv_layers = nn.ModuleList(
            [
                GatedConv(hidden_size, kernel_width=decoder_config.cnn_kernel_width, dropout=dropout)
                for _ in range(decoder_config.layers)
            ]
        )

        # Attention layers
        self.attn_layers = nn.ModuleList([ConvMultiStepAttention(hidden_size) for _ in range(decoder_config.layers)])

        # Decoder state
        self.state = {}

    @classmethod
    def from_config(cls, decoder_config, running_config=None, with_cross_attn=False):
        """Alternate constructor."""
        return cls(
            decoder_config,
            running_config,
            with_cross_attn=False,
        )

    def init_state(self, **kwargs):
        """Initialize decoder state."""
        enc_out = kwargs.pop("enc_out", None)
        enc_final_hs = kwargs.pop("enc_final_hs", None)
        self.state["src"] = (enc_out + enc_final_hs) * SCALE_WEIGHT
        self.state["previous_input"] = None

    def map_state(self, fn):
        self.state["src"] = fn(self.state["src"])
        if self.state["previous_input"] is not None:
            self.state["previous_input"] = fn(self.state["previous_input"])

    def detach_state(self):
        self.state["previous_input"] = self.state["previous_input"].detach()

    def forward(self, emb, enc_out, step=None, **kwargs):
        """
        emb: (batch, tgt_len, hidden)
        enc_out: (batch, src_len, hidden)
        """
        if self.state["previous_input"] is not None:
            emb = torch.cat([self.state["previous_input"], emb], dim=1)

        batch, tgt_len, hidden = emb.size()

        # Linear projection
        x = self.linear(emb.view(batch * tgt_len, hidden))
        x = x.view(batch, tgt_len, hidden)

        # Transpose for Conv1d: (B, hidden, T)
        x = x.transpose(1, 2)
        base_target_emb = x.clone()

        # Residual convolution + attention
        enc_out_c = self.state["src"]
        enc_out_t = enc_out
        for conv, attn in zip(self.conv_layers, self.attn_layers):
            out = conv(x)
            c, a = attn(base_target_emb, out, enc_out_t, enc_out_c)
            x = SCALE_WEIGHT * (x + SCALE_WEIGHT * (out + c))

        # Transpose back: (B, T, hidden)
        dec_outs = x.transpose(1, 2)  # (B, T, hidden)
        if self.state["previous_input"] is not None:
            dec_outs = dec_outs[:, self.state["previous_input"].size(1) :, :]
            attn = attn[:, self.state["previous_input"].size(1) :].squeeze()
            attn = torch.stack([attn])

        attns = {"std": attn}
        self.state["previous_input"] = emb

        return dec_outs, attns

    def update_dropout(self, dropout, attention_dropout=None):
        for layer in self.conv_layers:
            layer.dropout.p = dropout
