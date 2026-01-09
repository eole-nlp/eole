"""CNN Decoder implementation from 'Convolutional Sequence to Sequence Learning'."""

from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn

from eole.decoders.decoder import DecoderBase
from eole.modules.conv_multi_step_attention import ConvMultiStepAttention
from eole.utils.cnn_factory import GatedConv

SCALE_WEIGHT = 0.5**0.5


class CNNDecoder(DecoderBase):
    """Convolutional CNN Decoder with multi-step attention."""

    def __init__(self, decoder_config, running_config=None):
        super().__init__()

        self.cnn_kernel_width = decoder_config.cnn_kernel_width
        hidden_size = decoder_config.hidden_size
        dropout = getattr(running_config, "dropout", [0.0])[0] if running_config else 0.0
        attention_dropout = getattr(running_config, "attention_dropout", [0.0])[0] if running_config else 0.0

        # Input projection
        self.linear = nn.Linear(hidden_size, hidden_size)

        # Stacked CNN layers
        self.conv_layers = nn.ModuleList(
            [
                GatedConv(hidden_size, kernel_width=decoder_config.cnn_kernel_width, dropout=dropout, nopad=True)
                for _ in range(decoder_config.layers)
            ]
        )

        # Attention layers
        self.attn_layers = nn.ModuleList(
            [ConvMultiStepAttention(hidden_size, dropout=attention_dropout) for _ in range(decoder_config.layers)]
        )

        # Decoder state
        self.state = {}

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

    def forward(
        self,
        emb: torch.Tensor,
        enc_out: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Decode a sequence using convolutional layers.

        Args:
            emb (Tensor):
                Target embeddings of shape
                ``(batch_size, tgt_len, hidden_size)``.

            enc_out (Tensor):
                Encoder outputs of shape
                ``(batch_size, src_len, hidden_size)``.

            step (int, optional):
                Decoding step for incremental decoding.

            **kwargs:
                Additional decoder-specific arguments.

        Returns:
            (Tensor, Dict[str, Tensor]):

            * dec_outs:
                Decoder outputs of shape
                ``(batch_size, tgt_len, hidden_size)``.

            * attns:
                Dictionary of attention tensors.

                - ``attns["std"]``:
                  Attention weights of shape
                  ``(batch_size, tgt_len, src_len)``.
        """

        assert enc_out is not None, "CNNDecoder requires enc_out"

        src_pad_mask = kwargs.pop("src_pad_mask", None)

        # Incremental decoding: grow previous_input
        if self.state["previous_input"] is not None:
            emb = torch.cat([self.state["previous_input"], emb], dim=1)

        assert emb.dim() == 3

        enc_out_t = enc_out
        enc_out_c = self.state["src"]

        batch_size, tgt_len, _ = emb.size()

        projected = self.linear(emb.view(batch_size * tgt_len, -1))
        x = projected.view(batch_size, tgt_len, -1).transpose(1, 2)
        base_target_emb = x

        pad = torch.zeros(
            batch_size,
            x.size(1),
            self.cnn_kernel_width - 1,
            dtype=x.dtype,
            device=x.device,
        )

        attn = None
        for conv, attention in zip(self.conv_layers, self.attn_layers):
            new_target_input = torch.cat([pad, x], dim=2)
            out = conv(new_target_input)
            c, attn = attention(
                base_target_emb,
                out,
                enc_out_t,
                enc_out_c,
                src_pad_mask,
            )
            # x = (x + (c + out) * SCALE_WEIGHT) * SCALE_WEIGHT
            x = (x + c + out) * SCALE_WEIGHT

        dec_outs = x.transpose(1, 2)

        # Incremental slicing — keep batch dimension
        if self.state["previous_input"] is not None:
            prev_len = self.state["previous_input"].size(1)
            dec_outs = dec_outs[:, prev_len:, :]
            attn = attn[:, prev_len:, :]

        self.state["previous_input"] = emb

        attns = {"std": attn}
        return dec_outs, attns

    def update_dropout(self, dropout: float, attention_dropout: Optional[float] = None) -> None:
        """Update dropout rates for all convolutional layers."""
        for layer in self.conv_layers:
            if hasattr(layer, "dropout"):
                layer.dropout.p = dropout
        if attention_dropout is not None:
            for layer in self.attn_layers:
                if hasattr(layer, "dropout"):
                    layer.dropout.p = attention_dropout
