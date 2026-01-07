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

    def __init__(self, decoder_config, running_config=None, with_cross_attn=False):
        super().__init__()

        self.cnn_kernel_width = decoder_config.cnn_kernel_width
        hidden_size = decoder_config.hidden_size
        dropout = getattr(running_config, "dropout", [0.0])[0] if running_config else 0.0

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

    def forward(
        self, emb: torch.Tensor, enc_out: torch.Tensor, step: Optional[int] = None, **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Decode target embeddings with attention over encoder outputs.

        Args:
            emb: Target embeddings (batch, tgt_len, hidden_size)
            enc_out: Encoder output (batch, src_len, hidden_size)
            step: Current decoding step (optional, for incremental decoding)
            **kwargs: Additional arguments

        Returns:
            Tuple of:
                - Decoder outputs (batch, tgt_len, hidden_size)
                - Attention weights dict
        """
        # Concatenate with previous input for context
        if self.state["previous_input"] is not None:
            emb = torch.cat([self.state["previous_input"], emb], 1)

        attns = {"std": []}

        assert emb.dim() == 3, f"Expected 3D input, got {emb.dim()}D"

        # Encoder outputs
        enc_out_t = enc_out  # Encoder output (for attention)
        enc_out_c = self.state["src"]  # Combined encoder output + embeddings

        # Project input embeddings
        batch_size, tgt_len, _ = emb.shape
        projected = self.linear(emb.view(batch_size * tgt_len, -1))
        x = projected.view(batch_size, tgt_len, -1)

        # Convert to Conv1d format: (batch, hidden, tgt_len)
        x = x.transpose(1, 2)
        base_target_emb = x

        # Create causal padding (left-pad only, no future information)
        # Shape: (batch, hidden, kernel_width - 1)
        pad = torch.zeros(
            x.size(0),
            x.size(1),
            self.cnn_kernel_width - 1,
            dtype=x.dtype,
            device=x.device,
        )

        # Apply convolutional layers with attention
        for conv, attention in zip(self.conv_layers, self.attn_layers):
            # Add causal padding: only past context, no future
            new_target_input = torch.cat([pad, x], dim=2)

            # Apply gated convolution
            out = conv(new_target_input)

            # Apply attention
            c, attn = attention(base_target_emb, out, enc_out_t, enc_out_c)

            # Residual connection with double scaling
            x = (x + (c + out) * SCALE_WEIGHT) * SCALE_WEIGHT

        # Convert back to (batch, tgt_len, hidden)
        dec_outs = x.transpose(1, 2)

        # If using cached previous input, only return new outputs
        if self.state["previous_input"] is not None:
            prev_len = self.state["previous_input"].size(1)
            dec_outs = dec_outs[:, prev_len:, :]
            attn = attn[:, prev_len:].squeeze()
            attn = torch.stack([attn])

        attns["std"] = attn

        # Update state for next step
        self.state["previous_input"] = emb

        return dec_outs, attns

    def update_dropout(self, dropout: float, attention_dropout: Optional[float] = None) -> None:
        """Update dropout rates for all convolutional layers."""
        for layer in self.conv_layers:
            if hasattr(layer, "dropout"):
                layer.dropout.p = dropout
