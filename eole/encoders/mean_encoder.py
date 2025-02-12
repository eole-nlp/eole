"""Define a minimal encoder."""

from eole.encoders.encoder import EncoderBase
import torch


class MeanEncoder(EncoderBase):
    """A trivial non-recurrent encoder. Simply applies mean pooling.

    Args:
        encoder_config (eole.config.ModelConfig)
        embeddings (eole.modules.Embeddings):
          embeddings to use, should have positional encodings
        running_config (TrainingConfig / InferenceConfig)
    """

    def __init__(self, encoder_config, running_config=None):
        super(MeanEncoder, self).__init__()
        self.num_layers = encoder_config.layers

    @classmethod
    def from_config(cls, encoder_config, running_config=None):
        """Alternate constructor."""
        # config = opt.model.encoder  # MeanEncoderConfig
        return cls(encoder_config)

    def forward(self, emb, **kwargs):
        """See :func:`EncoderBase.forward()`"""
        pad_mask = kwargs.pop("pad_mask", None)
        batch, _, emb_dim = emb.size()

        if pad_mask is not None:
            # we avoid padding while mean pooling
            pad_mask = (~pad_mask).float()
            mean = torch.bmm(pad_mask, emb).mean(1)
        else:
            mean = emb.mean(1)

        mean = mean.expand(self.num_layers, batch, emb_dim)
        enc_out = emb
        enc_final_hs = (mean, mean)
        return enc_out, enc_final_hs
