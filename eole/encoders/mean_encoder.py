"""Define a minimal encoder."""
from eole.encoders.encoder import EncoderBase
import torch


class MeanEncoder(EncoderBase):
    """A trivial non-recurrent encoder. Simply applies mean pooling.

    Args:
        model_config (eole.config.ModelConfig)
        embeddings (eole.modules.Embeddings):
          embeddings to use, should have positional encodings
        running_config (TrainingConfig / InferenceConfig)
    """

    def __init__(self, model_config, running_config=None):
        super(MeanEncoder, self).__init__()
        self.num_layers = model_config.layers

    @classmethod
    def from_config(cls, model_config, running_config=None):
        """Alternate constructor."""
        # config = opt.model.encoder  # MeanEncoderConfig
        return cls(model_config)

    def forward(self, emb, mask=None):
        """See :func:`EncoderBase.forward()`"""

        batch, _, emb_dim = emb.size()

        if mask is not None:
            # we avoid padding while mean pooling
            mask = (~mask).float()
            mean = torch.bmm(mask.unsqueeze(1), emb).mean(1)
        else:
            mean = emb.mean(1)

        mean = mean.expand(self.num_layers, batch, emb_dim)
        enc_out = emb
        enc_final_hs = (mean, mean)
        return enc_out, enc_final_hs
