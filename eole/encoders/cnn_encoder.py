"""
Implementation of "Convolutional Sequence to Sequence Learning"
"""
import torch.nn as nn

from eole.encoders.encoder import EncoderBase
from eole.utils.cnn_factory import shape_transform, StackedCNN

SCALE_WEIGHT = 0.5**0.5


class CNNEncoder(EncoderBase):
    """Encoder based on "Convolutional Sequence to Sequence Learning"
    :cite:`DBLP:journals/corr/GehringAGYD17`.
    """

    def __init__(self, model_config, running_config=None):
        super(CNNEncoder, self).__init__()

        input_size = (
            model_config.hidden_size
        )  # we need embeddings.src_word_vec_size instead
        self.linear = nn.Linear(input_size, model_config.hidden_size)
        self.cnn = StackedCNN(
            model_config.layers,
            model_config.hidden_size,
            model_config.cnn_kernel_width,
            getattr(running_config, "dropout", [0.0])[0],
        )

    @classmethod
    def from_config(cls, model_config, running_config=None):
        """Alternate constructor."""
        # config = opt.model.encoder  # CnnEncoderConfig
        return cls(
            model_config,  # CnnEncoderConfig
            running_config,  # might be better to set this out of the model building logic (update_dropout call sometime in training) # noqa: E501
        )

    def forward(self, emb, mask=None):
        """See :func:`EncoderBase.forward()`"""
        # batch x len x dim

        emb_reshape = emb.view(emb.size(0) * emb.size(1), -1)
        emb_remap = self.linear(emb_reshape)
        emb_remap = emb_remap.view(emb.size(0), emb.size(1), -1)
        emb_remap = shape_transform(emb_remap)
        out = self.cnn(emb_remap)

        return out.squeeze(3), emb_remap.squeeze(3)

    def update_dropout(self, dropout, attention_dropout=None):
        self.cnn.dropout.p = dropout
