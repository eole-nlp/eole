"""Base class for encoders and generic multi encoders."""

import torch.nn as nn


class EncoderBase(nn.Module):
    """
    Base encoder class. Specifies the interface used by different encoder types
    and required by :class:
    `eole.Models.EncoderDecoderModel`
    `eole.Models.EncoderModel`

    """

    @classmethod
    def from_config(cls, model_config, running_config=None):
        raise NotImplementedError

    def forward(self, emb, mask=None):
        """
        Args:
            emb (FloatTensor):
               embeddings ``(batch, src_len, dim)``
            mask (BoolTensor):
               mask ``(batch, maxlen)`` False when value, True when pad
        Returns:
            (FloatTensor, FloatTensor, FloatTensor):

            * enc_out (encoder output used for attention),
              ``(batch, src_len, hidden_size)``
              for bidirectional rnn last dimension is 2x hidden_size
            * enc_final_hs: encoder final hidden state
              ``(num_layers x dir, batch, hidden_size)``
              In the case of LSTM this is a tuple.
        """

        raise NotImplementedError
