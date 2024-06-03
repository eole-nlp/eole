"""Define RNN-based encoders."""
import torch.nn as nn
import torch.nn.functional as F

from eole.encoders.encoder import EncoderBase


class RNNEncoder(EncoderBase):
    """A generic recurrent neural network encoder.

    Args:
        model_config (eole.config.ModelConfig)
        running_config (TrainingConfig / InferenceConfig)
    """

    def __init__(self, model_config, running_config=None):
        super(RNNEncoder, self).__init__()

        bidirectional = model_config.encoder_type == "brnn"

        num_directions = 2 if bidirectional else 1
        assert model_config.hidden_size % num_directions == 0
        hidden_size = model_config.hidden_size // num_directions

        self.rnn = getattr(nn, model_config.rnn_type)(
            batch_first=True,
            input_size=model_config.src_word_vec_size,
            hidden_size=hidden_size,
            num_layers=model_config.layers,
            dropout=getattr(running_config, "dropout", [0.0])[0],
            bidirectional=bidirectional,
        )

        # Initialize the bridge layer
        self.use_bridge = model_config.bridge
        if self.use_bridge:
            self._initialize_bridge(
                model_config.rnn_type, hidden_size, model_config.layers
            )

    @classmethod
    def from_config(cls, model_config, running_config=None):
        """Alternate constructor."""
        return cls(model_config, running_config=running_config)

    def forward(self, emb, mask=None):
        """See :func:`EncoderBase.forward()`"""

        enc_out, enc_final_hs = self.rnn(emb)

        if self.use_bridge:
            enc_final_hs = self._bridge(enc_final_hs)

        return enc_out, enc_final_hs

    def _initialize_bridge(self, rnn_type, hidden_size, num_layers):
        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if rnn_type == "LSTM" else 1
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers

        # Build a linear layer for each
        self.bridge = nn.ModuleList(
            [
                nn.Linear(self.total_hidden_dim, self.total_hidden_dim, bias=True)
                for _ in range(number_of_states)
            ]
        )

    def _bridge(self, hidden):
        """Forward hidden state through bridge.
        final hidden state ``(num_layers x dir, batch, hidden_size)``
        """

        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            states = states.permute(1, 0, 2).contiguous()
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            result = F.relu(result).view(size)
            return result.permute(1, 0, 2).contiguous()

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple(
                [
                    bottle_hidden(layer, hidden[ix])
                    for ix, layer in enumerate(self.bridge)
                ]
            )
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs

    def update_dropout(self, dropout, attention_dropout=None):
        self.rnn.dropout = dropout
