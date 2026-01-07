"""RNN-based encoder implementation."""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from eole.encoders.encoder import EncoderBase


class RNNEncoder(EncoderBase):
    """
    Generic recurrent neural network encoder supporting LSTM, GRU, and RNN.

    Args:
        encoder_config: Encoder configuration
        running_config: Runtime configuration (optional)
    """

    def __init__(self, encoder_config, running_config=None):
        super().__init__()

        # Determine bidirectionality and adjust hidden size
        self.bidirectional = encoder_config.encoder_type == "brnn"
        num_directions = 2 if self.bidirectional else 1

        if encoder_config.hidden_size % num_directions != 0:
            raise ValueError(
                f"hidden_size ({encoder_config.hidden_size}) must be divisible by num_directions ({num_directions})"
            )

        hidden_size = encoder_config.hidden_size // num_directions
        if running_config is None:
            dropout = 0.0
        else:
            dropout = getattr(running_config, "dropout", [0.0])[0]

        # Build RNN
        rnn_class = getattr(nn, encoder_config.rnn_type)
        self.rnn = rnn_class(
            input_size=encoder_config.src_word_vec_size,
            hidden_size=hidden_size,
            num_layers=encoder_config.layers,
            dropout=dropout,
            bidirectional=self.bidirectional,
            batch_first=True,
        )

        # Optional bridge network
        self.use_bridge = encoder_config.bridge
        if self.use_bridge:
            self.bridge = self._build_bridge(encoder_config.rnn_type, hidden_size, encoder_config.layers)

    def _build_bridge(self, rnn_type: str, hidden_size: int, num_layers: int) -> nn.ModuleList:
        """
        Build bridge network to transform encoder final states.

        Args:
            rnn_type: Type of RNN (LSTM requires two bridges for h and c)
            hidden_size: Hidden dimension size
            num_layers: Number of RNN layers

        Returns:
            ModuleList of linear transformations
        """
        self.total_hidden_dim = hidden_size * num_layers
        num_states = 2 if rnn_type == "LSTM" else 1

        return nn.ModuleList(
            [nn.Linear(self.total_hidden_dim, self.total_hidden_dim, bias=True) for _ in range(num_states)]
        )

    def forward(
        self, emb: torch.Tensor, pad_mask: Optional[torch.Tensor] = None, **kwargs
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Encode input embeddings through RNN.

        Args:
            emb: Input embeddings (batch, src_len, dim)
            pad_mask: Padding mask (optional, not used by base RNN)
            **kwargs: Additional arguments

        Returns:
            Tuple of:
                - RNN outputs (batch, src_len, hidden_size)
                - Final hidden state(s)
        """
        enc_out, enc_final_hs = self.rnn(emb)

        if self.use_bridge:
            enc_final_hs = self._apply_bridge(enc_final_hs)

        return enc_out, enc_final_hs

    def _apply_bridge(
        self, hidden: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Transform hidden states through bridge network.

        Args:
            hidden: Final hidden state(s) from RNN
                   Shape: (num_layers * directions, batch, hidden_size)

        Returns:
            Transformed hidden state(s)
        """
        if isinstance(hidden, tuple):  # LSTM case
            return tuple(self._transform_state(state, bridge) for state, bridge in zip(hidden, self.bridge))
        else:  # GRU/RNN case
            return self._transform_state(hidden, self.bridge[0])

    def _transform_state(self, state: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
        """
        Apply linear transformation to RNN state.

        Reshapes from (num_layers * dir, batch, hidden) to (batch, -1),
        applies linear + ReLU, then reshapes back.

        Args:
            state: Input state tensor
            linear: Linear transformation layer

        Returns:
            Transformed state tensor
        """
        # Permute to (batch, num_layers * dir, hidden)
        state = state.permute(1, 0, 2).contiguous()
        batch_size, num_layers_dir, hidden_size = state.shape

        # Flatten and transform
        state_flat = state.view(batch_size, -1)
        transformed = F.relu(linear(state_flat))

        # Reshape back to original dimensions
        transformed = transformed.view(batch_size, num_layers_dir, hidden_size)

        # Permute back to (num_layers * dir, batch, hidden)
        return transformed.permute(1, 0, 2).contiguous()

    def update_dropout(self, dropout: float, attention_dropout: Optional[float] = None) -> None:
        """Update RNN dropout rate."""
        self.rnn.dropout = dropout
