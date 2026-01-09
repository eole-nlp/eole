"""Stacked RNN implementations for input-feeding decoder."""

import torch
import torch.nn as nn
from typing import Tuple, List
from abc import ABC, abstractmethod


class StackedRNNBase(nn.Module, ABC):
    """
    Base class for stacked RNN cells.

    Used in decoders with input feeding, where we need manual
    control over each time step rather than using the batched
    RNN implementations.
    """

    def __init__(self, num_layers: int, input_size: int, hidden_size: int, dropout: float):
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if dropout < 0 or dropout > 1:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")

        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        self.layers = self._build_layers(num_layers, input_size, hidden_size)

    @abstractmethod
    def _build_layers(self, num_layers: int, input_size: int, hidden_size: int) -> nn.ModuleList:
        """Build the RNN cell layers."""
        raise NotImplementedError

    @abstractmethod
    def forward(
        self, input_feed: torch.Tensor, hidden: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Forward pass through all layers."""
        raise NotImplementedError

    def _apply_dropout(self, tensor: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Apply dropout between layers (but not after the last layer)."""
        if self.dropout is not None and layer_idx + 1 < self.num_layers:
            return self.dropout(tensor)
        return tensor


class StackedLSTM(StackedRNNBase):
    """
    Stacked LSTM implementation for input-feeding decoder.

    Each layer receives the hidden output from the previous layer,
    with dropout applied between layers.

    Args:
        num_layers: Number of LSTM layers to stack
        input_size: Input dimension (embeddings + attention context for first call)
        hidden_size: Hidden state dimension
        dropout: Dropout probability applied between layers
    """

    def _build_layers(self, num_layers: int, input_size: int, hidden_size: int) -> nn.ModuleList:
        """Build LSTM cell layers with changing input sizes."""
        layers = nn.ModuleList()

        for i in range(num_layers):
            # First layer takes input_size, subsequent layers take hidden_size
            layer_input_size = input_size if i == 0 else hidden_size
            layers.append(nn.LSTMCell(layer_input_size, hidden_size))

        return layers

    def forward(
        self, input_feed: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through stacked LSTM layers.

        Args:
            input_feed: Input tensor (batch, input_size)
            hidden: Tuple of (h_0, c_0) where each is (num_layers, batch, hidden_size)

        Returns:
            Tuple of:
                - Final layer output (batch, hidden_size)
                - New hidden state tuple (h_1, c_1)
        """
        h_0, c_0 = hidden
        h_1_list: List[torch.Tensor] = []
        c_1_list: List[torch.Tensor] = []

        layer_input = input_feed

        for i, layer in enumerate(self.layers):
            # Apply LSTM cell
            h_1_i, c_1_i = layer(layer_input, (h_0[i], c_0[i]))

            # Store new states
            h_1_list.append(h_1_i)
            c_1_list.append(c_1_i)

            # Prepare input for next layer (with dropout)
            layer_input = self._apply_dropout(h_1_i, i)

        # Stack all hidden states
        h_1 = torch.stack(h_1_list, dim=0)
        c_1 = torch.stack(c_1_list, dim=0)

        # Return final layer output and all hidden states
        return layer_input, (h_1, c_1)


class StackedGRU(StackedRNNBase):
    """
    Stacked GRU implementation for input-feeding decoder.

    Each layer receives the hidden output from the previous layer,
    with dropout applied between layers.

    Args:
        num_layers: Number of GRU layers to stack
        input_size: Input dimension (embeddings + attention context for first call)
        hidden_size: Hidden state dimension
        dropout: Dropout probability applied between layers
    """

    def _build_layers(self, num_layers: int, input_size: int, hidden_size: int) -> nn.ModuleList:
        """Build GRU cell layers with changing input sizes."""
        layers = nn.ModuleList()

        for i in range(num_layers):
            # First layer takes input_size, subsequent layers take hidden_size
            layer_input_size = input_size if i == 0 else hidden_size
            layers.append(nn.GRUCell(layer_input_size, hidden_size))

        return layers

    def forward(
        self, input_feed: torch.Tensor, hidden: Tuple[torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        """
        Forward pass through stacked GRU layers.

        Args:
            input_feed: Input tensor (batch, input_size)
            hidden: Tuple containing h_0 of shape (num_layers, batch, hidden_size)

        Returns:
            Tuple of:
                - Final layer output (batch, hidden_size)
                - New hidden state tuple (h_1,)
        """
        h_0 = hidden[0]
        h_1_list: List[torch.Tensor] = []

        layer_input = input_feed

        for i, layer in enumerate(self.layers):
            # Apply GRU cell
            h_1_i = layer(layer_input, h_0[i])

            # Store new state
            h_1_list.append(h_1_i)

            # Prepare input for next layer (with dropout)
            layer_input = self._apply_dropout(h_1_i, i)

        # Stack all hidden states
        h_1 = torch.stack(h_1_list, dim=0)

        # Return final layer output and all hidden states
        # Note: Wrapped in tuple for consistency with LSTM interface
        return layer_input, (h_1,)
