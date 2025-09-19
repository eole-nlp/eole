"""
Attention Entropy Computation Utilities

This module provides functions to compute entropy of attention weights
during training and inference, which can be useful for analyzing
attention patterns and model behavior.
"""

import torch
from typing import Dict, Optional, Union, List


def compute_attention_entropy(
    attention_weights: torch.Tensor, mask: Optional[torch.Tensor] = None, eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute entropy of attention weights.

    Entropy is computed as: H(A) = -Σ(A_ij * log(A_ij))
    where A_ij are the attention weights.

    Args:
        attention_weights (torch.Tensor): Attention weights tensor of shape
            [batch_size, num_heads, seq_len_q, seq_len_k] or
            [batch_size, seq_len_q, seq_len_k]
        mask (torch.Tensor, optional): Mask tensor to ignore padded positions.
            Should have the same shape as attention_weights or be broadcastable.
        eps (float): Small epsilon value to avoid log(0). Default: 1e-8

    Returns:
        torch.Tensor: Entropy values. Shape depends on input:
            - If input has heads dimension: [batch_size, num_heads]
            - Otherwise: [batch_size]
    """
    # Ensure attention weights are positive and normalized
    attn = torch.clamp(attention_weights, min=eps)

    # Apply mask if provided
    if mask is not None:
        # Convert mask to same dtype as attention weights
        mask = mask.to(attn.dtype)
        # Zero out masked positions
        attn = attn * mask
        # Renormalize after masking
        if len(attn.shape) == 4:  # [batch, heads, seq_q, seq_k]
            attn = attn / (attn.sum(dim=-1, keepdim=True) + eps)
        else:  # [batch, seq_q, seq_k]
            attn = attn / (attn.sum(dim=-1, keepdim=True) + eps)

    # Compute entropy: H = -Σ(p * log(p))
    log_attn = torch.log(attn + eps)
    entropy = -torch.sum(attn * log_attn, dim=-1)  # Sum over key dimension

    # Average over query dimension to get per-head or per-batch entropy
    entropy = torch.mean(entropy, dim=-1)

    return entropy


def compute_attention_entropy_from_dict(
    attns_dict: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
    attention_types: Optional[List[str]] = None,
    layer_indices: Optional[List[int]] = None,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """
    Compute attention entropy from attention dictionary returned by model.

    Args:
        attns_dict (Dict): Dictionary of attention weights from model forward pass.
            Keys are attention types (e.g., 'std', 'self', 'context').
            Values are either single tensors or lists of tensors (one per layer).
        attention_types (List[str], optional): Which attention types to compute entropy for.
            If None, computes for all available types.
        layer_indices (List[int], optional): Which layer indices to include.
            If None, includes all layers.
        eps (float): Small epsilon value to avoid log(0).

    Returns:
        Dict[str, torch.Tensor]: Dictionary with entropy values.
            Keys are in format "{attention_type}_layer_{idx}" or just "{attention_type}"
            if single tensor. Values are entropy tensors.
    """
    entropy_dict = {}

    if attention_types is None:
        attention_types = list(attns_dict.keys())

    for attn_type in attention_types:
        if attn_type not in attns_dict:
            continue

        attn_data = attns_dict[attn_type]

        if isinstance(attn_data, list):
            # Multiple layers
            for layer_idx, layer_attn in enumerate(attn_data):
                if layer_indices is None or layer_idx in layer_indices:
                    if layer_attn is not None:
                        entropy = compute_attention_entropy(layer_attn, eps=eps)
                        key = f"{attn_type}_layer_{layer_idx}"
                        entropy_dict[key] = entropy
        else:
            # Single tensor
            if attn_data is not None:
                entropy = compute_attention_entropy(attn_data, eps=eps)
                entropy_dict[attn_type] = entropy

    return entropy_dict


def aggregate_attention_entropy(
    entropy_dict: Dict[str, torch.Tensor], aggregation_method: str = "mean"
) -> torch.Tensor:
    """
    Aggregate attention entropy values across different attention types/layers.

    Args:
        entropy_dict (Dict[str, torch.Tensor]): Dictionary of entropy values.
        aggregation_method (str): How to aggregate. Options: "mean", "max", "min".

    Returns:
        torch.Tensor: Aggregated entropy value(s).
    """
    if not entropy_dict:
        return torch.tensor(0.0)

    # Stack all entropy values
    entropy_values = []
    for entropy_tensor in entropy_dict.values():
        if entropy_tensor.numel() > 0:
            # If tensor has multiple dimensions (e.g., per head), take mean
            entropy_values.append(entropy_tensor.mean())

    if not entropy_values:
        return torch.tensor(0.0)

    entropy_stack = torch.stack(entropy_values)

    if aggregation_method == "mean":
        return entropy_stack.mean()
    elif aggregation_method == "max":
        return entropy_stack.max()
    elif aggregation_method == "min":
        return entropy_stack.min()
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")


def compute_batch_attention_entropy(
    attns_dict: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
    attention_types: Optional[List[str]] = None,
    layer_indices: Optional[List[int]] = None,
    aggregation_method: str = "mean",
    eps: float = 1e-8,
) -> float:
    """
    Compute and aggregate attention entropy for a batch.

    This is a convenience function that combines entropy computation and aggregation.

    Args:
        attns_dict (Dict): Dictionary of attention weights from model forward pass.
        attention_types (List[str], optional): Which attention types to compute entropy for.
        layer_indices (List[int], optional): Which layer indices to include.
        aggregation_method (str): How to aggregate entropy values.
        eps (float): Small epsilon value to avoid log(0).

    Returns:
        float: Aggregated attention entropy value for the batch.
    """
    entropy_dict = compute_attention_entropy_from_dict(attns_dict, attention_types, layer_indices, eps)

    aggregated_entropy = aggregate_attention_entropy(entropy_dict, aggregation_method)

    return aggregated_entropy.item()
