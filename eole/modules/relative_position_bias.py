""" Relative position bias """

import torch
from math import log
from torch import Tensor
from typing import Optional


# Help functions for "Relative" position encoding
def relative_matmul(x: Tensor, z: Tensor, transpose: bool) -> Tensor:
    """
    Helper function for relative positions attention.
    https://arxiv.org/pdf/1803.02155.pdf
    x shape [batch_size x heads x q_len x k_len]
    """
    batch_size, heads, length, _ = x.size()
    x_t = x.permute(2, 0, 1, 3)
    x_t_r = x_t.contiguous().view(length, heads * batch_size, -1)
    if transpose:
        z = z.transpose(1, 2)
    x_tz_matmul = torch.matmul(x_t_r, z)
    x_tz_matmul_r = x_tz_matmul.view(length, batch_size, heads, -1)
    x_tz_matmul_r_t = x_tz_matmul_r.permute(1, 2, 0, 3)
    return x_tz_matmul_r_t


def gen_relative_positions(
    length: int,
    n_positions: int,
    cache: bool = False,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Generate the clipped relative positions matrix
    for a given length and maximum relative positions"""
    if cache:
        distance_mat = torch.arange(-length + 1, 1, 1, device=device).unsqueeze(0)
    else:
        range_vec = torch.arange(length, device=device)
        range_mat = range_vec.unsqueeze(-1).expand(-1, length).transpose(0, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
    distance_mat_clipped = torch.clamp(distance_mat, min=-n_positions, max=n_positions)
    # Shift values to be >= 0
    final_mat = distance_mat_clipped + n_positions
    return final_mat


def _relative_position_bucket(
    relative_position: Tensor, bidirectional: bool = True, num_buckets: int = 32, max_distance: int = 128
) -> Tensor:
    """
    Adapted from Mesh Tensorflow:
    https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/
    mesh_tensorflow/transformer/transformer_layers.py#L593
    Translate relative position to a bucket number for relative attention.
    The relative position is defined as memory_position - query_position,
    i.e. the distance in tokens from the attending position to the attended-to
    position. If bidirectional=False, then positive relative positions are invalid.
    We use smaller buckets for small absolute relative_position and larger buckets for
    larger absolute relative_positions. All relative positions >=max_distance map to the
    same bucket. All relative positions <=-max_distance map to the same bucket.
    This should allow for more graceful generalization to longer sequences than the
    model has been trained on

    Args:
        relative_position: an int32 Tensor
        bidirectional: a boolean - whether the attention is bidirectional
        num_buckets: an integer
        max_distance: an integer

    Returns:
        a Tensor with the same shape as relative_position, containing int32 values
        in the range [0, num_buckets)
    """
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)
    else:
        relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
    # now relative_position is in the range [0, inf)
    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions
    # up to max_distance
    relative_position_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact) / log(max_distance / max_exact) * (num_buckets - max_exact)
    ).to(torch.long)
    relative_position_if_large = torch.min(
        relative_position_if_large,
        torch.full_like(relative_position_if_large, num_buckets - 1),
    )

    relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
    return relative_buckets


def compute_bias(
    query_length: int,
    key_length: int,
    is_decoder: bool,
    n_positions: int,
    relative_positions_buckets: int,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Compute binned relative position bias"""
    context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
    memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
    relative_position = memory_position - context_position  # shape (query_length, key_length)
    relative_position_bucket = _relative_position_bucket(
        relative_position,  # shape (query_length, key_length)
        bidirectional=(not is_decoder),
        num_buckets=relative_positions_buckets,
        max_distance=n_positions,
    )
    return relative_position_bucket
