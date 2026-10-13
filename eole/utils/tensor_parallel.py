"""Keep generation decisions identical across tensor-parallel ranks."""


def broadcast_from_rank_zero(tensor, parallel_gpu):
    """Share selected tokens/scores, avoiding a full-vocabulary transfer."""
    if parallel_gpu > 1:
        import torch.distributed as dist

        dist.broadcast(tensor, src=0)
    return tensor
