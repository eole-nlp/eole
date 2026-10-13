"""Checkpoint sharding before GPTQ weights are repacked for inference."""


def slice_gptq_checkpoint(module, name, param, tensor, rank, *, column_parallel=False, row_parallel=False):
    """Slice GPTQ's [packed inputs, outputs] layout, not nn.Linear's axes.

    Column parallelism replicates group indices and slices output columns.
    Row parallelism slices packed inputs and complete quantization groups;
    channel-wise scales are replicated. Group indices become local to the shard.
    """
    if name not in ("qweight", "qzeros", "scales", "g_idx", "bias"):
        return None
    if tensor.shape == param.shape:
        return tensor
    if not column_parallel and not row_parallel:
        return tensor
    if column_parallel:
        if name == "g_idx":
            return tensor
        axis = 0 if name == "bias" else 1
    else:
        if getattr(module, "desc_act", False):
            raise ValueError("Row-parallel GPTQ loading with activation reordering is not supported")
        if name == "bias":
            return tensor
        if name in ("scales", "qzeros") and tensor.size(0) == 1:
            return tensor
        group_size = module.group_size
        input_size = getattr(module, "in_features", getattr(module, "infeatures", None))
        if group_size > 0 and tensor.shape != param.shape and input_size % group_size:
            raise ValueError("Row-parallel GPTQ shards must contain complete quantization groups")
        axis = 0
    width = param.size(axis)
    start = rank * width
    if tensor.size(axis) < start + width:
        raise ValueError(f"Packed checkpoint {name} is too small for tensor-parallel rank {rank}")
    result = tensor.narrow(axis, start, width)
    if row_parallel and name == "g_idx" and tensor.shape != param.shape:
        # The checkpoint references global scale rows. Each rank loaded only
        # its own rows (except channel-wise quantization, which uses group 0).
        result = result - result[0]
    return result
