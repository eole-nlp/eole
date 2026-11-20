"""Direct commit of short GDN prefixes, with an optional Triton implementation."""

import importlib.util

import torch


def commit_speculation_states(layers, n_kept):
    """Return False for unsupported layouts without changing any state.

    Only partial, batch-one BF16 four-token verification is supported. The
    caller keeps its existing full-chunk copy and general replay fallbacks.
    Launch failures propagate: retrying after an in-place update is unsafe.
    """
    if not layers or not 1 <= n_kept < 4 or torch.version.cuda is None or torch.is_grad_enabled():
        return False
    state = layers[0].recurrent_state
    if not isinstance(state, torch.Tensor) or not state.is_cuda or state.dtype != torch.bfloat16:
        return False
    if torch.cuda.get_device_capability(state.device)[0] < 8 or importlib.util.find_spec("triton") is None:
        return False
    from eole.modules.gated_delta_net import fused_recurrent_gated_delta_rule

    if fused_recurrent_gated_delta_rule is None:
        return False
    if state.ndim != 4 or state.shape[0] != 1 or state.shape[2:] != (128, 128):
        return False
    heads = state.shape[1]
    conv = layers[0].conv_state
    if not isinstance(conv, torch.Tensor) or conv.ndim != 3 or conv.shape[0] != 1:
        return False
    channels, kernel = conv.shape[1:]
    if heads <= 0 or channels <= 0 or kernel <= 0:
        return False
    tensors = []
    layouts = (
        ((1, 4, heads, 128), torch.bfloat16),  # key
        ((1, 4, heads, 128), torch.bfloat16),  # value
        ((1, 4, heads), torch.float32),  # decay
        ((1, 4, heads), torch.bfloat16),  # beta
        ((1, heads, 128, 128), torch.bfloat16),
        ((1, channels, kernel + 4), torch.bfloat16),
        ((1, channels, kernel), torch.bfloat16),
    )
    for layer in layers:
        spec = layer._spec_buffers
        if spec is None or layer._recurrent_gated_delta_rule is not fused_recurrent_gated_delta_rule:
            return False
        if layer.conv_kernel_size != kernel or layer.conv_dim != channels:
            return False
        values = [spec.get(name) for name in ("key", "value", "g", "beta")]
        values += [layer.recurrent_state, spec.get("conv_extended"), layer.conv_state]
        for tensor, (shape, dtype) in zip(values, layouts):
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.shape != shape
                or tensor.dtype != dtype
                or tensor.device != state.device
                or not tensor.is_contiguous()
            ):
                return False
        tensors.extend(values)
    stream = torch.cuda.current_stream(state.device)
    signature = (
        state.device,
        stream.cuda_stream,
        heads,
        channels,
        kernel,
        tuple((id(tensor), tensor.data_ptr()) for tensor in tensors),
    )
    owner = layers[0]
    cache = getattr(owner, "_spec_commit_cache", None)
    if cache is None or cache[0] != signature:
        # Layouts are checked on every call. Once geometry and addresses are
        # unchanged, the cached disjoint-storage check remains valid too.
        if any(pointer % 64 for _, pointer in signature[-1]):
            return False
        ranges = sorted(
            (pointer, pointer + tensor.numel() * tensor.element_size())
            for (_, pointer), tensor in zip(signature[-1], tensors)
        )
        if any(right[0] < left[1] for left, right in zip(ranges, ranges[1:])):
            return False
        # Pointer metadata is created outside capture. A new request, changed
        # buffer, layer order or stream gets a new table. Strong references
        # prevent allocator reuse; record_stream protects outstanding work
        # when a table is replaced on another stream.
        with torch.cuda.device(state.device):
            if torch.cuda.is_current_stream_capturing():
                return False
        for tensor in tensors:
            tensor.record_stream(stream)
        pointers = torch.tensor([tensor.data_ptr() for tensor in tensors], device=state.device, dtype=torch.int64)
        cache = (signature, pointers, tuple(tensors))
        owner._spec_commit_cache = cache

    from eole.triton.gdn_state import commit_states

    commit_states(cache[1], len(layers), n_kept, heads, channels, kernel)
    return True
