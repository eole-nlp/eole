"""GGUF quantized linear layer.

Stores weights in their native GGUF quantized format (uint8 packed blocks) and
dequantizes them on-the-fly during the forward pass via the vLLM fused op.

The quantization type is persisted as a model buffer (``gguf_qtype``, int32)
and stored alongside the quantized weight buffer (``weight``, uint8) in the
safetensors shard.  Both buffers are loaded automatically by
:meth:`~eole.models.model.BaseModel.load_safe_state_dict`.

vLLM fused op
-------------
Inference uses ``torch.ops.vllm._fused_mul_mat_gguf`` registered by
``vllm/model_executor/layers/quantization/gguf.py``.  It handles all GGUF
quantisation types — including Q4_K/M, Q5_K, Q6_K, Q3_K, Q2_K, Q8_0 and all
IQ-family types — without any CPU round-trip:

* ``ggml_mul_mat_vec_a8`` (MMVQ) for small batch sizes.
* ``ggml_mul_mat_a8`` (MMQ) for larger batches on standard + K-quant types.
* ``ggml_dequantize`` + ``x @ W.T`` for IQ types with large batches.

vLLM must be installed; an :class:`ImportError` is raised at forward time if
it is not available.
"""

import torch
import torch.nn as nn


# Float-typed GGUF types that should be loaded as regular tensors (not as
# uint8 quantized blocks).  Models that use these for linear weights don't
# need GGUFLinear – they are handled by normal nn.Linear loading.
_GGUF_FLOAT_TYPES: set[int] = set()

try:
    from gguf import GGMLQuantizationType

    _GGUF_FLOAT_TYPES = {
        GGMLQuantizationType.F32.value,
        GGMLQuantizationType.F16.value,
        GGMLQuantizationType.BF16.value,
        GGMLQuantizationType.F64.value,
    }
except ImportError:
    pass

# ---------------------------------------------------------------------------
# vLLM GGUF support via ``torch.ops.vllm._fused_mul_mat_gguf``.
#
# vLLM registers a single fused op that encapsulates the full MMVQ/MMQ/
# dequantize dispatch logic used by vLLM's own GGUF linear layers:
#   vllm/model_executor/layers/quantization/gguf.py::_fused_mul_mat_gguf
#
# Importing that module triggers the op registration and provides a direct
# callable handle.  The op handles all quant types and preserves the input
# activation dtype in the output.
# ---------------------------------------------------------------------------

_vllm_fused_mul_mat_gguf = None

try:
    from vllm.model_executor.layers.quantization.gguf import (
        fused_mul_mat_gguf as _vllm_fused_mul_mat_gguf,
    )
except Exception:
    pass


class GGUFLinear(nn.Module):
    """Linear layer whose weights are stored in GGUF quantized format.

    Inference is performed via the vLLM fused GGUF op, which must be installed.

    Parameters
    ----------
    in_features, out_features:
        Logical (float) dimensions of the linear transformation.
    bias:
        Whether the layer has a bias term.
    qtype_val:
        Integer value of the :class:`gguf.GGMLQuantizationType` enum for the
        stored weights.  Used to pre-populate the ``gguf_qtype`` buffer so that
        :meth:`forward` works correctly before any checkpoint is loaded.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        qtype_val: int = 0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Quantized weight data – filled by load_safe_state_dict.
        # Initialised to a 1-byte placeholder; the actual size depends on the
        # quantisation type and will be set when the shard is loaded.
        self.register_buffer("weight", torch.zeros(1, dtype=torch.uint8))

        # Quantization type ID (GGMLQuantizationType int value).
        # Stored as a buffer so it is persisted in the safetensors shard.
        self.register_buffer("gguf_qtype", torch.tensor([qtype_val], dtype=torch.int32))

        # Plain Python attribute — visible to dynamo as a constant.
        # Kept in sync with the buffer by _sync_qtype_attr().
        self._qtype_val: int = qtype_val
        if bias:
            self.register_parameter("bias", nn.Parameter(torch.zeros(out_features)))
        else:
            self.bias = None

    def _sync_qtype_attr(self) -> None:
        """Call after load_state_dict to sync the Python attr from the buffer."""
        self._qtype_val = int(self.gguf_qtype.item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dequantize weight on-the-fly via the vLLM fused op and apply linear.

        Requires vLLM to be installed. Raises :class:`ImportError` if it is not.

        Parameters
        ----------
        x : torch.Tensor
            Input activations of shape ``(*, in_features)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(*, out_features)``, same dtype as *x*.
        """
        if _vllm_fused_mul_mat_gguf is None:
            raise ImportError("vLLM is required for GGUF quantized inference. " "Install it with: pip install vllm")

        try:
            from gguf import GGML_QUANT_SIZES, GGMLQuantizationType
        except ImportError:
            raise ImportError("Install 'gguf' to run GGUF quantized inference: pip install gguf")

        block_size, type_size = GGML_QUANT_SIZES[GGMLQuantizationType(self._qtype_val)]
        bytes_per_row = self.in_features // block_size * type_size
        weight_2d = self.weight.reshape(self.out_features, bytes_per_row).contiguous()

        orig_shape = x.shape
        N = x.numel() // self.in_features
        x_2d = x.contiguous().reshape(N, self.in_features)

        if N == 0:
            return torch.empty(*orig_shape[:-1], self.out_features, dtype=x.dtype, device=x.device)

        y = _vllm_fused_mul_mat_gguf(x_2d, weight_2d, self._qtype_val)
        out = y.reshape(*orig_shape[:-1], self.out_features)

        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        qtype_name = "unknown"
        try:
            from gguf import GGMLQuantizationType

            qtype_name = GGMLQuantizationType(self._qtype_val).name
        except Exception:
            pass
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, qtype={qtype_name}"
        )


def replace_gguf_linear(
    model: nn.Module,
    module_to_convert: list[str],
) -> nn.Module:
    """Replace :class:`~torch.nn.Linear` layers with :class:`GGUFLinear`.

    Walks the model recursively and replaces any ``nn.Linear`` module whose
    name appears in *module_to_convert* with a ``GGUFLinear`` placeholder.
    The quantized weight data and ``gguf_qtype`` are populated later by
    :meth:`~eole.models.model.BaseModel.load_safe_state_dict`.

    Parameters
    ----------
    model:
        Root module to patch in-place.
    module_to_convert:
        List of *local* module names (``module.name``) to replace.
    """
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_gguf_linear(module, module_to_convert)

        if isinstance(module, nn.Linear) and name in module_to_convert:
            model._modules[name] = GGUFLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                # qtype_val=0 is a placeholder; the real value is loaded from
                # the safetensors shard via the gguf_qtype buffer.
                qtype_val=0,
            )
    return model
