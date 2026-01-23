from torch.library import custom_op
import torch
import warnings
from eole import EOLE_TORCH_COMPILE

# Try to import the C++ extension
try:
    from eole import _ops

    _CPP_OPS_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    _CPP_OPS_AVAILABLE = False
    warnings.warn(
        f"Could not import eole._ops C++ extension: {e}. "
        "Custom CUDA/C++ operations (rms_norm, rotary_embedding, silu_and_mul, "
        "gelu_and_mul, gelu_tanh_and_mul) will not be available. "
        "Make sure the extension is properly compiled.",
        RuntimeWarning,
        stacklevel=2,
    )
try:
    import flash_attn

    _FLASH_ATTN_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    _FLASH_ATTN_AVAILABLE = False
    warnings.warn(
        f"Could not import flash_attn: {e}. " "Make sure you install flash attention.",
        RuntimeWarning,
        stacklevel=2,
    )


if EOLE_TORCH_COMPILE:
    # We need the register_fake definitions to make torch.compile work
    # ============================================================================
    # Flash Attention KV Cache
    # ============================================================================
    if _FLASH_ATTN_AVAILABLE:

        @custom_op("eole::_flash_attn_kvcache", mutates_args={"k_cache", "v_cache"})
        def _flash_attn_kvcache(
            q: torch.Tensor,
            k_cache: torch.Tensor,
            v_cache: torch.Tensor,
            k: torch.Tensor | None,
            v: torch.Tensor | None,
            rotary_cos: torch.Tensor | None = None,
            rotary_sin: torch.Tensor | None = None,
            cache_seqlens: torch.Tensor | None = None,  # Removed Union with int, Union not supported
            # cache_seqlens: int | None = None,  # Removed Union with int, Union not supported
            cache_batch_idx: torch.Tensor | None = None,
            cache_leftpad: torch.Tensor | None = None,
            block_table: torch.Tensor | None = None,
            softmax_scale: float | None = None,
            causal: bool = False,
            window_size_left: int = -1,  # Changed: split tuple into two ints
            window_size_right: int = -1,  # Changed: split tuple into two ints
            softcap: float = 0.0,
            rotary_interleaved: bool = True,
            alibi_slopes: torch.Tensor | None = None,
            num_splits: int = 0,
            return_softmax_lse: bool = False,
        ) -> torch.Tensor:
            return flash_attn.flash_attn_with_kvcache(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                k=k,
                v=v,
                rotary_cos=rotary_cos,
                rotary_sin=rotary_sin,
                cache_seqlens=cache_seqlens,
                cache_batch_idx=cache_batch_idx,
                cache_leftpad=cache_leftpad,
                block_table=block_table,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=(window_size_left, window_size_right),  # Reconstruct tuple
                softcap=softcap,
                rotary_interleaved=rotary_interleaved,
                alibi_slopes=alibi_slopes,
                num_splits=num_splits,
                return_softmax_lse=return_softmax_lse,
            )

        @_flash_attn_kvcache.register_fake
        def _(
            q,
            k_cache,
            v_cache,
            k,
            v,
            rotary_cos=None,
            rotary_sin=None,
            cache_seqlens=None,
            cache_batch_idx=None,
            cache_leftpad=None,
            block_table=None,
            softmax_scale=None,
            causal=False,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            rotary_interleaved=True,
            alibi_slopes=None,
            num_splits=0,
            return_softmax_lse=False,
        ):
            if return_softmax_lse:
                lse = torch.empty(q.shape[0], q.shape[1], q.shape[2], device=q.device, dtype=torch.float32)
                return (torch.empty_like(q), lse)
            return torch.empty_like(q)

        # restore signature with tuple to match original code
        def flash_attn_kvcache(
            q: torch.Tensor,
            k_cache: torch.Tensor,
            v_cache: torch.Tensor,
            k: torch.Tensor | None = None,
            v: torch.Tensor | None = None,
            rotary_cos: torch.Tensor | None = None,
            rotary_sin: torch.Tensor | None = None,
            cache_seqlens: torch.Tensor | None = None,
            cache_batch_idx: torch.Tensor | None = None,
            cache_leftpad: torch.Tensor | None = None,
            block_table: torch.Tensor | None = None,
            softmax_scale: float | None = None,
            causal: bool = False,
            window_size: tuple[int, int] = (-1, -1),  # Accept tuple here!
            softcap: float = 0.0,
            rotary_interleaved: bool = True,
            alibi_slopes: torch.Tensor | None = None,
            num_splits: int = 0,
            return_softmax_lse: bool = False,
        ) -> torch.Tensor:
            """
            Flash attention with KV cache.

            Args:
                window_size: Tuple of (left, right) window sizes.
            """
            # Split tuple into separate args for custom_op
            window_size_left, window_size_right = window_size

            return _flash_attn_kvcache(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                k=k,
                v=v,
                rotary_cos=rotary_cos,
                rotary_sin=rotary_sin,
                cache_seqlens=cache_seqlens,
                cache_batch_idx=cache_batch_idx,
                cache_leftpad=cache_leftpad,
                block_table=block_table,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size_left=window_size_left,
                window_size_right=window_size_right,
                softcap=softcap,
                rotary_interleaved=rotary_interleaved,
                alibi_slopes=alibi_slopes,
                num_splits=num_splits,
                return_softmax_lse=return_softmax_lse,
            )

    else:

        def flash_attn_kvcache(*args, **kwargs):
            raise RuntimeError("flash_attn_kvcache is not available.")

    # ============================================================================
    # RMS Norm
    # ============================================================================
    if _CPP_OPS_AVAILABLE:

        @custom_op("eole::rms_norm", mutates_args={"out"})
        def rms_norm(
            out: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            epsilon: float,
            gemma: bool = False,
        ) -> None:
            _ops.rms_norm(out, input, weight, epsilon, gemma)

        @rms_norm.register_fake
        def _(out, input, weight, epsilon, gemma=False):
            # Mutates out in-place, returns None
            return None

    else:

        def rms_norm(*args, **kwargs):
            raise RuntimeError("rms_norm is not available. The eole._ops C++ extension is not loaded.")

    # ============================================================================
    # Rotary Embedding
    # ============================================================================
    if _CPP_OPS_AVAILABLE:

        @custom_op("eole::rotary_embedding", mutates_args={"query", "key"})
        def rotary_embedding(
            positions: torch.Tensor,
            query: torch.Tensor,
            key: torch.Tensor,
            head_size: int,
            cos_sin_cache: torch.Tensor,
            is_neox: bool = True,
        ) -> None:
            _ops.rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox)

        @rotary_embedding.register_fake
        def _(positions, query, key, head_size, cos_sin_cache, is_neox=True):
            # Mutates query and key in-place, returns None
            return None

    else:

        def rotary_embedding(*args, **kwargs):
            raise RuntimeError("rotary_embedding is not available. The eole._ops C++ extension is not loaded.")

    # ============================================================================
    # SiLU and Mul
    # ============================================================================
    if _CPP_OPS_AVAILABLE:

        @custom_op("eole::silu_and_mul", mutates_args={"out"})
        def silu_and_mul(
            out: torch.Tensor,
            input: torch.Tensor,
        ) -> None:
            _ops.silu_and_mul(out, input)

        @silu_and_mul.register_fake
        def _(out, input):
            # Mutates out in-place, returns None
            return None

    else:

        def silu_and_mul(*args, **kwargs):
            raise RuntimeError("silu_and_mul is not available. The eole._ops C++ extension is not loaded.")

    # ============================================================================
    # GELU and Mul
    # ============================================================================
    if _CPP_OPS_AVAILABLE:

        @custom_op("eole::gelu_and_mul", mutates_args={"out"})
        def gelu_and_mul(
            out: torch.Tensor,
            input: torch.Tensor,
        ) -> None:
            _ops.gelu_and_mul(out, input)

        @gelu_and_mul.register_fake
        def _(out, input):
            # Mutates out in-place, returns None
            return None

    else:

        def gelu_and_mul(*args, **kwargs):
            raise RuntimeError("gelu_and_mul is not available. The eole._ops C++ extension is not loaded.")

    # ============================================================================
    # GELU-Tanh and Mul
    # ============================================================================
    if _CPP_OPS_AVAILABLE:

        @custom_op("eole::gelu_tanh_and_mul", mutates_args={"out"})
        def gelu_tanh_and_mul(
            out: torch.Tensor,
            input: torch.Tensor,
        ) -> None:
            _ops.gelu_tanh_and_mul(out, input)

        @gelu_tanh_and_mul.register_fake
        def _(out, input):
            # Mutates out in-place, returns None
            return None

    else:

        def gelu_tanh_and_mul(*args, **kwargs):
            raise RuntimeError("gelu_tanh_and_mul is not available. The eole._ops C++ extension is not loaded.")

else:
    # NO EOLE_TORCH_COMPILE, we take the native cuda kernel without wrappers for performance
    if _FLASH_ATTN_AVAILABLE:
        flash_attn_kvcache = flash_attn.flash_attn_with_kvcache
    else:
        raise RuntimeError("flash_attn is not available. Install it before using flash attention.")

    # Export C++ ops directly without wrapping
    if _CPP_OPS_AVAILABLE:
        rms_norm = _ops.rms_norm
        rotary_embedding = _ops.rotary_embedding
        silu_and_mul = _ops.silu_and_mul
        gelu_and_mul = _ops.gelu_and_mul
        gelu_tanh_and_mul = _ops.gelu_tanh_and_mul
    else:

        def _unavailable_op(name):
            def wrapper(*args, **kwargs):
                raise RuntimeError(f"{name} is not available. The eole._ops C++ extension is not loaded.")

            return wrapper

        rms_norm = _unavailable_op("rms_norm")
        rotary_embedding = _unavailable_op("rotary_embedding")
        silu_and_mul = _unavailable_op("silu_and_mul")
        gelu_and_mul = _unavailable_op("gelu_and_mul")
        gelu_tanh_and_mul = _unavailable_op("gelu_tanh_and_mul")

# ============================================================================
# Exports
# ============================================================================
__all__ = [
    "flash_attn_kvcache",
    "rms_norm",
    "rotary_embedding",
    "silu_and_mul",
    "gelu_and_mul",
    "gelu_tanh_and_mul",
]
