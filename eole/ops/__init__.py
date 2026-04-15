from torch.library import custom_op
from typing import Optional
import torch
import torch.nn.functional as F
import warnings
from eole import EOLE_TORCH_COMPILE

# Try to import the C++ extension
try:
    from eole import _ops

    _CPP_OPS_AVAILABLE = torch.cuda.is_available()
except (ImportError, ModuleNotFoundError) as e:
    _CPP_OPS_AVAILABLE = False
    warnings.warn(
        f"Could not import eole._ops C++ extension: {e}. "
        "Custom CUDA/C++ operations (rms_norm, rotary_embedding, silu_and_mul, "
        "gelu_and_mul, gelu_tanh_and_mul, gptq_marlin_gemm, moe_wna16_marlin_gemm) "
        "will not be available. Make sure the extension is properly compiled.",
        RuntimeWarning,
        stacklevel=2,
    )
try:
    import flash_attn.cute

    _FLASH_ATTN_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
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
            window_size: tuple[int, int] = (-1, -1),  # Accept tuple here
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
            return None

    else:

        def gelu_tanh_and_mul(*args, **kwargs):
            raise RuntimeError("gelu_tanh_and_mul is not available. The eole._ops C++ extension is not loaded.")

    # ============================================================================
    # MoE align block size
    # ============================================================================
    if _CPP_OPS_AVAILABLE:

        @custom_op("eole::moe_align_block_size", mutates_args={})
        def moe_align_block_size(
            topk_ids: torch.Tensor,
            num_experts: int,
            block_size: int,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            num_tokens = topk_ids.numel()
            max_padded = num_tokens + num_experts * (block_size - 1)
            max_blocks = (max_padded + block_size - 1) // block_size
            device = topk_ids.device
            sorted_ids = torch.empty(max_padded, dtype=torch.int32, device=device)
            expert_ids = torch.empty(max_blocks, dtype=torch.int32, device=device)
            ntpp = torch.empty(1, dtype=torch.int32, device=device)
            _ops.moe_align_block_size(
                topk_ids,
                num_experts,
                block_size,
                sorted_ids,
                expert_ids,
                ntpp,
            )
            return sorted_ids, expert_ids, ntpp

        @moe_align_block_size.register_fake
        def _(topk_ids, num_experts, block_size):
            num_tokens = topk_ids.numel()
            max_padded = num_tokens + num_experts * (block_size - 1)
            max_blocks = (max_padded + block_size - 1) // block_size
            device = topk_ids.device
            return (
                torch.empty(max_padded, dtype=torch.int32, device=device),
                torch.empty(max_blocks, dtype=torch.int32, device=device),
                torch.empty(1, dtype=torch.int32, device=device),
            )

    else:

        def moe_align_block_size(*args, **kwargs):
            raise RuntimeError("moe_align_block_size is not available.")

    # ============================================================================
    # GPTQ Marlin Repack
    # ============================================================================
    if _CPP_OPS_AVAILABLE:

        @custom_op("eole::gptq_marlin_repack", mutates_args={})
        def gptq_marlin_repack(
            b_q_weight: torch.Tensor,
            perm: torch.Tensor,
            size_k: int,
            size_n: int,
            num_bits: int,
        ) -> torch.Tensor:
            return _ops.gptq_marlin_repack(b_q_weight, perm, size_k, size_n, num_bits)

        @gptq_marlin_repack.register_fake
        def _(b_q_weight, perm, size_k, size_n, num_bits):
            # Marlin tile layout: rows = size_k // 16, cols = size_n * (num_bits // 2)
            # which is equivalent to the original pack layout; preserve dtype.
            rows = size_k // 16
            cols = size_n * num_bits // 2
            return b_q_weight.new_empty((rows, cols))

    else:

        def gptq_marlin_repack(*args, **kwargs):
            raise RuntimeError("gptq_marlin_repack is not available. The eole._ops C++ extension is not loaded.")

    # ============================================================================
    # GPTQ Marlin GEMM (dense)
    # ============================================================================
    if _CPP_OPS_AVAILABLE:

        @custom_op("eole::gptq_marlin_gemm", mutates_args={"workspace"})
        def gptq_marlin_gemm(
            a: torch.Tensor,
            c: Optional[torch.Tensor],
            b_q_weight: torch.Tensor,
            b_bias: Optional[torch.Tensor],
            b_scales: torch.Tensor,
            global_scale: Optional[torch.Tensor],
            b_zeros: Optional[torch.Tensor],
            g_idx: Optional[torch.Tensor],
            perm: Optional[torch.Tensor],
            workspace: torch.Tensor,
            b_q_type_id: int,
            size_m: int,
            size_n: int,
            size_k: int,
            is_k_full: bool = True,
            use_atomic_add: bool = False,
            use_fp32_reduce: bool = True,
            is_zp_float: bool = False,
        ) -> torch.Tensor:
            return _ops.gptq_marlin_gemm(
                a,
                c,
                b_q_weight,
                b_bias,
                b_scales,
                global_scale,
                b_zeros,
                g_idx,
                perm,
                workspace,
                b_q_type_id,
                size_m,
                size_n,
                size_k,
                is_k_full,
                use_atomic_add,
                use_fp32_reduce,
                is_zp_float,
            )

        @gptq_marlin_gemm.register_fake
        def _(
            a,
            c,
            b_q_weight,
            b_bias,
            b_scales,
            global_scale,
            b_zeros,
            g_idx,
            perm,
            workspace,
            b_q_type_id,
            size_m,
            size_n,
            size_k,
            is_k_full=True,
            use_atomic_add=False,
            use_fp32_reduce=True,
            is_zp_float=False,
        ):
            # Output shape is [size_m, size_n], same dtype as activations.
            return a.new_empty((size_m, size_n))

    else:

        def gptq_marlin_gemm(*args, **kwargs):
            raise RuntimeError("gptq_marlin_gemm is not available. The eole._ops C++ extension is not loaded.")

    # ============================================================================
    # MoE GPTQ Marlin GEMM
    # ============================================================================
    if _CPP_OPS_AVAILABLE:

        @custom_op("eole::moe_wna16_marlin_gemm", mutates_args={"c", "workspace"})
        def moe_wna16_marlin_gemm(
            a: torch.Tensor,
            c: torch.Tensor,
            c_tmp: Optional[torch.Tensor],
            b_q_weight: torch.Tensor,
            b_bias: Optional[torch.Tensor],
            b_scales: torch.Tensor,
            b_act_input: Optional[torch.Tensor],
            b_global_scale: Optional[torch.Tensor],
            b_zeros: Optional[torch.Tensor],
            g_idx: Optional[torch.Tensor],
            perm: Optional[torch.Tensor],
            workspace: torch.Tensor,
            sorted_token_ids: torch.Tensor,
            expert_ids: torch.Tensor,
            num_tokens_post_padded: torch.Tensor,
            topk_weights: torch.Tensor,
            moe_block_size: int,
            top_k: int,
            mul_topk_weights: bool,
            b_q_type_id: int,
            size_m: int,
            size_n: int,
            size_k: int,
            is_k_full: bool = True,
            use_atomic_add: bool = False,
            use_fp32_reduce: bool = True,
            is_zp_float: bool = False,
        ) -> None:
            _ops.moe_wna16_marlin_gemm(
                a,
                c,
                c_tmp,
                b_q_weight,
                b_bias,
                b_scales,
                b_act_input,
                b_global_scale,
                b_zeros,
                g_idx,
                perm,
                workspace,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                topk_weights,
                moe_block_size,
                top_k,
                mul_topk_weights,
                b_q_type_id,
                size_m,
                size_n,
                size_k,
                is_k_full,
                use_atomic_add,
                use_fp32_reduce,
                is_zp_float,
            )

        @moe_wna16_marlin_gemm.register_fake
        def _(
            a,
            c,
            c_tmp,
            b_q_weight,
            b_bias,
            b_scales,
            b_act_input,
            b_global_scale,
            b_zeros,
            g_idx,
            perm,
            workspace,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_weights,
            moe_block_size,
            top_k,
            mul_topk_weights,
            b_q_type_id,
            size_m,
            size_n,
            size_k,
            is_k_full=True,
            use_atomic_add=False,
            use_fp32_reduce=True,
            is_zp_float=False,
        ):
            return None  # void: c is mutated in-place

    else:

        def moe_wna16_marlin_gemm(*args, **kwargs):
            raise RuntimeError(
                "moe_wna16_marlin_gemm is not available: " "eole._ops was not compiled with the Marlin MoE kernel."
            )

    # ============================================================================
    # Wrapped SDPA (transpose may break contiguous needed for torch.compile)
    # ============================================================================

    @custom_op("eole::_sdpa", mutates_args=())
    def _sdpa(
        q: torch.Tensor,  # (b, l, h, d) — transpose happens inside
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        scale: float | None = None,
    ) -> torch.Tensor:

        return F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            scale=scale,
        ).transpose(1, 2)

    @_sdpa.register_fake
    def _(q, k, v, attn_mask=None, dropout_p=0.0, scale=None):
        return torch.empty_like(q)

else:
    # NO EOLE_TORCH_COMPILE — use native CUDA kernels directly for performance.
    if _FLASH_ATTN_AVAILABLE:
        flash_attn_kvcache = flash_attn.flash_attn_with_kvcache
    else:
        flash_attn_kvcache = None

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

    # ── Dense Marlin GEMM ─────────────────────────────────────────────────────
    if _CPP_OPS_AVAILABLE:

        def moe_align_block_size(
            topk_ids: torch.Tensor,
            num_experts: int,
            block_size: int,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            num_tokens = topk_ids.numel()
            max_padded = num_tokens + num_experts * (block_size - 1)
            max_blocks = (max_padded + block_size - 1) // block_size
            device = topk_ids.device
            sorted_ids = torch.empty(max_padded, dtype=torch.int32, device=device)
            expert_ids = torch.empty(max_blocks, dtype=torch.int32, device=device)
            ntpp = torch.empty(1, dtype=torch.int32, device=device)
            _ops.moe_align_block_size(
                topk_ids,
                num_experts,
                block_size,
                sorted_ids,
                expert_ids,
                ntpp,
            )
            return sorted_ids, expert_ids, ntpp

        def gptq_marlin_repack(
            b_q_weight: torch.Tensor,
            perm: torch.Tensor,
            size_k: int,
            size_n: int,
            num_bits: int,
        ) -> torch.Tensor:
            return _ops.gptq_marlin_repack(b_q_weight, perm, size_k, size_n, num_bits)

        def gptq_marlin_gemm(
            a: torch.Tensor,
            c: Optional[torch.Tensor],
            b_q_weight: torch.Tensor,
            b_bias: Optional[torch.Tensor],
            b_scales: torch.Tensor,
            global_scale: Optional[torch.Tensor],
            b_zeros: Optional[torch.Tensor],
            g_idx: Optional[torch.Tensor],
            perm: Optional[torch.Tensor],
            workspace: torch.Tensor,
            b_q_type_id: int,
            size_m: int,
            size_n: int,
            size_k: int,
            is_k_full: bool = True,
            use_atomic_add: bool = False,
            use_fp32_reduce: bool = True,
            is_zp_float: bool = False,
        ) -> torch.Tensor:
            return _ops.gptq_marlin_gemm(
                a,
                c,
                b_q_weight,
                b_bias,
                b_scales,
                global_scale,
                b_zeros,
                g_idx,
                perm,
                workspace,
                b_q_type_id,
                size_m,
                size_n,
                size_k,
                is_k_full,
                use_atomic_add,
                use_fp32_reduce,
                is_zp_float,
            )

        def moe_wna16_marlin_gemm(
            a,
            c,
            c_tmp,
            b_q_weight,
            b_bias,
            b_scales,
            b_act_input,
            b_global_scale,
            b_zeros,
            g_idx,
            perm,
            workspace,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            topk_weights,
            moe_block_size,
            top_k,
            mul_topk_weights,
            b_q_type_id: int,
            size_m,
            size_n,
            size_k,
            is_k_full=True,
            use_atomic_add=False,
            use_fp32_reduce=True,
            is_zp_float=False,
        ) -> None:
            _ops.moe_wna16_marlin_gemm(
                a,
                c,
                c_tmp,
                b_q_weight,
                b_bias,
                b_scales,
                b_act_input,
                b_global_scale,
                b_zeros,
                g_idx,
                perm,
                workspace,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                topk_weights,
                moe_block_size,
                top_k,
                mul_topk_weights,
                b_q_type_id,
                size_m,
                size_n,
                size_k,
                is_k_full,
                use_atomic_add,
                use_fp32_reduce,
                is_zp_float,
            )

    else:

        def moe_align_block_size(*args, **kwargs):
            raise RuntimeError("moe_align_block_size is not available.")

        def gptq_marlin_repack(*args, **kwargs):
            raise RuntimeError("gptq_marlin_repack is not available: eole._ops C++ extension not loaded.")

        def gptq_marlin_gemm(*args, **kwargs):
            raise RuntimeError("gptq_marlin_gemm is not available. The eole._ops C++ extension is not loaded.")

        def moe_wna16_marlin_gemm(*args, **kwargs):
            raise RuntimeError(
                "moe_wna16_marlin_gemm is not available: " "eole._ops was not compiled with the Marlin MoE kernel."
            )

    def _sdpa(q, k, v, attn_mask=None, dropout_p=0.0, scale=None):

        return F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            scale=scale,
        ).transpose(1, 2)


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
    "moe_align_block_size",
    "gptq_marlin_repack",
    "gptq_marlin_gemm",
    "moe_wna16_marlin_gemm",
    "_sdpa",
]
