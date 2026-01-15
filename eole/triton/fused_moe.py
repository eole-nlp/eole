import torch
import triton
import triton.language as tl

# ===========================
# FUSED VERSION (W1 + Activation in one kernel)
# ===========================


@triton.jit
def fused_w1_activation_kernel(
    X_ptr,
    W1_ptr,
    Y_ptr,
    expert_ids_ptr,
    token_ids_ptr,
    stride_xm,
    stride_xk,
    stride_we,
    stride_wn,
    stride_wk,
    stride_ym,
    stride_yn,
    num_pairs: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused W1 matmul + gated SiLU activation.
    """
    pid_pair = tl.program_id(0)
    pid_n = tl.program_id(1)

    if pid_pair >= num_pairs:
        return

    expert_id = tl.load(expert_ids_ptr + pid_pair)
    token_id = tl.load(token_ids_ptr + pid_pair)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < I

    acc_gate = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_N,), dtype=tl.float32)

    x_base = X_ptr + token_id * stride_xm

    for k_start in range(0, H, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < H

        x_ptrs = x_base + (offs_k * stride_xk)
        x = tl.load(x_ptrs, mask=mask_k, other=0.0)

        w_gate_ptrs = W1_ptr + expert_id * stride_we + (offs_n[:, None] * stride_wn) + (offs_k[None, :] * stride_wk)
        w_gate = tl.load(w_gate_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)

        w_up_ptrs = W1_ptr + expert_id * stride_we + ((offs_n[:, None] + I) * stride_wn) + (offs_k[None, :] * stride_wk)
        w_up = tl.load(w_up_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)

        acc_gate += tl.sum(x[None, :] * w_gate, axis=1)
        acc_up += tl.sum(x[None, :] * w_up, axis=1)

    # Gated SiLU: gate * sigmoid(gate) * up
    activated = acc_gate * tl.sigmoid(acc_gate) * acc_up

    y_ptrs = Y_ptr + pid_pair * stride_ym + offs_n * stride_yn
    tl.store(y_ptrs, activated.to(X_ptr.dtype.element_ty), mask=mask_n)


# ===========================
# UNFUSED VERSION (Separate W1 and Activation kernels)
# ===========================


@triton.jit
def w1_kernel(
    X_ptr,
    W1_ptr,
    Y_ptr,
    expert_ids_ptr,
    token_ids_ptr,
    stride_xm,
    stride_xk,
    stride_we,
    stride_wn,
    stride_wk,
    stride_ym,
    stride_yn,
    num_pairs: tl.constexpr,
    H: tl.constexpr,
    I2: tl.constexpr,  # 2*I (gate + up concatenated)
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    W1 matmul: produces concatenated [gate; up] output.
    """
    pid_pair = tl.program_id(0)
    pid_n = tl.program_id(1)

    if pid_pair >= num_pairs:
        return

    expert_id = tl.load(expert_ids_ptr + pid_pair)
    token_id = tl.load(token_ids_ptr + pid_pair)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < I2

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    x_base = X_ptr + token_id * stride_xm

    for k_start in range(0, H, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < H

        x_ptrs = x_base + (offs_k * stride_xk)
        x = tl.load(x_ptrs, mask=mask_k, other=0.0)

        w_ptrs = W1_ptr + expert_id * stride_we + (offs_n[:, None] * stride_wn) + (offs_k[None, :] * stride_wk)
        w = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)

        acc += tl.sum(x[None, :] * w, axis=1)

    y_ptrs = Y_ptr + pid_pair * stride_ym + offs_n * stride_yn
    tl.store(y_ptrs, acc.to(X_ptr.dtype.element_ty), mask=mask_n)


@triton.jit
def gated_activation_kernel(
    X_ptr,  # Input: [num_pairs, 2*I] (gate and up concatenated)
    Y_ptr,  # Output: [num_pairs, I]
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    num_pairs: tl.constexpr,
    I: tl.constexpr,
    BLOCK_N: tl.constexpr,
    activation: tl.constexpr,  # 0=silu, 1=gelu, 2=relu
):
    """
    Gated activation: applies activation(gate) * up
    Supports: silu, gelu, relu
    """
    pid_pair = tl.program_id(0)
    pid_n = tl.program_id(1)

    if pid_pair >= num_pairs:
        return

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < I

    # Load gate values
    gate_ptrs = X_ptr + pid_pair * stride_xm + offs_n * stride_xn
    gate = tl.load(gate_ptrs, mask=mask_n, other=0.0).to(tl.float32)

    # Load up values
    up_ptrs = X_ptr + pid_pair * stride_xm + (offs_n + I) * stride_xn
    up = tl.load(up_ptrs, mask=mask_n, other=0.0).to(tl.float32)

    # Apply activation
    if activation == 0:  # silu
        activated_gate = gate * tl.sigmoid(gate)
    elif activation == 1:  # gelu (approximation)
        # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        sqrt_2_over_pi = 0.7978845608028654
        gate_cubed = gate * gate * gate
        tanh_arg = sqrt_2_over_pi * (gate + 0.044715 * gate_cubed)
        activated_gate = 0.5 * gate * (1.0 + tl.math.tanh(tanh_arg))
    elif activation == 2:  # relu
        activated_gate = tl.where(gate > 0, gate, 0.0)
    else:  # default to silu
        activated_gate = gate * tl.sigmoid(gate)

    result = activated_gate * up

    # Store result
    y_ptrs = Y_ptr + pid_pair * stride_ym + offs_n * stride_yn
    tl.store(y_ptrs, result.to(X_ptr.dtype.element_ty), mask=mask_n)


# ===========================
# W2 KERNEL (shared by both versions)
# ===========================


@triton.jit
def w2_reduce_kernel(
    X_ptr,
    W2_ptr,
    Y_ptr,
    expert_ids_ptr,
    token_ids_ptr,
    weights_ptr,
    stride_xm,
    stride_xk,
    stride_we,
    stride_wn,
    stride_wk,
    stride_ym,
    stride_yn,
    num_pairs: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    W2 matmul + weighted reduce with atomic add.
    """
    pid_pair = tl.program_id(0)
    pid_n = tl.program_id(1)

    if pid_pair >= num_pairs:
        return

    expert_id = tl.load(expert_ids_ptr + pid_pair)
    token_id = tl.load(token_ids_ptr + pid_pair)
    weight = tl.load(weights_ptr + pid_pair)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < H

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    x_base = X_ptr + pid_pair * stride_xm

    for k_start in range(0, I, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < I

        x_ptrs = x_base + (offs_k * stride_xk)
        x = tl.load(x_ptrs, mask=mask_k, other=0.0)

        w_ptrs = W2_ptr + expert_id * stride_we + (offs_n[:, None] * stride_wn) + (offs_k[None, :] * stride_wk)
        w = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)

        acc += tl.sum(x[None, :] * w, axis=1)

    weighted_result = acc * weight
    y_ptrs = Y_ptr + token_id * stride_ym + offs_n * stride_yn
    tl.atomic_add(y_ptrs, weighted_result.to(X_ptr.dtype.element_ty), mask=mask_n)


# ===========================
# PYTHON WRAPPER
# ===========================


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace=True,
    activation="silu",
    use_sorted: bool = False,
    fuse_w1_activation: bool = True,
) -> torch.Tensor:
    """
    MoE implementation with flexible options.

    Args:
        use_sorted: If True, sort tokens by expert (better for large models).
                   If False, process in original order (better for small models).
        fuse_w1_activation: If True, fuse W1+activation in one kernel (faster).
                           If False, separate kernels (more flexible).
        activation: "silu", "gelu", or "relu" (only used when fuse_w1_activation=False)
    """
    M, H = hidden_states.shape
    E, double_I, _ = w1.shape
    I = double_I // 2  # noqaE741
    K = topk_ids.shape[1]
    device = hidden_states.device
    dtype = hidden_states.dtype

    num_pairs = M * K

    # Prepare token-expert pairs
    if use_sorted:
        flat_topk_ids = topk_ids.flatten()
        sorted_indices = torch.argsort(flat_topk_ids, stable=True)
        expert_ids = flat_topk_ids[sorted_indices]
        token_ids = torch.arange(M, device=device, dtype=torch.int32).repeat_interleave(K)[sorted_indices]
        weights = topk_weights.flatten()[sorted_indices]
    else:
        expert_ids = topk_ids.flatten()
        token_ids = torch.arange(M, device=device, dtype=torch.int32).repeat_interleave(K)
        weights = topk_weights.flatten()

    # Launch config
    BLOCK_N, BLOCK_K = 64, 64

    if fuse_w1_activation:
        # ===== FUSED PATH =====
        intermediate = torch.empty((num_pairs, I), device=device, dtype=dtype)
        final_output = torch.zeros((M, H), device=device, dtype=dtype)

        grid_w1 = (num_pairs, triton.cdiv(I, BLOCK_N))
        fused_w1_activation_kernel[grid_w1](
            hidden_states,
            w1,
            intermediate,
            expert_ids,
            token_ids,
            hidden_states.stride(0),
            hidden_states.stride(1),
            w1.stride(0),
            w1.stride(1),
            w1.stride(2),
            intermediate.stride(0),
            intermediate.stride(1),
            num_pairs=num_pairs,
            H=H,
            I=I,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )

    else:
        # ===== UNFUSED PATH =====
        gate_up = torch.empty((num_pairs, double_I), device=device, dtype=dtype)
        intermediate = torch.empty((num_pairs, I), device=device, dtype=dtype)
        final_output = torch.zeros((M, H), device=device, dtype=dtype)

        # W1 kernel
        grid_w1 = (num_pairs, triton.cdiv(double_I, BLOCK_N))
        w1_kernel[grid_w1](
            hidden_states,
            w1,
            gate_up,
            expert_ids,
            token_ids,
            hidden_states.stride(0),
            hidden_states.stride(1),
            w1.stride(0),
            w1.stride(1),
            w1.stride(2),
            gate_up.stride(0),
            gate_up.stride(1),
            num_pairs=num_pairs,
            H=H,
            I2=double_I,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )

        # Activation kernel
        activation_type = {"silu": 0, "gelu": 1, "relu": 2}.get(activation.lower(), 0)
        grid_act = (num_pairs, triton.cdiv(I, BLOCK_N))
        gated_activation_kernel[grid_act](
            gate_up,
            intermediate,
            gate_up.stride(0),
            gate_up.stride(1),
            intermediate.stride(0),
            intermediate.stride(1),
            num_pairs=num_pairs,
            I=I,
            BLOCK_N=BLOCK_N,
            activation=activation_type,
        )

    # W2 kernel (shared by both paths)
    grid_w2 = (num_pairs, triton.cdiv(H, BLOCK_N))
    w2_reduce_kernel[grid_w2](
        intermediate,
        w2,
        final_output,
        expert_ids,
        token_ids,
        weights,
        intermediate.stride(0),
        intermediate.stride(1),
        w2.stride(0),
        w2.stride(1),
        w2.stride(2),
        final_output.stride(0),
        final_output.stride(1),
        num_pairs=num_pairs,
        H=H,
        I=I,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return final_output
