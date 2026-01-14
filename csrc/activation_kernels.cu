#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>

namespace eole {

// LDG (Load Global) hint for better caching
#define EOLE_LDG(arg) __ldg(arg)

// Check if pointer is 16-byte aligned for int4 vectorized access
__device__ __forceinline__ bool is_16byte_aligned(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 15) == 0;
}

// ============================================================================
// Activation Functions
// ============================================================================

template <typename T>
__device__ __forceinline__ T silu_kernel(const T& x) {
  // x * sigmoid(x)
  return (T)(((float)x) / (1.0f + expf((float)-x)));
}

template <typename T>
__device__ __forceinline__ T gelu_kernel(const T& x) {
  // Equivalent to PyTorch GELU with 'none' approximation
  const float f = (float)x;
  constexpr float ALPHA = M_SQRT1_2;
  return (T)(f * 0.5f * (1.0f + ::erf(f * ALPHA)));
}

template <typename T>
__device__ __forceinline__ T gelu_tanh_kernel(const T& x) {
  // Equivalent to PyTorch GELU with 'tanh' approximation
  const float f = (float)x;
  constexpr float BETA = M_SQRT2 * M_2_SQRTPI * 0.5f;
  constexpr float KAPPA = 0.044715f;
  float x_cube = f * f * f;
  float inner = BETA * (f + KAPPA * x_cube);
  return (T)(0.5f * f * (1.0f + ::tanhf(inner)));
}

// ============================================================================
// Generic Activation and Gating Kernel
// ============================================================================

// Template for computing activation and gating
// ACT_FIRST: if true, apply activation to first half (gate), else to second half (up)
template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&), bool ACT_FIRST>
__device__ __forceinline__ scalar_t compute_gate(const scalar_t& x, const scalar_t& y) {
  return ACT_FIRST ? ACT_FN(x) * y : x * ACT_FN(y);
}

// Main activation and gating kernel
template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&), bool ACT_FIRST>
__global__ void act_and_mul_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2*d]
    const int d) {
  
  constexpr int VEC_SIZE = 16 / sizeof(scalar_t);
  const int64_t token_idx = blockIdx.x;
  const scalar_t* x_ptr = input + token_idx * 2 * d;
  const scalar_t* y_ptr = x_ptr + d;
  scalar_t* out_ptr = out + token_idx * d;

  // Check alignment for 128-bit vectorized access
  const bool aligned = is_16byte_aligned(x_ptr) && 
                       is_16byte_aligned(y_ptr) &&
                       is_16byte_aligned(out_ptr);

  if (aligned && d >= VEC_SIZE) {
    // Fast path: 128-bit vectorized loop
    const int4* x_vec = reinterpret_cast<const int4*>(x_ptr);
    const int4* y_vec = reinterpret_cast<const int4*>(y_ptr);
    int4* out_vec = reinterpret_cast<int4*>(out_ptr);
    const int num_vecs = d / VEC_SIZE;
    const int vec_end = num_vecs * VEC_SIZE;

    for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
      int4 x = EOLE_LDG(&x_vec[i]);
      int4 y = EOLE_LDG(&y_vec[i]);
      int4 r;
      
      auto* xp = reinterpret_cast<scalar_t*>(&x);
      auto* yp = reinterpret_cast<scalar_t*>(&y);
      auto* rp = reinterpret_cast<scalar_t*>(&r);
      
#pragma unroll
      for (int j = 0; j < VEC_SIZE; j++) {
        rp[j] = compute_gate<scalar_t, ACT_FN, ACT_FIRST>(xp[j], yp[j]);
      }
      out_vec[i] = r;
    }
    
    // Scalar cleanup for remaining elements
    for (int i = vec_end + threadIdx.x; i < d; i += blockDim.x) {
      out_ptr[i] = compute_gate<scalar_t, ACT_FN, ACT_FIRST>(
          EOLE_LDG(&x_ptr[i]), 
          EOLE_LDG(&y_ptr[i])
      );
    }
  } else {
    // Scalar fallback for unaligned data or small d
    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
      const scalar_t x = EOLE_LDG(&x_ptr[idx]);
      const scalar_t y = EOLE_LDG(&y_ptr[idx]);
      out_ptr[idx] = compute_gate<scalar_t, ACT_FN, ACT_FIRST>(x, y);
    }
  }
}

// Template launcher
template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&), bool ACT_FIRST>
void launch_act_and_mul_kernel(
    scalar_t* out,
    const scalar_t* input,
    int d,
    int64_t num_tokens,
    cudaStream_t stream) {
  
  dim3 grid(num_tokens);
  dim3 block(std::min(d, 1024));
  
  act_and_mul_kernel<scalar_t, ACT_FN, ACT_FIRST><<<grid, block, 0, stream>>>(
      out, input, d
  );
}

} // namespace eole

// ============================================================================
// Python Binding Functions
// ============================================================================

void silu_and_mul(torch::Tensor& out, torch::Tensor& input) {
  TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
  TORCH_CHECK(out.is_cuda(), "Output must be a CUDA tensor");
  TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "Output must be contiguous");
  
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  
  if (num_tokens == 0) {
    return;
  }
  
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      input.scalar_type(), "silu_and_mul", [&] {
    eole::launch_act_and_mul_kernel<scalar_t, eole::silu_kernel<scalar_t>, true>(
        out.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        d,
        num_tokens,
        stream
    );
  });
}

void gelu_and_mul(torch::Tensor& out, torch::Tensor& input) {
  TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
  TORCH_CHECK(out.is_cuda(), "Output must be a CUDA tensor");
  TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "Output must be contiguous");
  
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  
  if (num_tokens == 0) {
    return;
  }
  
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      input.scalar_type(), "gelu_and_mul", [&] {
    eole::launch_act_and_mul_kernel<scalar_t, eole::gelu_kernel<scalar_t>, true>(
        out.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        d,
        num_tokens,
        stream
    );
  });
}

void gelu_tanh_and_mul(torch::Tensor& out, torch::Tensor& input) {
  TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
  TORCH_CHECK(out.is_cuda(), "Output must be a CUDA tensor");
  TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "Output must be contiguous");
  
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  
  if (num_tokens == 0) {
    return;
  }
  
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      input.scalar_type(), "gelu_tanh_and_mul", [&] {
    eole::launch_act_and_mul_kernel<scalar_t, eole::gelu_tanh_kernel<scalar_t>, true>(
        out.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        d,
        num_tokens,
        stream
    );
  });
}
