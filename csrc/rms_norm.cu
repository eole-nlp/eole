#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cub/cub.cuh>
#include <numeric>

namespace eole {
// ------------------------------------------------------------------------------
// Code taken from the vLLM kernel further modified to support Gemm-style RMSNorm 
// ------------------------------------------------------------------------------



// ------------------------------------------------
// Vector type utility for vectorized memory access
// ------------------------------------------------
template <typename T, int N>
struct vec_n_t {
  T val[N];  // fixed-size array for N elements
};

// ---------------------------------------------
// CUB addition operator for block reduction
// ---------------------------------------------
struct CubAddOp {
  template <typename T>
  __device__ __forceinline__ T operator()(const T& a, const T& b) const {
    return a + b;
  }
};

// ---------------------------------------------
// Vectorized read helper with remainder handling
// This allows efficient memory access with vectorized loads
// and handles leftover elements that don't fit the vector size.
// ---------------------------------------------
template <int VEC_SIZE, typename scalar_t, typename VecOp, typename ScalarOp>
__device__ void vectorize_read_with_alignment(
    const scalar_t* input, int size, int tid, int nthreads,
    VecOp vec_op, ScalarOp scalar_op) {
  
  using VecT = vec_n_t<scalar_t, VEC_SIZE>;
  const int vec_size = size / VEC_SIZE;
  const int remainder = size % VEC_SIZE;
  
  // Process main vectorized portion
  const VecT* vec_input = reinterpret_cast<const VecT*>(input);
  for (int i = tid; i < vec_size; i += nthreads) {
    VecT vec = vec_input[i];
    vec_op(vec);
  }
  
  // Handle remaining elements
  for (int i = vec_size * VEC_SIZE + tid; i < size; i += nthreads) {
    scalar_op(input[i]);
  }
}

// ---------------------------------------------
// RMSNorm CUDA kernel
// Applies root-mean-square normalization on the last dimension
// Supports 2D, 3D, and 4D inputs for standard and transformer-style tensors.
// Optional GEMMA-style scaling: multiply by (1 + weight) instead of weight only.
// ---------------------------------------------
template <typename scalar_t, int VEC_SIZE, int NUM_DIMS, bool GEMMA_STYLE>
__global__ void rms_norm_kernel(
    scalar_t* __restrict__ out,           // output tensor [..., hidden_size]
    const scalar_t* __restrict__ input,   // input tensor [..., hidden_size]
    const int64_t input_stride_d2,        // stride of last dim (-2)
    const int64_t input_stride_d3,        // stride of -3 dimension
    const int64_t input_stride_d4,        // stride of -4 dimension
    const int64_t input_shape_d2,         // size of -2 dimension
    const int64_t input_shape_d3,         // size of -3 dimension
    const scalar_t* __restrict__ weight,  // learnable weight [hidden_size]
    const float epsilon,                  // small number for numerical stability
    const int num_tokens,                 // total number of rows to process
    const int hidden_size) {              // last-dimension size
  
  __shared__ float s_variance;           // shared variance for block
  float variance = 0.0f;
  const scalar_t* input_row;

  // -----------------------------
  // Determine pointer to current row based on NUM_DIMS
  // -----------------------------
  if constexpr (NUM_DIMS == 2) {
    // Standard [batch, hidden]
    input_row = input + blockIdx.x * input_stride_d2;
  } else if constexpr (NUM_DIMS == 3) {
    // [batch, num_heads, head_size]
    int batch_idx = blockIdx.x / input_shape_d2;
    int head_idx = blockIdx.x % input_shape_d2;
    input_row = input + batch_idx * input_stride_d3 + head_idx * input_stride_d2;
  } else if constexpr (NUM_DIMS == 4) {
    // [batch, seq, head, head_dim]
    int batch_idx = blockIdx.x / (input_shape_d3 * input_shape_d2);
    int remaining = blockIdx.x % (input_shape_d3 * input_shape_d2);
    int seq_idx = remaining / input_shape_d2;
    int head_idx = remaining % input_shape_d2;
    input_row = input + batch_idx * input_stride_d4 +
                seq_idx * input_stride_d3 + head_idx * input_stride_d2;
  }

  // -----------------------------
  // Compute variance in a vectorized manner
  // -----------------------------
  auto vec_op = [&variance](const vec_n_t<scalar_t, VEC_SIZE>& vec) {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      float x = static_cast<float>(vec.val[i]);
      variance += x * x;
    }
  };

  auto scalar_op = [&variance](const scalar_t& val) {
    float x = static_cast<float>(val);
    variance += x * x;
  };

  // Vectorized read over hidden dimension
  vectorize_read_with_alignment<VEC_SIZE>(
      input_row, hidden_size, threadIdx.x, blockDim.x, vec_op, scalar_op);

  // -----------------------------
  // Reduce variance across threads in the block using CUB
  // -----------------------------
  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  // Compute inverse root-mean-square
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  // -----------------------------
  // Normalize input and apply weight
  // -----------------------------
  scalar_t* out_row = out + blockIdx.x * hidden_size;
  auto* v_in = reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE>*>(input_row);
  auto* v_w = reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE>*>(weight);
  auto* v_out = reinterpret_cast<vec_n_t<scalar_t, VEC_SIZE>*>(out_row);

  for (int i = threadIdx.x; i < hidden_size / VEC_SIZE; i += blockDim.x) {
    vec_n_t<scalar_t, VEC_SIZE> dst;
    vec_n_t<scalar_t, VEC_SIZE> src1 = v_in[i];
    vec_n_t<scalar_t, VEC_SIZE> src2 = v_w[i];
#pragma unroll
    for (int j = 0; j < VEC_SIZE; j++) {
      float x = static_cast<float>(src1.val[j]);
      float w = static_cast<float>(src2.val[j]);
      if constexpr (GEMMA_STYLE) {
        w = w + 1.0f;  // Gemma-style scaling
      }
      dst.val[j] = ((scalar_t)(x * s_variance)) * ((scalar_t)w);
    }
    v_out[i] = dst;
  }
}

// ---------------------------------------------
// Host wrapper to launch RMSNorm kernel
// Computes block/grid dimensions and dispatches kernel
// ---------------------------------------------
template <typename scalar_t, int VEC_SIZE, int NUM_DIMS, bool GEMMA_STYLE>
void launch_rms_norm_kernel(
    scalar_t* out, const scalar_t* input,
    int64_t input_stride_d2, int64_t input_stride_d3, int64_t input_stride_d4,
    int64_t input_shape_d2, int64_t input_shape_d3,
    const scalar_t* weight, float epsilon, int num_tokens, int hidden_size,
    cudaStream_t stream) {

  // Determine reasonable block size
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;
  const int calculated_vec_size = std::gcd(16 / sizeof(scalar_t), hidden_size);
  const int block_size = std::min(hidden_size / calculated_vec_size, max_block_size);

  dim3 grid(num_tokens);
  dim3 block(block_size);

  rms_norm_kernel<scalar_t, VEC_SIZE, NUM_DIMS, GEMMA_STYLE>
      <<<grid, block, 0, stream>>>(
          out, input, input_stride_d2, input_stride_d3, input_stride_d4,
          input_shape_d2, input_shape_d3, weight, epsilon, num_tokens, hidden_size);
}

} // namespace eole

// ---------------------------------------------
// Helper to compute best vectorization size
// ---------------------------------------------
static inline int compute_vec_size(at::ScalarType dtype, int hidden_size) {
  int vec_size;
  if (dtype == at::ScalarType::Half || dtype == at::ScalarType::BFloat16) {
    vec_size = std::min(8, hidden_size);
  } else {
    vec_size = std::min(4, hidden_size);
  }
  while (vec_size > 1 && hidden_size % vec_size != 0) {
    vec_size /= 2;
  }
  return vec_size;
}

// ---------------------------------------------
// Public API: RMSNorm entry point
// Handles tensor preparation and kernel dispatch
// ---------------------------------------------
void rms_norm(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight, 
              double epsilon, bool gemma) {

  TORCH_CHECK(out.is_contiguous());
  if (input.stride(-1) != 1) input = input.contiguous();
  TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");

  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  int num_dims = input.dim();

  int64_t input_stride_d2 = input.stride(-2);
  int64_t input_stride_d3 = (num_dims >= 3) ? input.stride(-3) : 0;
  int64_t input_stride_d4 = (num_dims >= 4) ? input.stride(-4) : 0;
  int64_t input_shape_d2 = (num_dims >= 3) ? input.size(-2) : 0;
  int64_t input_shape_d3 = (num_dims >= 4) ? input.size(-3) : 0;

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  int vec_size = compute_vec_size(input.scalar_type(), hidden_size);

  #define LAUNCH_KERNEL(scalar_t, vec_size, num_dims) \
    if (gemma) { \
      eole::launch_rms_norm_kernel<scalar_t, vec_size, num_dims, true>( \
          out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), \
          input_stride_d2, input_stride_d3, input_stride_d4, \
          input_shape_d2, input_shape_d3, weight.data_ptr<scalar_t>(), \
          epsilon, num_tokens, hidden_size, stream); \
    } else { \
      eole::launch_rms_norm_kernel<scalar_t, vec_size, num_dims, false>( \
          out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), \
          input_stride_d2, input_stride_d3, input_stride_d4, \
          input_shape_d2, input_shape_d3, weight.data_ptr<scalar_t>(), \
          epsilon, num_tokens, hidden_size, stream); \
    }

  // Dispatch kernel based on number of input dimensions
  if (num_dims == 2) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "rms_norm_kernel", [&] {
      if (vec_size == 8) { LAUNCH_KERNEL(scalar_t, 8, 2); }
      else if (vec_size == 4) { LAUNCH_KERNEL(scalar_t, 4, 2); }
      else if (vec_size == 2) { LAUNCH_KERNEL(scalar_t, 2, 2); }
      else { LAUNCH_KERNEL(scalar_t, 1, 2); }
    });
  } else if (num_dims == 3) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "rms_norm_kernel", [&] {
      if (vec_size == 8) { LAUNCH_KERNEL(scalar_t, 8, 3); }
      else if (vec_size == 4) { LAUNCH_KERNEL(scalar_t, 4, 3); }
      else if (vec_size == 2) { LAUNCH_KERNEL(scalar_t, 2, 3); }
      else { LAUNCH_KERNEL(scalar_t, 1, 3); }
    });
  } else if (num_dims == 4) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "rms_norm_kernel", [&] {
      if (vec_size == 8) { LAUNCH_KERNEL(scalar_t, 8, 4); }
      else if (vec_size == 4) { LAUNCH_KERNEL(scalar_t, 4, 4); }
      else if (vec_size == 2) { LAUNCH_KERNEL(scalar_t, 2, 4); }
      else { LAUNCH_KERNEL(scalar_t, 1, 4); }
    });
  }

  #undef LAUNCH_KERNEL
}

