// Group-32 W8A8 projections. Weights store four signed INT8 codes per int32
// in contiguous output-major order; group scales retain the compute dtype.
// Per-row dynamic activation quantization feeds exact INT32 DP4A group dots,
// followed by FP32 scaling/reduction. No second weight layout is retained.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {
// Explicit round-to-nearest division preserves the group-32 quantizer even
// when the extension is built with --use_fast_math.
template <typename T, bool RoundOnly>
__global__ void quantize(const T* x, int* q, float* scales, T* rounded, int K) {
  int lane = threadIdx.x & 31;
  int group = blockIdx.x * 4 + threadIdx.x / 32;
  int groups = K / 32;
  if (group >= groups) return;
  int offset = blockIdx.y * K + group * 32 + lane;
  float value = static_cast<float>(x[offset]);
  float maximum = fabsf(value);
  for (int d = 16; d; d >>= 1) maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, d));
  float scale = fmaxf(__fdiv_rn(maximum, 127.f), 1.e-30f);
  int code = max(-127, min(127, __float2int_rn(__fdiv_rn(value, scale))));
  if constexpr (RoundOnly) {
    rounded[offset] = static_cast<T>(code * scale);
  } else {
    unsigned packed = static_cast<unsigned>(code) & 255;
    packed |= __shfl_down_sync(0xffffffff, packed, 1) << 8;
    packed |= __shfl_down_sync(0xffffffff, packed, 2) << 16;
    if (!(lane & 3)) q[blockIdx.y * (K / 4) + group * 8 + lane / 4] = packed;
    if (!lane) scales[blockIdx.y * groups + group] = scale;
  }
}

// Reuse each weight group across up to eight activation rows. A separate
// single-row instantiation keeps decode register pressure low.
template <typename T, int Rows>
__global__ void gemv(const int* q, const float* xs, const int* w, const T* ws,
                     T* out, int K, int N, int M) {
  int lane = threadIdx.x & 31;
  int n = blockIdx.x * 4 + threadIdx.x / 32;
  if (n >= N) return;
  int groups = K / 32;
  int first_row = blockIdx.y * Rows;
  float result[Rows] = {};
  for (int g = lane; g < groups; g += 32) {
    int weights[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) weights[i] = w[n * (K / 4) + g * 8 + i];
    float scale_w = static_cast<float>(ws[n * groups + g]);
    #pragma unroll
    for (int row = 0; row < Rows; ++row) {
      if (first_row + row < M) {
        int dot = 0;
        const int* a = q + (first_row + row) * (K / 4) + g * 8;
        #pragma unroll
        for (int i = 0; i < 8; ++i) dot = __dp4a(a[i], weights[i], dot);
        result[row] += __fmul_rn(__fmul_rn(static_cast<float>(dot), xs[(first_row + row) * groups + g]), scale_w);
      }
    }
  }
  #pragma unroll
  for (int row = 0; row < Rows; ++row) {
    for (int d = 16; d; d >>= 1) result[row] += __shfl_down_sync(0xffffffff, result[row], d);
    if (!lane && first_row + row < M) out[(first_row + row) * N + n] = static_cast<T>(result[row]);
  }
}

void check_input(const torch::Tensor& x) {
  TORCH_CHECK(x.is_cuda() && x.dim() >= 1 && x.is_contiguous(), "contiguous CUDA input required");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf, "FP16/BF16 input required");
  TORCH_CHECK(x.size(-1) >= 128 && x.size(-1) % 128 == 0, "input width must be a multiple of 128");
}
}  // namespace

torch::Tensor int8_projection_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor ws) {
  check_input(x);
  c10::cuda::CUDAGuard guard(x.device());
  int K = x.size(-1), rows = x.numel() / K;
  TORCH_CHECK(w.is_cuda() && ws.is_cuda() && w.device() == x.device() && ws.device() == x.device(), "device mismatch");
  TORCH_CHECK(w.dim() == 2 && w.size(1) == K / 4 && w.scalar_type() == at::kInt && w.is_contiguous(), "invalid INT8 packed weights");
  int N = w.size(0);
  TORCH_CHECK(ws.dim() == 2 && ws.size(0) == N && ws.size(1) == K / 32 && ws.scalar_type() == x.scalar_type() && ws.is_contiguous(), "invalid group-32 scales");
  auto shape = x.sizes().vec(); shape.back() = N;
  auto out = torch::empty(shape, x.options());
  if (!rows || !N) return out;
  auto q = torch::empty({rows, K / 4}, x.options().dtype(at::kInt));
  auto scales = torch::empty({rows, K / 32}, x.options().dtype(at::kFloat));
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_REDUCED_FLOATING_TYPES(x.scalar_type(), "int8_projection", [&] {
    quantize<scalar_t, false><<<dim3((K / 32 + 3) / 4, rows), 128, 0, stream>>>(x.data_ptr<scalar_t>(), q.data_ptr<int>(), scales.data_ptr<float>(), nullptr, K);
    if (rows == 1)
      gemv<scalar_t, 1><<<dim3((N + 3) / 4), 128, 0, stream>>>(q.data_ptr<int>(), scales.data_ptr<float>(), w.data_ptr<int>(), ws.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), K, N, rows);
    else
      gemv<scalar_t, 8><<<dim3((N + 3) / 4, (rows + 7) / 8), 128, 0, stream>>>(q.data_ptr<int>(), scales.data_ptr<float>(), w.data_ptr<int>(), ws.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), K, N, rows);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor round_int8_inputs_cuda(torch::Tensor x) {
  check_input(x);
  c10::cuda::CUDAGuard guard(x.device());
  auto out = torch::empty_like(x);
  int K = x.size(-1), rows = x.numel() / K;
  if (!rows) return out;
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_REDUCED_FLOATING_TYPES(x.scalar_type(), "round_int8_inputs", [&] {
    quantize<scalar_t, true><<<dim3((K / 32 + 3) / 4, rows), 128, 0, stream>>>(x.data_ptr<scalar_t>(), nullptr, nullptr, out.data_ptr<scalar_t>(), K);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
