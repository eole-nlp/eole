/*
 * Dense Marlin GEMM kernel for eole.
 * Adapted from gptqmodel/gptqmodel_ext/marlin/gptq_marlin.cu (Apache-2.0).
 * Original: https://github.com/ModelCloud/GPTQModel
 *
 * The dense path uses the unified Marlin<> kernel (is_moe=false).  The 6 MoE
 * routing parameters are present in the signature but are passed as nullptr/0
 * and are never accessed – if constexpr(is_moe==false) elides them entirely.
 */

#include "marlin_kernel.h"   // deps + MARLIN_KERNEL_PARAMS + namespace marlin { Marlin<> }
#include "marlin_kernel_shapes.h"
#include "marlin_type_ids.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_fp16.h>
#include <optional>
#include <type_traits>

namespace marlin {

__global__ void permute_cols_kernel(
    int4 const* __restrict__ a_int4_ptr,
    int const* __restrict__ perm_int_ptr,
    int4* __restrict__ out_int4_ptr,
    int size_m, int size_k, int lda, int block_rows);

using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

// ── Thread / exec config ──────────────────────────────────────────────────────

typedef struct {
  int thread_k;
  int thread_n;
  int num_threads;
} thread_config_t;

thread_config_t small_batch_thread_configs[] = {
    {128, 128, 256},
    {64,  128, 128},
    {128,  64, 128},
};

thread_config_t large_batch_thread_configs[] = {
    {64,  256, 256},
    {64,  128, 128},
    {128,  64, 128},
};

typedef struct {
  int blocks_per_sm;
  thread_config_t tb_cfg;
} exec_config_t;

// ── Shared-memory size helpers ────────────────────────────────────────────────

int get_scales_cache_size(thread_config_t const& th_config,
                          int /*prob_m*/, int /*prob_n*/, int /*prob_k*/,
                          int /*num_bits*/, int group_size,
                          bool has_act_order, bool is_k_full) {
  bool cache_scales_chunk = has_act_order && !is_k_full;
  int tb_n = th_config.thread_n;
  int tb_k = th_config.thread_k;

  int tb_groups = (group_size == -1) ? 1
                : (group_size == 0)  ? div_ceil(tb_k, 32)
                                     : div_ceil(tb_k, group_size);

  if (cache_scales_chunk) {
    int load_groups = max(tb_groups * pipe_stages * 2, 32);
    return load_groups * tb_n * 2;
  }
  return tb_groups * tb_n * 2 * pipe_stages;
}

int get_kernel_cache_size(thread_config_t const& th_config, int thread_m_blocks,
                          int prob_m, int prob_n, int prob_k, int num_bits,
                          int group_size, bool has_act_order, bool is_k_full,
                          bool has_zp, bool is_zp_float) {
  int pack_factor = 32 / num_bits;
  int tb_k = th_config.thread_k;
  int tb_n = th_config.thread_n;
  int tb_m = thread_m_blocks * 16;

  int sh_a_size   = pipe_stages * (tb_m * tb_k) * 2;
  int sh_b_size   = pipe_stages * (tb_k * tb_n / pack_factor) * 4;
  int sh_red_size = tb_m * (tb_n + 8) * 2;
  int sh_bias_size = tb_n * 2;

  int tmp_min = min(sh_b_size, sh_red_size);
  int tmp_max = max(sh_b_size, sh_red_size);
  int tmp_size = max(tmp_max, tmp_min + sh_bias_size);

  int sh_s_size = get_scales_cache_size(th_config, prob_m, prob_n, prob_k,
                                        num_bits, group_size, has_act_order,
                                        is_k_full);
  int sh_g_idx_size = (has_act_order && !is_k_full) ? pipe_stages * tb_k / 4 : 0;

  int sh_zp_size = 0;
  if (has_zp) {
    sh_zp_size = is_zp_float ? sh_s_size
               : (num_bits == 4) ? sh_s_size / 4
                                 : sh_s_size / 2;
  }

  return tmp_size + sh_a_size + sh_s_size + sh_zp_size + sh_g_idx_size;
}

bool is_valid_config(thread_config_t const& th_config, int thread_m_blocks,
                     int prob_m, int prob_n, int prob_k, int num_bits,
                     int group_size, bool has_act_order, bool is_k_full,
                     bool has_zp, bool is_zp_float, int max_shared_mem) {
  if (th_config.thread_k == -1 || th_config.thread_n == -1 ||
      th_config.num_threads == -1)
    return false;
  if (prob_k % th_config.thread_k != 0 || prob_n % th_config.thread_n != 0)
    return false;
  if (th_config.thread_n < min_thread_n || th_config.thread_k < min_thread_k)
    return false;
  if (th_config.num_threads < 128)
    return false;

  int cache_size = get_kernel_cache_size(th_config, thread_m_blocks,
                                         prob_m, prob_n, prob_k, num_bits,
                                         group_size, has_act_order, is_k_full,
                                         has_zp, is_zp_float);
  return cache_size <= max_shared_mem;
}

// ── Kernel selection ──────────────────────────────────────────────────────────
// The Marlin<> template from marlin_kernel.h takes ScalarTypeId non-type
// parameters (a_type_id, b_type_id, c_type_id, s_type_id, ...).
// We enumerate instantiations via MATCH/K macros (same style as MoE dispatch).

MarlinFuncPtr get_marlin_kernel(
    vllm::ScalarTypeId a_id, vllm::ScalarTypeId b_id,
    vllm::ScalarTypeId c_id, vllm::ScalarTypeId s_id,
    int threads, int tm, int tn, int tk,
    bool m8, int group_blocks, bool is_zp_float)
{
#define MATCH(A,B,C,S,TH,TM_,TN_,TK_,M8_,GB_,ZPF_) \
  (a_id==(A) && b_id==(B) && c_id==(C) && s_id==(S) && \
   threads==(TH) && tm==(TM_) && tn==(TN_) && tk==(TK_) && \
   m8==(M8_) && group_blocks==(GB_) && is_zp_float==(ZPF_))

#define K(A,B,C,S,TH,TM_,TN_,TK_,M8_,GB_,ZPF_) \
  Marlin<A, B, C, S, TH, TM_, TN_, TK_, M8_, pipe_stages, GB_, ZPF_, /*is_moe=*/false>

#define DISPATCH_U4B8_FP16(TH,TM_,TN_,TK_,M8_,GB_) \
  if (MATCH(FP16_ID,U4B8_ID,FP16_ID,FP16_ID,TH,TM_,TN_,TK_,M8_,GB_,false)) return K(FP16_ID,U4B8_ID,FP16_ID,FP16_ID,TH,TM_,TN_,TK_,M8_,GB_,false);
#define DISPATCH_U4B8_BF16(TH,TM_,TN_,TK_,M8_,GB_) \
  if (MATCH(BF16_ID,U4B8_ID,BF16_ID,BF16_ID,TH,TM_,TN_,TK_,M8_,GB_,false)) return K(BF16_ID,U4B8_ID,BF16_ID,BF16_ID,TH,TM_,TN_,TK_,M8_,GB_,false);
#define DISPATCH_U8B128_FP16(TH,TM_,TN_,TK_,M8_,GB_) \
  if (MATCH(FP16_ID,U8B128_ID,FP16_ID,FP16_ID,TH,TM_,TN_,TK_,M8_,GB_,false)) return K(FP16_ID,U8B128_ID,FP16_ID,FP16_ID,TH,TM_,TN_,TK_,M8_,GB_,false);
#define DISPATCH_U8B128_BF16(TH,TM_,TN_,TK_,M8_,GB_) \
  if (MATCH(BF16_ID,U8B128_ID,BF16_ID,BF16_ID,TH,TM_,TN_,TK_,M8_,GB_,false)) return K(BF16_ID,U8B128_ID,BF16_ID,BF16_ID,TH,TM_,TN_,TK_,M8_,GB_,false);
#define DISPATCH_U4_FP16_ZP(TH,TM_,TN_,TK_,M8_,GB_) \
  if (MATCH(FP16_ID,U4_ID,FP16_ID,FP16_ID,TH,TM_,TN_,TK_,M8_,GB_,false)) return K(FP16_ID,U4_ID,FP16_ID,FP16_ID,TH,TM_,TN_,TK_,M8_,GB_,false);

  // ── FP16 weight types ────────────────────────────────────────────────────
  // uint4b8  (symmetric 4-bit), fp16 activations
  // Group layouts: -1 (per-col), 2, 4, 8 blocks; act-order (gb=0)
#define ENUMERATE_U4B8_FP16(GB) \
  MARLIN_FOR_EACH_SHAPE_WITH_GB(DISPATCH_U4B8_FP16, GB)

  ENUMERATE_U4B8_FP16(-1)
  ENUMERATE_U4B8_FP16(2)
  ENUMERATE_U4B8_FP16(4)
  ENUMERATE_U4B8_FP16(8)
  ENUMERATE_U4B8_FP16(0)  // act-order

  // ── BF16 weight types ────────────────────────────────────────────────────
#define ENUMERATE_U4B8_BF16(GB) \
  MARLIN_FOR_EACH_SHAPE_WITH_GB(DISPATCH_U4B8_BF16, GB)

  ENUMERATE_U4B8_BF16(-1)
  ENUMERATE_U4B8_BF16(2)
  ENUMERATE_U4B8_BF16(4)
  ENUMERATE_U4B8_BF16(8)
  ENUMERATE_U4B8_BF16(0)  // act-order

  // ── uint8b128 (symmetric 8-bit) ──────────────────────────────────────────
#define ENUMERATE_U8B128_FP16(GB) \
  MARLIN_FOR_EACH_SHAPE_WITH_GB(DISPATCH_U8B128_FP16, GB)

  ENUMERATE_U8B128_FP16(-1)
  ENUMERATE_U8B128_FP16(2)
  ENUMERATE_U8B128_FP16(4)
  ENUMERATE_U8B128_FP16(8)

#define ENUMERATE_U8B128_BF16(GB) \
  MARLIN_FOR_EACH_SHAPE_WITH_GB(DISPATCH_U8B128_BF16, GB)

  ENUMERATE_U8B128_BF16(-1)
  ENUMERATE_U8B128_BF16(2)
  ENUMERATE_U8B128_BF16(4)
  ENUMERATE_U8B128_BF16(8)

  // ── Asymmetric (u4 with zero-points, fp16 only / is_zp_float=false) ──────
  // group_blocks=4 is the most common for AWQ/GPTQ asymmetric
#define ENUMERATE_U4_FP16_ZP(GB) \
  MARLIN_FOR_EACH_SHAPE_WITH_GB(DISPATCH_U4_FP16_ZP, GB)

  ENUMERATE_U4_FP16_ZP(-1)
  ENUMERATE_U4_FP16_ZP(2)
  ENUMERATE_U4_FP16_ZP(4)
  ENUMERATE_U4_FP16_ZP(8)

#undef MATCH
#undef K
#undef DISPATCH_U4B8_FP16
#undef DISPATCH_U4B8_BF16
#undef DISPATCH_U8B128_FP16
#undef DISPATCH_U8B128_BF16
#undef DISPATCH_U4_FP16_ZP
#undef ENUMERATE_U4B8_FP16
#undef ENUMERATE_U4B8_BF16
#undef ENUMERATE_U8B128_FP16
#undef ENUMERATE_U8B128_BF16
#undef ENUMERATE_U4_FP16_ZP

  return nullptr;
}

// ── Exec config selection ─────────────────────────────────────────────────────

exec_config_t determine_exec_config(
    vllm::ScalarTypeId a_id, vllm::ScalarTypeId b_id,
    vllm::ScalarTypeId c_id, vllm::ScalarTypeId s_id,
    int prob_m, int prob_n, int prob_k,
    int thread_m_blocks, bool m_block_size_8,
    int num_bits, int group_size, bool has_act_order, bool is_k_full,
    bool has_zp, bool is_zp_float, int max_shared_mem, int sms)
{
  exec_config_t best = {1, {-1, -1, -1}};
  thread_config_t* cfgs = (thread_m_blocks > 1)
                          ? large_batch_thread_configs
                          : small_batch_thread_configs;
  int n_cfgs = (thread_m_blocks > 1)
      ? (int)(sizeof(large_batch_thread_configs) / sizeof(thread_config_t))
      : (int)(sizeof(small_batch_thread_configs) / sizeof(thread_config_t));

  for (int i = 0; i < n_cfgs; i++) {
    auto& th = cfgs[i];
    if (!is_valid_config(th, thread_m_blocks, prob_m, prob_n, prob_k, num_bits,
                         group_size, has_act_order, is_k_full, has_zp,
                         is_zp_float, max_shared_mem))
      continue;

    int group_blocks = has_act_order ? 0 : (group_size == -1 ? -1 : group_size / 16);

    auto kernel = get_marlin_kernel(a_id, b_id, c_id, s_id,
                                    th.num_threads, thread_m_blocks,
                                    th.thread_n / 16, th.thread_k / 16,
                                    m_block_size_8, group_blocks, is_zp_float);
    if (kernel == nullptr) continue;
    return {1, th};
  }
  return best;
}

// ── Inner GEMM dispatcher ─────────────────────────────────────────────────────

void marlin_mm(const void* A, const void* B, void* C, void* C_tmp,
               void* b_bias, void* s, void* zp, void* g_idx, void* perm,
               void* a_tmp, int prob_m, int prob_n, int prob_k, int lda,
               void* workspace, vllm::ScalarTypeId a_id, vllm::ScalarTypeId b_id,
               vllm::ScalarTypeId c_id, vllm::ScalarTypeId s_id,
               bool has_bias, bool has_act_order, bool is_k_full, bool has_zp,
               int num_groups, int group_size, int dev, cudaStream_t stream,
               int sms, bool use_atomic_add, bool use_fp32_reduce,
               bool is_zp_float) {
  TORCH_CHECK(prob_m > 0 && prob_n > 0 && prob_k > 0,
              "Invalid MNK=[", prob_m, ",", prob_n, ",", prob_k, "]");

  int group_blocks = 0;
  if (has_act_order) {
    if (is_k_full) {
      TORCH_CHECK(group_size != -1);
      group_blocks = group_size / 16;
    } else {
      TORCH_CHECK(group_size == 0);
    }
  } else {
    group_blocks = (group_size == -1) ? -1 : group_size / 16;
  }

  vllm::ScalarType b_type = vllm::ScalarType::from_id(b_id);
  int num_bits = b_type.size_bits();

  const int4* A_ptr     = (const int4*)A;
  const int4* B_ptr     = (const int4*)B;
  int4*       C_ptr     = (int4*)C;
  int4*       C_tmp_ptr = (int4*)C_tmp;
  const int4* bias_ptr  = (const int4*)b_bias;
  const int4* s_ptr     = (const int4*)s;
  const int4* zp_ptr    = (const int4*)zp;
  const int*  g_idx_ptr = (const int*)g_idx;
  const int*  perm_ptr  = (const int*)perm;
  int4*       a_tmp_ptr = (int4*)a_tmp;
  int*        locks     = (int*)workspace;

  if (has_act_order) {
    int block_rows = div_ceil(prob_m, sms);
    permute_cols_kernel<<<sms, default_threads, 0, stream>>>(
        A_ptr, perm_ptr, a_tmp_ptr, prob_m, prob_k, lda, block_rows);
    A_ptr = a_tmp_ptr;
    lda   = prob_k;
    if (is_k_full) has_act_order = false;
  }

  int max_shared_mem = 0;
  cudaDeviceGetAttribute(&max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  TORCH_CHECK(max_shared_mem > 0);

  int max_par            = (prob_n <= 4096) ? 16 * 8 : 16;
  int max_shared_mem_new = max_shared_mem;
  int rest_m             = prob_m;
  int max_thread_m_blocks = 4;

  while (rest_m) {
    int par_count = rest_m / (max_thread_m_blocks * 16);
    if (par_count > max_par) par_count = max_par;
    int prob_m_split = (par_count > 0) ? par_count * max_thread_m_blocks * 16
                                       : rest_m;

    int thread_m_blocks = min(div_ceil(prob_m_split, 16), max_thread_m_blocks);
    bool m_block_size_8 = prob_m_split <= 8;

    exec_config_t exec_cfg = determine_exec_config(
        a_id, b_id, c_id, s_id,
        prob_m_split, prob_n, prob_k,
        thread_m_blocks, m_block_size_8,
        num_bits, group_size, has_act_order, is_k_full,
        has_zp, is_zp_float, max_shared_mem, sms);

    auto& thread_tfg = exec_cfg.tb_cfg;
    if (thread_tfg.thread_k == -1 && max_thread_m_blocks > 1) {
      max_thread_m_blocks--;
      continue;
    }

    int num_threads     = thread_tfg.num_threads;
    int thread_k        = thread_tfg.thread_k;
    int thread_n        = thread_tfg.thread_n;
    int blocks          = sms * exec_cfg.blocks_per_sm;
    int thread_k_blocks = thread_k / 16;
    int thread_n_blocks = thread_n / 16;

    TORCH_CHECK(is_valid_config(thread_tfg, thread_m_blocks, prob_m_split,
                                prob_n, prob_k, num_bits, group_size,
                                has_act_order, is_k_full, has_zp, is_zp_float,
                                max_shared_mem_new),
                "No valid dense Marlin thread config for MNK=[", prob_m, ",",
                prob_n, ",", prob_k, "] num_bits=", num_bits);

    auto kernel = get_marlin_kernel(a_id, b_id, c_id, s_id,
                                    num_threads, thread_m_blocks,
                                    thread_n_blocks, thread_k_blocks,
                                    m_block_size_8, group_blocks, is_zp_float);

    TORCH_CHECK(kernel != nullptr,
                "Unsupported dense Marlin shape MNK=[", prob_m, ",", prob_n,
                ",", prob_k, "]");

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_new);

    bool part_use_atomic_add =
        use_atomic_add && div_ceil(prob_m_split, 64) * prob_n <= 2048;

    kernel<<<blocks, num_threads, max_shared_mem_new, stream>>>(
        A_ptr, B_ptr, C_ptr, C_tmp_ptr, bias_ptr,
        /*a_scales_ptr=*/nullptr,
        s_ptr, /*global_scale_ptr=*/nullptr, zp_ptr, g_idx_ptr,
        /*sorted_token_ids_ptr=*/nullptr, /*expert_ids_ptr=*/nullptr,
        /*num_tokens_past_padded_ptr=*/nullptr, /*topk_weights_ptr=*/nullptr,
        /*top_k=*/0, /*mul_topk_weights=*/false,
        num_groups, prob_m_split, prob_n, prob_k, locks,
        has_bias, part_use_atomic_add, use_fp32_reduce);

    A_ptr  += prob_m_split * (lda / 8);
    C_ptr  += prob_m_split * (prob_n / 8);
    rest_m -= prob_m_split;
  }
}

__global__ void permute_cols_kernel(
    int4 const* __restrict__ a_int4_ptr,
    int const* __restrict__ perm_int_ptr,
    int4* __restrict__ out_int4_ptr,
    int size_m, int size_k, int lda, int block_rows) {

  int start_row = block_rows * blockIdx.x;
  int finish_row = min(start_row + block_rows, size_m);

  int input_row_stride  = lda * sizeof(half) / 16;
  int output_row_stride = size_k * sizeof(half) / 16;

  for (int row = start_row; row < finish_row; row++) {

    half const* a_row_half =
        reinterpret_cast<half const*>(a_int4_ptr + row * input_row_stride);

    half* out_half =
        reinterpret_cast<half*>(out_int4_ptr + row * output_row_stride);

    int base_k = 0;
    int iters = size_k / default_threads;
    int rest  = size_k % default_threads;

    for (int i = 0; i < iters; i++) {
      int k = base_k + threadIdx.x;
      int src = perm_int_ptr[k];
      out_half[k] = a_row_half[src];
      base_k += default_threads;
    }

    if (threadIdx.x < rest) {
      int k = base_k + threadIdx.x;
      int src = perm_int_ptr[k];
      out_half[k] = a_row_half[src];
    }
  }
}

}  // namespace marlin

// ── Host entry point ──────────────────────────────────────────────────────────

torch::Tensor gptq_marlin_gemm(
    torch::Tensor&                          a,
    std::optional<torch::Tensor>            c_or_none,
    torch::Tensor&                          b_q_weight,
    std::optional<torch::Tensor> const&     b_bias_or_none,
    torch::Tensor&                          b_scales,
    std::optional<torch::Tensor> const&     global_scale_or_none,
    std::optional<torch::Tensor> const&     b_zeros_or_none,
    std::optional<torch::Tensor> const&     g_idx_or_none,
    std::optional<torch::Tensor> const&     perm_or_none,
    torch::Tensor&                          workspace,
    int64_t                                 b_q_type_id,
    int64_t                                 size_m,
    int64_t                                 size_n,
    int64_t                                 size_k,
    bool                                    is_k_full,
    bool                                    use_atomic_add,
    bool                                    use_fp32_reduce,
    bool                                    is_zp_float) {
  using namespace marlin;

  vllm::ScalarType b_q_type = vllm::ScalarType::from_id(b_q_type_id);
  int pack_factor = 32 / b_q_type.size_bits();

  // Determine activation ScalarTypeId from tensor dtype
  vllm::ScalarTypeId a_id, c_id, s_id;
  if (a.scalar_type() == at::ScalarType::Half) {
    a_id = vllm::kFloat16.id();
    c_id = vllm::kFloat16.id();
    s_id = vllm::kFloat16.id();
  } else if (a.scalar_type() == at::ScalarType::BFloat16) {
    a_id = vllm::kBFloat16.id();
    c_id = vllm::kBFloat16.id();
    s_id = vllm::kBFloat16.id();
  } else {
    TORCH_CHECK(false, "gptq_marlin_gemm: unsupported activation dtype");
  }
  vllm::ScalarTypeId b_id = b_q_type_id;

  TORCH_CHECK(a.size(0) == size_m);
  TORCH_CHECK(a.size(1) == size_k);
  TORCH_CHECK(size_k % tile_size == 0);
  TORCH_CHECK((size_k / tile_size) == b_q_weight.size(0));
  TORCH_CHECK(b_q_weight.size(1) % tile_size == 0);
  int actual_n = (b_q_weight.size(1) / tile_size) * pack_factor;
  TORCH_CHECK(size_n == actual_n,
              "size_n=", size_n, " actual_n=", actual_n);
  TORCH_CHECK(a.device().is_cuda());
  TORCH_CHECK(a.stride(1) == 1);
  TORCH_CHECK(a.stride(0) % 8 == 0);
  TORCH_CHECK(b_q_weight.device().is_cuda() && b_q_weight.is_contiguous());
  TORCH_CHECK(b_scales.device().is_cuda() && b_scales.is_contiguous());

  int sms = -1;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, a.get_device());

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto options = torch::TensorOptions().dtype(a.dtype()).device(a.device());

  torch::Tensor c;
  if (c_or_none.has_value()) {
    c = c_or_none.value();
    TORCH_CHECK(c.device().is_cuda() && c.is_contiguous());
    TORCH_CHECK(c.size(0) == size_m && c.size(1) == size_n);
  } else {
    c = torch::empty({size_m, size_n}, options);
  }
  if (size_m == 0) return c;

  // Temporary fp32 buffer for cross-block reduction
  torch::Tensor c_tmp;
  if (use_fp32_reduce) {
    int max_m_block = min((int)((size_m + 15) / 16 * 16), 64);
    c_tmp = torch::empty({sms * max_m_block * max_thread_n},
                         options.dtype(at::kFloat));
  } else {
    c_tmp = torch::empty({0}, options.dtype(at::kFloat));
  }

  TORCH_CHECK(b_scales.sizes().size() == 2);
  TORCH_CHECK(b_scales.size(1) == size_n);
  int num_groups = (int)b_scales.size(0);

  bool has_act_order = false;
  int  group_size    = -1;
  torch::Tensor g_idx, perm, a_tmp;

  if (g_idx_or_none.has_value() && perm_or_none.has_value()) {
    g_idx = g_idx_or_none.value();
    perm  = perm_or_none.value();
    if (g_idx.numel() > 0 && perm.numel() > 0) {
      TORCH_CHECK(g_idx.size(0) == size_k && perm.size(0) == size_k);
      has_act_order = true;
      a_tmp = torch::empty({size_m, size_k}, options);
    }
  }
  group_size = (num_groups == 1) ? -1 : (int)(size_k / num_groups);

  bool has_zp = b_zeros_or_none.has_value() &&
                b_zeros_or_none.value().numel() > 0;
  torch::Tensor b_zeros;
  if (has_zp) b_zeros = b_zeros_or_none.value();

  bool has_bias = b_bias_or_none.has_value() &&
                  b_bias_or_none.value().numel() > 0;
  torch::Tensor b_bias;
  if (has_bias) b_bias = b_bias_or_none.value();

  auto stream = at::cuda::getCurrentCUDAStream(a.get_device()).stream();

  marlin::marlin_mm(
      a.data_ptr(), b_q_weight.data_ptr(), c.data_ptr(), c_tmp.data_ptr(),
      has_bias  ? b_bias.data_ptr()   : nullptr,
      b_scales.data_ptr(),
      has_zp    ? b_zeros.data_ptr()  : nullptr,
      has_act_order ? g_idx.data_ptr() : nullptr,
      has_act_order ? perm.data_ptr()  : nullptr,
      has_act_order ? a_tmp.data_ptr() : nullptr,
      (int)size_m, (int)size_n, (int)size_k, (int)a.stride(0),
      workspace.data_ptr(),
      a_id, b_id, c_id, s_id,
      has_bias, has_act_order, is_k_full, has_zp,
      num_groups, group_size, a.get_device(), stream, sms,
      use_atomic_add, use_fp32_reduce, is_zp_float);

  return c;
}
