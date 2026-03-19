/*
 * Eole-NLP – Marlin MoE GEMM kernel (adapted from vLLM)
 * Copyright (C) Marlin.2024 Elias Frantar / Neural Magic.
 *
 * Only the Marlin<> instantiations actually used by typical Marlin MoE
 * inference (group_size=128, fp16/bf16, uint4b8/uint8b128, Ampere+) are
 * compiled.  Add more following the INST_ macros below if needed.
 *
 * Eole additions vs vLLM ops.cu:
 *   1. Static device-property cache  – eliminates 3x cudaDeviceGetAttribute.
 *   2. Static kernel register-count cache – eliminates cudaFuncGetAttributes.
 *   3. Static exec_config cache – O(1) config lookup after first call.
 *   4. c_tmp_or_none – optional pre-allocated fp32 scratch buffer.
 */

#include "marlin_kernel.h"   // deps + MARLIN_KERNEL_PARAMS + namespace marlin { Marlin<> }
#include "marlin_kernel_shapes.h"
#include "marlin_type_ids.h"

#include <mutex>
#include <unordered_map>

namespace marlin {

using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

// =============================================================================
// Explicit kernel instantiations
//
// Only the configs needed for typical Marlin MoE inference are compiled.
// This keeps the .so small.  Pattern: (a_type, b_type, c_type, s_type,
//   threads, thread_m_blocks, thread_n_blocks, thread_k_blocks,
//   m_block_size_8, stages, group_blocks, is_zp_float)
//
// group_blocks=8  ← group_size=128 (standard gptqmodel Marlin)
// stages=4        ← Ampere / Ada (sm80+); add stages=2 for Turing (sm75)
// is_zp_float=false ← symmetric quantisation always
//
// To add more: copy an INST_xxx line, adjust the config, recompile.
// =============================================================================

#define INST(A, B, C, S, TH, TM, TN, TK, M8, ST, GB)            \
  template __global__ void Marlin<A, B, C, S,                    \
      TH, TM, TN, TK, M8, ST, GB, /*is_zp_float=*/false, /*is_moe=*/true>(MARLIN_KERNEL_PARAMS);

// Macros for the two common dtype/weight-type combinations
#define INST_FP16_U4B8(TH, TM, TN, TK, M8, ST) \
  INST(FP16_ID, U4B8_ID, FP16_ID, FP16_ID, TH, TM, TN, TK, M8, ST, 8)
#define INST_BF16_U4B8(TH, TM, TN, TK, M8, ST) \
  INST(BF16_ID, U4B8_ID, BF16_ID, BF16_ID, TH, TM, TN, TK, M8, ST, 8)

// ── Decode (moe_block_size=8, m_block_size_8=true, thread_m_blocks=1) ─────────
// All three thread configs from small_batch_thread_configs.
INST_FP16_U4B8(256, 1, 8, 8, true,  4)
INST_FP16_U4B8(128, 1, 8, 4, true,  4)
INST_FP16_U4B8(128, 1, 4, 8, true,  4)
INST_BF16_U4B8(256, 1, 8, 8, true,  4)
INST_BF16_U4B8(128, 1, 8, 4, true,  4)
INST_BF16_U4B8(128, 1, 4, 8, true,  4)

// ── Prefill tm=1 (moe_block_size>=16, m_block_size_8=false) ───────────────────
INST_FP16_U4B8(256, 1, 8, 8, false, 4)
INST_FP16_U4B8(128, 1, 8, 4, false, 4)
INST_FP16_U4B8(128, 1, 4, 8, false, 4)
INST_BF16_U4B8(256, 1, 8, 8, false, 4)
INST_BF16_U4B8(128, 1, 8, 4, false, 4)
INST_BF16_U4B8(128, 1, 4, 8, false, 4)

// ── Prefill tm=2 (large_batch_thread_configs) ─────────────────────────────────
INST_FP16_U4B8(256, 2, 16, 4, false, 4)
INST_FP16_U4B8(128, 2,  8, 4, false, 4)
INST_FP16_U4B8(128, 2,  4, 8, false, 4)
INST_BF16_U4B8(256, 2, 16, 4, false, 4)
INST_BF16_U4B8(128, 2,  8, 4, false, 4)
INST_BF16_U4B8(128, 2,  4, 8, false, 4)

// ── Prefill tm=3 (moe_block_size=48) ──────────────────────────────────────
INST_FP16_U4B8(256, 3, 16, 4, false, 4)
INST_FP16_U4B8(128, 3,  8, 4, false, 4)
INST_FP16_U4B8(128, 3,  4, 8, false, 4)
INST_BF16_U4B8(256, 3, 16, 4, false, 4)
INST_BF16_U4B8(128, 3,  4, 8, false, 4)
INST_BF16_U4B8(256, 3,  8, 4, false, 4)

// ── Prefill tm=4 ──────────────────────────────────────────────────────────────
INST_FP16_U4B8(256, 4, 16, 4, false, 4)
INST_FP16_U4B8(128, 4,  8, 4, false, 4)
INST_FP16_U4B8(128, 4,  4, 8, false, 4)
INST_BF16_U4B8(256, 4, 16, 4, false, 4)
INST_BF16_U4B8(128, 4,  8, 4, false, 4)
INST_BF16_U4B8(128, 4,  4, 8, false, 4)

#undef INST
#undef INST_FP16_U4B8
#undef INST_BF16_U4B8

// =============================================================================
// Cache 1: per-device properties
// =============================================================================

struct CachedDeviceProps {
  int max_shared_mem;
  int stages;
  int sms;
};

static std::unordered_map<int, CachedDeviceProps> g_dev_cache;
static std::mutex                                  g_dev_mutex;

static const CachedDeviceProps& get_device_props(int dev) {
  {
    std::lock_guard<std::mutex> lk(g_dev_mutex);
    auto it = g_dev_cache.find(dev);
    if (it != g_dev_cache.end()) return it->second;
  }
  CachedDeviceProps p;
  cudaDeviceGetAttribute(&p.max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  cudaDeviceGetAttribute(&p.sms, cudaDevAttrMultiProcessorCount, dev);
  int major, minor;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
  TORCH_CHECK(major * 10 + minor >= 80,
              "Marlin requires Ampere (sm80) or newer.");
  // Only 4-stage kernels are compiled/used; restrict to architectures that support them.
  p.stages = 4;
  std::lock_guard<std::mutex> lk(g_dev_mutex);
  g_dev_cache[dev] = p;
  return g_dev_cache[dev];
}

// =============================================================================
// Cache 2: kernel register counts
// =============================================================================

static std::unordered_map<const void*, int> g_regs_cache;
static std::mutex                            g_regs_mutex;

static int get_kernel_regs(MarlinFuncPtr k) {
  const void* key = reinterpret_cast<const void*>(k);
  {
    std::lock_guard<std::mutex> lk(g_regs_mutex);
    auto it = g_regs_cache.find(key);
    if (it != g_regs_cache.end()) return it->second;
  }
  cudaFuncAttributes attr;
  cudaFuncGetAttributes(&attr, k);
  int r = std::max(attr.numRegs, 1);
  std::lock_guard<std::mutex> lk(g_regs_mutex);
  g_regs_cache[key] = r;
  return r;
}

// =============================================================================
// Cache 3: exec_config per problem shape
// =============================================================================

struct ExecCfgKey {
  int m, n, k, block, bits, gs, dev;
  bool act, full_k, zp, zp_float, a8;
  bool operator==(const ExecCfgKey& o) const {
    return m==o.m && n==o.n && k==o.k && block==o.block
        && bits==o.bits && gs==o.gs && dev==o.dev
        && act==o.act && full_k==o.full_k && zp==o.zp
        && zp_float==o.zp_float && a8==o.a8;
  }
};
struct ExecCfgHash {
  size_t operator()(const ExecCfgKey& k) const {
    size_t h = 0;
    auto mix = [&](int v){
      h ^= std::hash<int>{}(v) + 0x9e3779b9u + (h<<6) + (h>>2); };
    mix(k.m); mix(k.n); mix(k.k); mix(k.block);
    mix(k.bits); mix(k.gs); mix(k.dev);
    mix((int)k.act); mix((int)k.full_k); mix((int)k.zp);
    mix((int)k.zp_float); mix((int)k.a8);
    return h;
  }
};

typedef struct { int thread_k, thread_n, num_threads; } thread_config_t;
typedef struct { int blocks_per_sm; thread_config_t tb_cfg; } exec_config_t;

using ExecCfgMap = std::unordered_map<ExecCfgKey, exec_config_t, ExecCfgHash>;
static ExecCfgMap g_exec_cache;
static std::mutex g_exec_mutex;

// =============================================================================
// Column-permutation kernel (act_order path)
// =============================================================================

template <int moe_block_size>
__global__ void permute_cols_kernel(
    int4 const* __restrict__ a_int4, int const* __restrict__ perm,
    int4* __restrict__ out,
    const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ ntpp,
    int size_m, int size_k, int top_k)
{
  int total    = ntpp[0];
  int n_blocks = div_ceil(total, moe_block_size);
  int32_t block_ids[moe_block_size];
  int n_valid     = 0;
  int64_t old_eid = 0, eid = 0;
  int row_stride  = size_k * sizeof(half) / 16;

  auto load_block = [&](int bid) {
    n_valid = moe_block_size;
    int4* dst = reinterpret_cast<int4*>(block_ids);
    for (int i = 0; i < moe_block_size / 4; i++)
      dst[i] = ((int4*)sorted_ids)[bid * moe_block_size / 4 + i];
    for (int i = 0; i < moe_block_size; i++)
      if (block_ids[i] >= size_m * top_k) { n_valid = i; break; }
  };

  auto permute_row = [&](int row) {
    int in_off  = (row / top_k) * row_stride;
    int out_off = row * row_stride;
    half const* src = reinterpret_cast<half const*>(a_int4 + in_off);
    half*       dst = reinterpret_cast<half*>(out + out_off);
    int iters = size_k / default_threads;
    int rest  = size_k % default_threads;
    int base  = 0;
    for (int i = 0; i < iters; i++) {
      int ck = base + threadIdx.x; dst[ck] = src[perm[ck]]; base += default_threads;
    }
    if (rest && threadIdx.x < rest) { int ck = base+threadIdx.x; dst[ck]=src[perm[ck]]; }
  };

  for (int idx = blockIdx.x; idx < n_blocks; idx += gridDim.x) {
    old_eid = eid;
    int tmp = expert_ids[idx];
    if (tmp == -1) continue;
    eid = tmp;
    perm += (eid - old_eid) * size_k;
    load_block(idx);
    for (int i = 0; i < n_valid; i++) permute_row(block_ids[i]);
  }
}

// =============================================================================
// Thread / exec configuration helpers
// =============================================================================

thread_config_t small_cfgs[] = {{128,128,256},{64,128,128},{128,64,128}};
thread_config_t large_cfgs[] = {{64,256,256},{64,128,128},{128,64,128}};

int scales_cache_size(thread_config_t const& th, int num_bits, int group_size,
                      bool has_act_order, bool is_k_full, int stages) {
  bool chunk = has_act_order && !is_k_full;
  int tg = (group_size==-1)?1:(group_size==0)?div_ceil(th.thread_k,32)
                                              :div_ceil(th.thread_k,group_size);
  if (chunk) return max(tg*stages*2,32)*th.thread_n*2;
  return tg*th.thread_n*2*stages;
}

int kernel_cache_size(thread_config_t const& th, int tm_blocks, int num_bits,
                      int group_size, bool has_act_order, bool is_k_full,
                      bool has_zp, bool is_zp_float, bool is_a8, int stages) {
  int pf=32/num_bits, tbm=tm_blocks*16;
  int sh_meta=tbm*16;
  int sh_a=stages*tbm*th.thread_k*(is_a8?1:2);
  int sh_b=stages*th.thread_k*th.thread_n/pf*4;
  int sh_red=tbm*(th.thread_n+8)*2;
  int sh_bias=th.thread_n*2;
  int tmp=max(max(sh_b,sh_red),min(sh_b,sh_red)+sh_bias);
  int sh_s=scales_cache_size(th,num_bits,group_size,has_act_order,is_k_full,stages);
  int sh_gi=(has_act_order&&!is_k_full)?stages*th.thread_k/4:0;
  int sh_zp=0;
  if(has_zp){
    if(is_zp_float)sh_zp=sh_s;
    else if(num_bits==4)sh_zp=sh_s/4;
    else sh_zp=sh_s/2;
  }
  return tmp+sh_a+sh_s+sh_zp+sh_gi+sh_meta;
}

bool valid_config(thread_config_t const& th, int tm_blocks, int prob_m,
                  int prob_n, int prob_k, int num_bits, int group_size,
                  bool has_act_order, bool is_k_full, bool has_zp,
                  bool is_zp_float, bool is_a8, int stages, int max_smem) {
  if(th.thread_k<0||th.thread_n<0||th.num_threads<0) return false;
  if(prob_k%th.thread_k||prob_n%th.thread_n) return false;
  if(th.thread_n<min_thread_n||th.thread_k<min_thread_k) return false;
  if(th.num_threads<128) return false;
  return kernel_cache_size(th,tm_blocks,num_bits,group_size,has_act_order,
                            is_k_full,has_zp,is_zp_float,is_a8,stages)<=max_smem;
}

// =============================================================================
// Kernel dispatch
// Returns the instantiated kernel matching the given config, or nullptr.
// Only the instantiations compiled above are reachable; any other combination
// returns nullptr and is skipped by determine_exec_config.
// =============================================================================

MarlinFuncPtr get_marlin_kernel(
    vllm::ScalarTypeId a_id, vllm::ScalarTypeId b_id,
    vllm::ScalarTypeId c_id, vllm::ScalarTypeId s_id,
    int threads, int tm, int tn, int tk,
    bool m8, int stages, int gb, bool is_zp_float)
{
  if (is_zp_float || stages != 4 || gb != 8) return nullptr;

#define MATCH(A,B,C,S,TH,TM_,TN_,TK_,M8_) \
  (a_id==(A) && b_id==(B) && c_id==(C) && s_id==(S) && \
   threads==(TH) && tm==(TM_) && tn==(TN_) && tk==(TK_) && m8==(M8_))

#define K(A,B,C,S,TH,TM_,TN_,TK_,M8_) \
  Marlin<A, B, C, S, TH, TM_, TN_, TK_, M8_, /*stages=*/4, /*group_blocks=*/8, /*is_zp_float=*/false, /*is_moe=*/true>

#define DISPATCH_U4B8_FP16(TH,TM_,TN_,TK_,M8_) \
  if (MATCH(FP16_ID,U4B8_ID,FP16_ID,FP16_ID,TH,TM_,TN_,TK_,M8_)) return K(FP16_ID,U4B8_ID,FP16_ID,FP16_ID,TH,TM_,TN_,TK_,M8_);
#define DISPATCH_U4B8_BF16(TH,TM_,TN_,TK_,M8_) \
  if (MATCH(BF16_ID,U4B8_ID,BF16_ID,BF16_ID,TH,TM_,TN_,TK_,M8_)) return K(BF16_ID,U4B8_ID,BF16_ID,BF16_ID,TH,TM_,TN_,TK_,M8_);

  // fp16 + uint4b8
  if (a_id==FP16_ID && b_id==U4B8_ID && c_id==FP16_ID && s_id==FP16_ID) {
    MARLIN_FOR_EACH_SHAPE(DISPATCH_U4B8_FP16)
  }

  // bfloat16 + uint4b8
  if (a_id==BF16_ID && b_id==U4B8_ID && c_id==BF16_ID && s_id==BF16_ID) {
    MARLIN_FOR_EACH_SHAPE(DISPATCH_U4B8_BF16)
  }

#undef MATCH
#undef K
#undef DISPATCH_U4B8_FP16
#undef DISPATCH_U4B8_BF16
  return nullptr;
}

exec_config_t determine_exec_config(
    vllm::ScalarTypeId a_id, vllm::ScalarTypeId b_id,
    vllm::ScalarTypeId c_id, vllm::ScalarTypeId s_id,
    int prob_m, int prob_n, int prob_k, int num_experts, int top_k,
    int tm_blocks, bool m8, int num_bits, int group_size,
    bool has_act_order, bool is_k_full, bool has_zp, bool is_zp_float,
    bool is_a8, int stages, int max_smem, int sms)
{
  exec_config_t best = {1, {-1,-1,-1}};
  thread_config_t* cfgs  = (tm_blocks > 1) ? large_cfgs : small_cfgs;
  int n_cfgs = (tm_blocks > 1)
      ? (int)(sizeof(large_cfgs)/sizeof(thread_config_t))
      : (int)(sizeof(small_cfgs)/sizeof(thread_config_t));
  int count = 0;
  constexpr int dev_regs = 255*1024;

  for (int i = 0; i < n_cfgs; i++) {
    auto& th = cfgs[i];
    if (!valid_config(th, tm_blocks, prob_m, prob_n, prob_k, num_bits,
                      group_size, has_act_order, is_k_full, has_zp,
                      is_zp_float, is_a8, stages, max_smem - 512)) continue;

    int cs = kernel_cache_size(th, tm_blocks, num_bits, group_size,
                               has_act_order, is_k_full, has_zp,
                               is_zp_float, is_a8, stages);
    int gb = has_act_order ? 0 : (group_size == -1 ? -1 : group_size / 16);

    auto k = get_marlin_kernel(a_id, b_id, c_id, s_id,
                               th.num_threads, tm_blocks,
                               th.thread_n/16, th.thread_k/16,
                               m8, stages, gb, is_zp_float);
    if (k == nullptr) continue;  // not instantiated, skip

    int rs    = get_kernel_regs(k) * th.num_threads * 4;
    int allow = min(dev_regs/rs, max_smem/(cs+1536));
    allow = (tm_blocks==1) ? max(min(allow,4),1) : max(min(allow,2),1);
    if (prob_n/th.thread_n*prob_m*top_k*4 < sms*allow)
      allow = max(prob_n/th.thread_n*prob_m*top_k*4/sms, 1);
    if (allow > count) { count = allow; best = {count, th}; }
  }
  return best;
}

// =============================================================================
// marlin_mm – core dispatch
// =============================================================================

void marlin_mm(
    const void* A, const void* B, void* C, void* C_tmp,
    void* b_bias, void* a_s, void* b_s, void* g_s, void* zp,
    void* g_idx, void* perm, void* a_tmp,
    void* sorted_ids, void* expert_ids, void* ntpp, void* topk_weights,
    int moe_block_size, int num_experts, int top_k, bool mul_topk_weights,
    int prob_m, int prob_n, int prob_k, void* workspace,
    vllm::ScalarTypeId a_id, vllm::ScalarTypeId b_id,
    vllm::ScalarTypeId c_id, vllm::ScalarTypeId s_id,
    bool has_bias, bool has_act_order, bool is_k_full, bool has_zp,
    int num_groups, int group_size, int dev, cudaStream_t stream,
    int thread_k, int thread_n, int sms, int blocks_per_sm,
    bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float)
{
  const int  tm_blocks = div_ceil(moe_block_size, 16);
  const bool m8        = (moe_block_size == 8);
  const int  num_bits  = vllm::ScalarType::from_id(b_id).size_bits();
  const bool is_a8     = vllm::ScalarType::from_id(a_id).size_bits() == 8;

  TORCH_CHECK(prob_m>0 && prob_n>0 && prob_k>0);

  int gb = 0;
  if (has_act_order) {
    if (is_k_full) { TORCH_CHECK(group_size!=-1); gb=group_size/16; TORCH_CHECK(prob_k%gb==0); }
    else { TORCH_CHECK(group_size==0); }
  } else {
    gb = (group_size==-1) ? -1 : group_size/16;
    if (gb>0) TORCH_CHECK(prob_k%gb==0);
  }

  const int4*    Ap  = (const int4*)A;
  const int4*    Bp  = (const int4*)B;
  int4*          Cp  = (int4*)C;
  int4*          Ctp = (int4*)C_tmp;
  const int4*    bp  = (const int4*)b_bias;
  const float*   asp = (const float*)a_s;
  const int4*    bsp = (const int4*)b_s;
  const uint16_t* gsp = (const uint16_t*)g_s;
  const int4*    zpp = (const int4*)zp;
  const int*     gip = (const int*)g_idx;
  const int*     pp  = (const int*)perm;
  int4*          atp = (int4*)a_tmp;
  const int32_t* sip = (const int32_t*)sorted_ids;
  const int32_t* eip = (const int32_t*)expert_ids;
  const int32_t* ntp = (const int32_t*)ntpp;
  const float*   twp = (const float*)topk_weights;
  int*           lks = (int*)workspace;

  if (has_act_order) {
    auto pckernel = permute_cols_kernel<8>;
    if      (moe_block_size==16) pckernel=permute_cols_kernel<16>;
    else if (moe_block_size==32) pckernel=permute_cols_kernel<32>;
    else if (moe_block_size==48) pckernel=permute_cols_kernel<48>;
    else if (moe_block_size==64) pckernel=permute_cols_kernel<64>;
    else TORCH_CHECK(moe_block_size==8,"unsupported moe_block_size=",moe_block_size);
    // clang-format off
    pckernel<<<sms, default_threads, 0, stream>>>(Ap,pp,atp,sip,eip,ntp,prob_m,prob_k,top_k);
    // clang-format on
    Ap=atp; prob_m*=top_k; top_k=1;
    if (is_k_full) has_act_order=false;
  }

  const CachedDeviceProps& dp = get_device_props(dev);
  int max_smem = dp.max_shared_mem;
  int stages   = dp.stages;

  exec_config_t   exec_cfg;
  thread_config_t th;

  if (thread_k != -1 && thread_n != -1) {
    th = {thread_k, thread_n, thread_k*thread_n/64};
    if (blocks_per_sm==-1) blocks_per_sm=1;
    exec_cfg = {blocks_per_sm, th};
  } else {
    ExecCfgKey key{prob_m,prob_n,prob_k,moe_block_size,(int)num_bits,
                   group_size,dev,has_act_order,is_k_full,has_zp,is_zp_float,is_a8};
    {
      std::lock_guard<std::mutex> lk(g_exec_mutex);
      auto it = g_exec_cache.find(key);
      if (it != g_exec_cache.end()) {
        exec_cfg = it->second;
      } else {
        exec_cfg = determine_exec_config(
            a_id, b_id, c_id, s_id,
            prob_m, prob_n, prob_k, num_experts, top_k,
            tm_blocks, m8, num_bits, group_size,
            has_act_order, is_k_full, has_zp, is_zp_float,
            is_a8, stages, max_smem, sms);
        g_exec_cache[key] = exec_cfg;
      }
    }
    th = exec_cfg.tb_cfg;
  }

  thread_k = th.thread_k; thread_n = th.thread_n;
  int blocks = sms * exec_cfg.blocks_per_sm;
  if (exec_cfg.blocks_per_sm > 1)
    max_smem = max_smem / exec_cfg.blocks_per_sm - 1024;

  TORCH_CHECK(valid_config(th, tm_blocks, prob_m, prob_n, prob_k, num_bits,
                           group_size, has_act_order, is_k_full, has_zp,
                           is_zp_float, is_a8, stages, max_smem),
              "No valid Marlin config for MNK=[",prob_m,",",prob_n,",",prob_k,
              "]. Add the required Marlin<> instantiation in marlin_moe_wna16.cu.");

  auto kernel = get_marlin_kernel(a_id, b_id, c_id, s_id,
                                  th.num_threads, tm_blocks,
                                  thread_n/16, thread_k/16,
                                  m8, stages, gb, is_zp_float);
  TORCH_CHECK(kernel != nullptr,
              "Marlin kernel not compiled for this config. "
              "Add the required instantiation in marlin_moe_wna16.cu.");

  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem);
  // clang-format off
  kernel<<<blocks, th.num_threads, max_smem, stream>>>(
      Ap, Bp, Cp, Ctp, bp, asp, bsp, gsp, zpp, gip,
      sip, eip, ntp, twp,
      top_k, mul_topk_weights, num_groups,
      prob_m, prob_n, prob_k,
      lks, has_bias, use_atomic_add, use_fp32_reduce);
  // clang-format on
}

}  // namespace marlin

// =============================================================================
// Public entry point – registered in bindings.cpp
// =============================================================================

torch::Tensor moe_wna16_marlin_gemm(
    torch::Tensor&                          a,
    std::optional<torch::Tensor>            c_or_none,
    std::optional<torch::Tensor>            c_tmp_or_none,
    torch::Tensor&                          b_q_weight,
    std::optional<torch::Tensor> const&     b_bias_or_none,
    torch::Tensor&                          b_scales,
    std::optional<torch::Tensor> const&     a_scales_or_none,
    std::optional<torch::Tensor> const&     global_scale_or_none,
    std::optional<torch::Tensor> const&     b_zeros_or_none,
    std::optional<torch::Tensor> const&     g_idx_or_none,
    std::optional<torch::Tensor> const&     perm_or_none,
    torch::Tensor&                          workspace,
    torch::Tensor&                          sorted_token_ids,
    torch::Tensor&                          expert_ids,
    torch::Tensor&                          num_tokens_past_padded,
    torch::Tensor&                          topk_weights,
    int64_t moe_block_size, int64_t top_k,  bool mul_topk_weights,
    int64_t b_type_id,
    int64_t size_m, int64_t size_n, int64_t size_k,
    bool    is_k_full       = true,
    bool    use_atomic_add  = false,
    bool    use_fp32_reduce = true,
    bool    is_zp_float     = false,
    int64_t thread_k        = -1,
    int64_t thread_n        = -1,
    int64_t blocks_per_sm   = -1)
{
  using namespace marlin;
  int dev = a.get_device();
  const marlin::CachedDeviceProps& dp =
      marlin::get_device_props(dev);
  int sms = dp.sms;

  // Derive a_type / c_type ids from tensor dtypes
  vllm::ScalarTypeId a_id, c_id, s_id;
  auto c_dtype = a.dtype();

  switch (a.scalar_type()) {
    case at::ScalarType::Half:
      a_id = FP16_ID; c_id = FP16_ID; break;
    case at::ScalarType::BFloat16:
      a_id = BF16_ID; c_id = BF16_ID; break;
    case at::ScalarType::Float8_e4m3fn: {
      a_id    = vllm::kFE4M3fn.id();
      c_dtype = b_scales.dtype();
      c_id    = (b_scales.scalar_type()==at::ScalarType::Half) ? FP16_ID : BF16_ID;
      int major, minor;
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
      TORCH_CHECK(major*10+minor>=89,"FP8 requires Ada Lovelace (sm89)+");
      break;
    }
    case at::ScalarType::Char:
      a_id    = vllm::kS8.id();
      c_dtype = b_scales.dtype();
      c_id    = (b_scales.scalar_type()==at::ScalarType::Half) ? FP16_ID : BF16_ID;
      break;
    default:
      TORCH_CHECK(false,"Unsupported activation dtype: ",a.scalar_type());
  }

  s_id = c_id;
  if (b_type_id == vllm::kFE2M1f.id()) {
    if      (b_scales.scalar_type()==at::ScalarType::Float8_e4m3fn)
      s_id = vllm::kFE4M3fn.id();
    else if (b_scales.scalar_type()==at::ScalarType::Float8_e8m0fnu)
      s_id = vllm::kFE8M0fnu.id();
    else TORCH_CHECK(false,"FP4: scales must be fp8_e4m3fn or fp8_e8m0fnu.");
  }

  const int pack_factor = 32 / vllm::ScalarType::from_id(b_type_id).size_bits();
  const int num_experts = b_q_weight.size(0);

  if (moe_block_size != 8) {
    TORCH_CHECK(moe_block_size%16==0 && moe_block_size>=16 && moe_block_size<=64,
                "unsupported moe_block_size=", moe_block_size);
  }

  TORCH_CHECK(a.size(0)==size_m && a.size(1)==size_k);
  TORCH_CHECK(size_k % marlin::tile_size == 0);
  TORCH_CHECK((size_k/marlin::tile_size)==b_q_weight.size(1));
  TORCH_CHECK(b_q_weight.size(2)%marlin::tile_size==0);
  TORCH_CHECK(size_n==(b_q_weight.size(2)/marlin::tile_size)*pack_factor,
              "size_n mismatch");
  TORCH_CHECK(a.device().is_cuda() && a.is_contiguous());
  TORCH_CHECK(b_q_weight.device().is_cuda() && b_q_weight.is_contiguous());
  TORCH_CHECK(b_scales.device().is_cuda()   && b_scales.is_contiguous());

  auto opts      = torch::TensorOptions().dtype(c_dtype).device(a.device());
  auto opts_fp32 = torch::TensorOptions().dtype(at::kFloat).device(a.device());

  torch::Tensor a_scales;
  if (a_scales_or_none.has_value()) {
    a_scales = a_scales_or_none.value();
    TORCH_CHECK(vllm::ScalarType::from_id(a_id).size_bits()==8);
  } else {
    a_scales = torch::empty({0}, opts_fp32);
    TORCH_CHECK(vllm::ScalarType::from_id(a_id).size_bits()!=8);
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));

  torch::Tensor c;
  if (c_or_none.has_value()) {
    c = c_or_none.value();
    TORCH_CHECK(c.device().is_cuda() && c.is_contiguous());
    TORCH_CHECK(c.size(0)==size_m*top_k && c.size(1)==size_n);
  } else {
    c = torch::empty({size_m*top_k, size_n}, opts);
  }

  torch::Tensor c_tmp;
  if (use_fp32_reduce && !use_atomic_add) {
    long max_ctmp = std::min(
        (long)size_n * sorted_token_ids.size(0),
        (long)sms * 4 * moe_block_size * marlin::max_thread_n);
    if (moe_block_size==8) max_ctmp*=2;
    if (c_tmp_or_none.has_value()) {
      c_tmp = c_tmp_or_none.value();
      TORCH_CHECK(c_tmp.numel()>=max_ctmp,
                  "c_tmp too small: need ",max_ctmp," got ",c_tmp.numel());
      c_tmp = c_tmp.narrow(0,0,max_ctmp);
    } else {
      c_tmp = torch::empty({max_ctmp}, opts_fp32);
    }
  } else {
    c_tmp = torch::empty({0}, opts_fp32);
  }

  TORCH_CHECK(b_scales.sizes().size()==3 && b_scales.size(2)==size_n);
  int num_groups = b_scales.size(1);
  int group_size = -1;

  torch::Tensor g_idx, perm, a_tmp;
  bool has_act_order = false;
  if (g_idx_or_none.has_value() && perm_or_none.has_value()) {
    g_idx = g_idx_or_none.value();
    perm  = perm_or_none.value();
    TORCH_CHECK(g_idx.device().is_cuda() && g_idx.is_contiguous());
    TORCH_CHECK(perm.device().is_cuda()  && perm.is_contiguous());
    has_act_order = g_idx.size(-1)>0 && perm.size(-1)>0;
  } else {
    g_idx = perm = torch::empty({0}, opts);
  }

  if (has_act_order) {
    a_tmp = torch::empty({size_m*top_k, size_k}, opts);
    if (is_k_full) {
      TORCH_CHECK(num_groups>1 && size_k%num_groups==0);
      group_size = size_k/num_groups;
    } else { group_size=0; }
  } else {
    a_tmp = torch::empty({0}, opts);
    if (num_groups>1) { TORCH_CHECK(size_k%num_groups==0); group_size=size_k/num_groups; }
  }

  torch::Tensor global_scale;
  if (global_scale_or_none.has_value()) {
    global_scale = global_scale_or_none.value();
    TORCH_CHECK(b_type_id==vllm::kFE2M1f.id() && s_id==vllm::kFE4M3fn.id());
  } else {
    global_scale = torch::empty({0}, opts);
    TORCH_CHECK(!(b_type_id==vllm::kFE2M1f.id() && s_id==vllm::kFE4M3fn.id()));
  }

  bool has_bias = b_bias_or_none.has_value();
  torch::Tensor b_bias;
  if (has_bias) {
    b_bias = b_bias_or_none.value();
    TORCH_CHECK(b_bias.device().is_cuda() && b_bias.is_contiguous());
    TORCH_CHECK(b_bias.size(1)==size_n && b_bias.stride(1)==1);
  } else { b_bias = torch::empty({0}, opts); }

  torch::Tensor b_zeros;
  if (b_zeros_or_none.has_value()) {
    b_zeros = b_zeros_or_none.value();
    TORCH_CHECK(b_zeros.device().is_cuda() && b_zeros.is_contiguous());
  } else { b_zeros = torch::empty({0}, opts); }
  bool has_zp = b_zeros.size(-1)>0;

  int max_n_tiles = size_n / marlin::min_thread_n;
  int min_ws = std::min(max_n_tiles*(int)(sorted_token_ids.size(0)/moe_block_size), sms*4);
  TORCH_CHECK(workspace.numel()>=min_ws,
              "workspace too small: need ",min_ws," got ",workspace.numel());

  marlin::marlin_mm(
      a.data_ptr(),           b_q_weight.data_ptr(),
      c.data_ptr(),           c_tmp.data_ptr(),
      b_bias.data_ptr(),      a_scales.data_ptr(),
      b_scales.data_ptr(),    global_scale.data_ptr(),
      b_zeros.data_ptr(),     g_idx.data_ptr(),
      perm.data_ptr(),        a_tmp.data_ptr(),
      sorted_token_ids.data_ptr(),
      expert_ids.data_ptr(),
      num_tokens_past_padded.data_ptr(),
      topk_weights.data_ptr(),
      moe_block_size, num_experts, top_k, mul_topk_weights,
      size_m, size_n, size_k, workspace.data_ptr(),
      a_id, b_type_id, c_id, s_id,
      has_bias, has_act_order, is_k_full, has_zp,
      num_groups, group_size, dev,
      at::cuda::getCurrentCUDAStream(dev),
      thread_k, thread_n, sms, blocks_per_sm,
      use_atomic_add, use_fp32_reduce, is_zp_float);

  return c;
}
