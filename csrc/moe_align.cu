/*
 * Eole-NLP – moe_align_block_size CUDA kernel
 * Adapted from vLLM's csrc/moe/moe_ops.cu (Apache-2.0).
 *
 * Builds the three routing tensors required by the Marlin MoE GEMM kernel:
 *
 *   sorted_token_ids      : (num_tokens_post_padded,) int32
 *       For each padded block slot: the flat token index that belongs here,
 *       or sentinel = M*topk for padding slots.
 *
 *   expert_ids            : (num_blocks,) int32
 *       Which expert handles each block of block_size consecutive slots.
 *
 *   num_tokens_post_padded: (1,) int32
 *       Total number of slots (= sum of per-expert padded counts).
 *
 * Algorithm (three CUDA kernel launches):
 *   1. moe_count_kernel        – one atomic per token: count[expert]++
 *   2. moe_prefix_sum (device) – exclusive prefix sum of padded counts
 *                                (called from host via thrust or scan kernel)
 *   3. moe_fill_kernel         – each token CAS-reserves its slot;
 *                                padding slots filled with sentinel
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1: count tokens per expert
// ─────────────────────────────────────────────────────────────────────────────

__global__ void moe_count_kernel(
    const int32_t* __restrict__ topk_ids,   // (num_tokens,) flattened
    int32_t*       __restrict__ counts,     // (num_experts,) zero-initialised
    int32_t        num_tokens)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_tokens) {
    atomicAdd(&counts[topk_ids[tid]], 1);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2: compute cumulative start offsets and fill expert_ids
//
// Launched with num_experts threads (single block or grid).
// Each thread e:
//   - reads counts[e], pads it to a multiple of block_size
//   - writes padded_count[e] and prefix-sum start[e]
//   - fills expert_ids[start[e]/block_size ... (start[e]+padded)/block_size)
// We do a sequential prefix sum inside a single warp/block via shared memory.
// For typical num_experts values (≤ 256) a single block is fine.
// ─────────────────────────────────────────────────────────────────────────────

__global__ void moe_prefix_and_fill_kernel(
    const int32_t* __restrict__ counts,       // (num_experts,)
    int32_t*       __restrict__ start_offsets,// (num_experts,) output
    int32_t*       __restrict__ expert_ids,   // (max_blocks,)  output
    int32_t*       __restrict__ ntpp,         // (1,)           output: total padded
    int32_t        num_experts,
    int32_t        block_size)
{
  // Use shared memory for the prefix sum.
  // Supports up to 1024 experts (one thread per expert).
  extern __shared__ int32_t shmem[];  // (num_experts,)

  int e = threadIdx.x;
  if (e >= num_experts) return;

  int32_t cnt     = counts[e];
  int32_t padded  = ((cnt + block_size - 1) / block_size) * block_size;
  shmem[e]        = padded;
  __syncthreads();

  // Exclusive prefix sum (sequential, done by thread 0)
  if (e == 0) {
    int32_t running = 0;
    for (int i = 0; i < num_experts; i++) {
      int32_t v   = shmem[i];
      shmem[i]    = running;   // store exclusive prefix
      running    += v;
    }
    ntpp[0] = running;         // total padded tokens
  }
  __syncthreads();

  int32_t off    = shmem[e];   // exclusive start offset for expert e
  int32_t padded2 = ((counts[e] + block_size - 1) / block_size) * block_size;
  start_offsets[e] = off;

  // Fill expert_ids for this expert's blocks
  int32_t n_blocks = padded2 / block_size;
  int32_t blk_start = off / block_size;
  for (int b = 0; b < n_blocks; b++) {
    expert_ids[blk_start + b] = e;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 3: fill sorted_token_ids
//
// Each token atomically claims a slot in its expert's region.
// Remaining (padding) slots were pre-filled with sentinel by the host.
// ─────────────────────────────────────────────────────────────────────────────

__global__ void moe_fill_kernel(
    const int32_t* __restrict__ topk_ids,      // (num_tokens,) flat
    const int32_t* __restrict__ start_offsets, // (num_experts,)
    int32_t*       __restrict__ write_pos,     // (num_experts,) atomic cursors
    int32_t*       __restrict__ sorted_ids,    // (total_padded,)
    int32_t        num_tokens)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_tokens) return;

  int32_t expert = topk_ids[tid];
  // Atomically grab the next write slot for this expert
  int32_t slot = atomicAdd(&write_pos[expert], 1);
  sorted_ids[slot] = tid;
}

// ─────────────────────────────────────────────────────────────────────────────
// Host-side entry point
// ─────────────────────────────────────────────────────────────────────────────

void moe_align_block_size(
    torch::Tensor  topk_ids,               // (M * topk,) int32 or int64
    int64_t        num_experts,
    int64_t        block_size,
    torch::Tensor  sorted_token_ids,       // pre-allocated output
    torch::Tensor  expert_ids,             // pre-allocated output
    torch::Tensor  num_tokens_post_padded) // pre-allocated output (1,)
{
  const at::cuda::OptionalCUDAGuard guard(topk_ids.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  // Flatten and ensure int32
  auto flat = topk_ids.flatten();
  if (flat.scalar_type() != at::ScalarType::Int) {
    flat = flat.to(at::ScalarType::Int);
  }
  int32_t num_tokens   = (int32_t)flat.numel();
  int32_t n_experts    = (int32_t)num_experts;
  int32_t blk          = (int32_t)block_size;
  int32_t sentinel     = num_tokens;   // padding value

  TORCH_CHECK(sorted_token_ids.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(expert_ids.scalar_type()       == at::ScalarType::Int);
  TORCH_CHECK(num_tokens_post_padded.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(n_experts <= 1024,
              "moe_align_block_size: num_experts=", n_experts,
              " exceeds kernel limit of 1024.");

  auto opts32  = torch::TensorOptions()
                    .dtype(at::ScalarType::Int)
                    .device(flat.device());

  // Workspace tensors (small, allocated once per call)
  auto counts       = torch::zeros({n_experts}, opts32);
  auto start_offs   = torch::empty({n_experts}, opts32);
  auto write_cursors = torch::empty({n_experts}, opts32);

  // Pre-fill sorted_token_ids with sentinel
  sorted_token_ids.fill_(sentinel);

  // ── Kernel 1: count ──────────────────────────────────────────────────────
  {
    int threads = 256;
    int blocks  = (num_tokens + threads - 1) / threads;
    if (blocks > 0) {
      moe_count_kernel<<<blocks, threads, 0, stream>>>(
          flat.data_ptr<int32_t>(),
          counts.data_ptr<int32_t>(),
          num_tokens);
    }
  }

  // ── Kernel 2: prefix sum + expert_ids fill ───────────────────────────────
  {
    // Single block, one thread per expert, shared mem = num_experts * 4 bytes
    int smem = n_experts * sizeof(int32_t);
    moe_prefix_and_fill_kernel<<<1, n_experts, smem, stream>>>(
        counts.data_ptr<int32_t>(),
        start_offs.data_ptr<int32_t>(),
        expert_ids.data_ptr<int32_t>(),
        num_tokens_post_padded.data_ptr<int32_t>(),
        n_experts,
        blk);
  }

  // ── Kernel 3: fill sorted_token_ids ─────────────────────────────────────
  // Initialise write cursors to the start offset of each expert
  write_cursors.copy_(start_offs);
  {
    int threads = 256;
    int blocks  = (num_tokens + threads - 1) / threads;
    if (blocks > 0) {
      moe_fill_kernel<<<blocks, threads, 0, stream>>>(
          flat.data_ptr<int32_t>(),
          start_offs.data_ptr<int32_t>(),
          write_cursors.data_ptr<int32_t>(),
          sorted_token_ids.data_ptr<int32_t>(),
          num_tokens);
    }
  }
}
