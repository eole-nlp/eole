#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

namespace eole {

/**
 * @brief CUDA kernel to apply rotary positional embeddings to query and key tensors.
 * 
 * Initially inspired from vLLM implementation, further optimized
 *
 * Rotary embeddings rotate pairs of hidden dimensions according to a precomputed
 * cosine/sine cache, allowing the model to incorporate positional information
 * into attention computations efficiently.
 * 
 * @tparam scalar_t Type of the query/key elements (float, half, bfloat16)
 * @tparam IS_NEOX   Flag indicating whether to use the NeoX layout (splits rot dim differently)
 * 
 * @param positions        Array of token positions (length = batch seq length)
 * @param query            Query tensor to be rotated, shape: [num_tokens, num_heads, head_size] (flattened)
 * @param key              Optional key tensor to be rotated, same shape as query
 * @param cos_sin_cache    Precomputed cos/sin values, shape: [max_pos, rot_dim * 2]
 * @param rot_dim          Number of dimensions for rotation (half of embedding size for NeoX)
 * @param query_stride     Stride to access sequential tokens in query tensor
 * @param key_stride       Stride to access sequential tokens in key tensor
 * @param head_stride      Stride to access each head in query/key tensor
 * @param num_heads        Number of attention heads in query
 * @param num_kv_heads     Number of attention heads in key (can differ for cross-attention)
 * @param embed_dim        Dimension of each head embedding (rot_dim / 2 for NeoX)
 */
template <typename scalar_t, bool IS_NEOX>
__global__ void rotary_embedding_kernel(
    const int64_t* __restrict__ positions,
    scalar_t* __restrict__ query,           
    scalar_t* __restrict__ key,             
    const scalar_t* __restrict__ cos_sin_cache,
    const int rot_dim,
    const int64_t query_stride,
    const int64_t key_stride,
    const int64_t head_stride,
    const int num_heads,
    const int num_kv_heads,
    const int embed_dim) {

    // Each block handles one token
    const int token_idx = blockIdx.x;
    const int64_t pos = positions[token_idx];

    // ----------------------------
    // 1. Allocate shared memory for cos/sin
    // ----------------------------
    extern __shared__ char shared_storage[];
    scalar_t* s_cos = reinterpret_cast<scalar_t*>(shared_storage);
    scalar_t* s_sin = s_cos + embed_dim;

    // Pointers into global cos/sin cache for this position
    const scalar_t* g_cos_ptr = cos_sin_cache + pos * rot_dim;
    const scalar_t* g_sin_ptr = g_cos_ptr + embed_dim;

    // ----------------------------
    // 2. Load cos/sin values into shared memory (cooperative loading)
    // ----------------------------
    for (int i = threadIdx.x; i < embed_dim; i += blockDim.x) {
        s_cos[i] = g_cos_ptr[i];
        s_sin[i] = g_sin_ptr[i];
    }
    __syncthreads();

    // ----------------------------
    // 3. Rotate Query tensor
    // ----------------------------
    const int nq = num_heads * embed_dim; // total elements per token
    for (int i = threadIdx.x; i < nq; i += blockDim.x) {
        const int head_idx = i / embed_dim;
        const int rot_offset = i % embed_dim;
        const int64_t token_head = token_idx * query_stride + head_idx * head_stride;

        int x_index, y_index;
        if (IS_NEOX) {
            // NeoX layout: first half of head dims rotated with second half
            x_index = rot_offset;
            y_index = embed_dim + rot_offset;
        } else {
            // Standard layout: pairs of consecutive dims
            x_index = 2 * rot_offset;
            y_index = 2 * rot_offset + 1;
        }

        const scalar_t cos = s_cos[rot_offset];
        const scalar_t sin = s_sin[rot_offset];

        const scalar_t x = query[token_head + x_index];
        const scalar_t y = query[token_head + y_index];

        // Apply rotation: [x', y'] = [x * cos - y * sin, y * cos + x * sin]
        query[token_head + x_index] = x * cos - y * sin;
        query[token_head + y_index] = y * cos + x * sin;
    }

    // ----------------------------
    // 4. Rotate Key tensor (if provided)
    // ----------------------------
    if (key != nullptr) {
        const int nk = num_kv_heads * embed_dim;
        for (int i = threadIdx.x; i < nk; i += blockDim.x) {
            const int head_idx = i / embed_dim;
            const int rot_offset = i % embed_dim;
            const int64_t token_head = token_idx * key_stride + head_idx * head_stride;

            int x_index, y_index;
            if (IS_NEOX) {
                x_index = rot_offset;
                y_index = embed_dim + rot_offset;
            } else {
                x_index = 2 * rot_offset;
                y_index = 2 * rot_offset + 1;
            }

            const scalar_t cos = s_cos[rot_offset];
            const scalar_t sin = s_sin[rot_offset];

            const scalar_t x = key[token_head + x_index];
            const scalar_t y = key[token_head + y_index];

            key[token_head + x_index] = x * cos - y * sin;
            key[token_head + y_index] = y * cos + x * sin;
        }
    }
}

} // namespace eole

/**
 * @brief Entry point for applying rotary embeddings to query and key tensors.
 * 
 * This function calculates grid/block dimensions, shared memory size, and launches
 * the CUDA kernel. Supports both standard and NeoX layouts.
 * 
 * @param positions      1D tensor of token positions
 * @param query          Query tensor of shape [num_tokens, num_heads, head_size]
 * @param key            Optional key tensor (can be empty)
 * @param head_size      Size of each attention head
 * @param cos_sin_cache  Precomputed cos/sin tensor of shape [max_pos, rot_dim * 2]
 * @param is_neox        Whether to use NeoX layout (true/false)
 */
void rotary_embedding(
    torch::Tensor& positions,
    torch::Tensor& query,
    torch::Tensor& key,
    int64_t head_size,
    torch::Tensor& cos_sin_cache,
    bool is_neox) {

    int64_t num_tokens = positions.numel();
    int positions_ndim = positions.dim();
    int rot_dim = cos_sin_cache.size(1);
    int embed_dim = rot_dim / 2;

    int query_hidden_size = query.numel() / num_tokens;
    int num_heads = query_hidden_size / head_size;
    int num_kv_heads = (key.numel() > 0) ? (key.numel() / num_tokens) / head_size : 0;

    int seq_dim_idx = positions_ndim - 1;
    int64_t query_stride = query.stride(seq_dim_idx);
    int64_t key_stride = (key.numel() > 0) ? key.stride(seq_dim_idx) : 0;
    int64_t head_stride = (query.dim() == positions_ndim + 2) ? query.stride(-2) : head_size;

    // Launch one block per token
    dim3 grid(num_tokens);
    // Limit threads per block to 512 for most GPUs
    dim3 block(std::min<int>(num_heads * embed_dim, 512));

    // Shared memory size: cos + sin per embedding dimension
    int shared_mem_size = rot_dim * query.element_size();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // Dispatch kernel for float, half, or bfloat16
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        query.scalar_type(), "rotary_embedding", [&] {
            scalar_t* key_ptr = (key.numel() > 0) ? key.data_ptr<scalar_t>() : nullptr;

            if (is_neox) {
                eole::rotary_embedding_kernel<scalar_t, true><<<grid, block, shared_mem_size, stream>>>(
                    positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(), key_ptr,
                    cos_sin_cache.data_ptr<scalar_t>(), rot_dim, query_stride, key_stride, 
                    head_stride, num_heads, num_kv_heads, embed_dim);
            } else {
                eole::rotary_embedding_kernel<scalar_t, false><<<grid, block, shared_mem_size, stream>>>(
                    positions.data_ptr<int64_t>(), query.data_ptr<scalar_t>(), key_ptr,
                    cos_sin_cache.data_ptr<scalar_t>(), rot_dim, query_stride, key_stride, 
                    head_stride, num_heads, num_kv_heads, embed_dim);
            }
        });
}

