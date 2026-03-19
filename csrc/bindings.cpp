#include <torch/extension.h>
#include <optional>

// ── Forward declarations ──────────────────────────────────────────────────────

void rms_norm(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight,
              double epsilon, bool gemma);

void rotary_embedding(torch::Tensor& positions, torch::Tensor& query,
                      torch::Tensor& key, int64_t head_size,
                      torch::Tensor& cos_sin_cache, bool is_neox);

void silu_and_mul(torch::Tensor& out, torch::Tensor& input);
void gelu_and_mul(torch::Tensor& out, torch::Tensor& input);
void gelu_tanh_and_mul(torch::Tensor& out, torch::Tensor& input);

// Marlin MoE routing helper (moe_align.cu)
void moe_align_block_size(
    torch::Tensor  topk_ids,
    int64_t        num_experts,
    int64_t        block_size,
    torch::Tensor  sorted_token_ids,
    torch::Tensor  expert_ids,
    torch::Tensor  num_tokens_post_padded);

torch::Tensor gptq_marlin_repack(
    torch::Tensor& b_q_weight,
    torch::Tensor& perm,
    int64_t        size_k,
    int64_t        size_n,
    int64_t        num_bits);

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
    bool                                    is_zp_float);

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
    int64_t moe_block_size,
    int64_t top_k,
    bool    mul_topk_weights,
    int64_t b_type_id,
    int64_t size_m,
    int64_t size_n,
    int64_t size_k,
    bool    is_k_full,
    bool    use_atomic_add,
    bool    use_fp32_reduce,
    bool    is_zp_float,
    int64_t thread_k,
    int64_t thread_n,
    int64_t blocks_per_sm);


// ── Module ────────────────────────────────────────────────────────────────────

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  // RMS Norm
  m.def("rms_norm", &rms_norm,
        "RMS Normalization",
        py::arg("out"),
        py::arg("input"),
        py::arg("weight"),
        py::arg("epsilon"),
        py::arg("gemma") = false);

  // Rotary Embedding
  m.def("rotary_embedding", &rotary_embedding,
        "Apply Rotary Position Embedding",
        py::arg("positions"),
        py::arg("query"),
        py::arg("key"),
        py::arg("head_size"),
        py::arg("cos_sin_cache"),
        py::arg("is_neox") = true);

  // Activation Kernels
  m.def("silu_and_mul", &silu_and_mul,
        "SiLU activation and gating: out = silu(x[:, :d]) * x[:, d:]",
        py::arg("out"),
        py::arg("input"));

  m.def("gelu_and_mul", &gelu_and_mul,
        "GELU activation and gating: out = gelu(x[:, :d]) * x[:, d:]",
        py::arg("out"),
        py::arg("input"));

  m.def("gelu_tanh_and_mul", &gelu_tanh_and_mul,
        "GELU-tanh activation and gating: out = gelu_tanh(x[:, :d]) * x[:, d:]",
        py::arg("out"),
        py::arg("input"));

  // MoE routing helper (always compiled – no MARLIN guard needed)
  m.def("moe_align_block_size", &moe_align_block_size,
        "Build sorted_token_ids / expert_ids routing tensors for Marlin MoE",
        py::arg("topk_ids"),
        py::arg("num_experts"),
        py::arg("block_size"),
        py::arg("sorted_token_ids"),
        py::arg("expert_ids"),
        py::arg("num_tokens_post_padded"));

  // GPTQ Marlin weight repack
  m.def("gptq_marlin_repack", &gptq_marlin_repack,
        "Repack GPTQ int4/int8 weights into Marlin tiled layout",
        py::arg("b_q_weight"),
        py::arg("perm"),
        py::arg("size_k"),
        py::arg("size_n"),
        py::arg("num_bits"));

  // Dense Marlin GEMM (gptqmodel-compatible, no MoE routing overhead)
  m.def("gptq_marlin_gemm", &gptq_marlin_gemm,
        "Dense Marlin GEMM for GPTQ-quantized weights",
        py::arg("a"),
        py::arg("c") = py::none(),
        py::arg("b_q_weight"),
        py::arg("b_bias") = py::none(),
        py::arg("b_scales"),
        py::arg("global_scale") = py::none(),
        py::arg("b_zeros") = py::none(),
        py::arg("g_idx") = py::none(),
        py::arg("perm") = py::none(),
        py::arg("workspace"),
        py::arg("b_q_type_id"),
        py::arg("size_m"),
        py::arg("size_n"),
        py::arg("size_k"),
        py::arg("is_k_full") = true,
        py::arg("use_atomic_add") = false,
        py::arg("use_fp32_reduce") = true,
        py::arg("is_zp_float") = false);

  // MoE WNA16 Marlin GEMM
  // b_type_id : plain integer (4=uint4b8, 5=uint8b128); vllm::ScalarType is
  //             reconstructed inside the C++ kernel, never visible in Python.
  // c_tmp     : optional pre-allocated float32 scratch buffer that eliminates
  //             the largest per-call GPU allocation inside the kernel.
  m.def("moe_wna16_marlin_gemm", &moe_wna16_marlin_gemm,
        "Fused MoE WNA16 GEMM using the Marlin kernel",
        py::arg("a"),
        py::arg("c")                     = py::none(),
        py::arg("c_tmp")                 = py::none(),
        py::arg("b_q_weight"),
        py::arg("b_bias")                = py::none(),
        py::arg("b_scales"),
        py::arg("b_act_input")           = py::none(),
        py::arg("b_global_scale")        = py::none(),
        py::arg("b_zeros")               = py::none(),
        py::arg("g_idx")                 = py::none(),
        py::arg("perm")                  = py::none(),
        py::arg("workspace"),
        py::arg("sorted_token_ids"),
        py::arg("expert_ids"),
        py::arg("num_tokens_past_padded"),
        py::arg("topk_weights"),
        py::arg("moe_block_size"),
        py::arg("top_k"),
        py::arg("mul_topk_weights"),
        py::arg("b_type_id"),
        py::arg("size_m"),
        py::arg("size_n"),
        py::arg("size_k"),
        py::arg("is_k_full")             = true,
        py::arg("use_atomic_add")        = false,
        py::arg("use_fp32_reduce")       = true,
        py::arg("is_zp_float")           = false,
        py::arg("thread_k")              = -1,
        py::arg("thread_n")              = -1,
        py::arg("blocks_per_sm")         = -1);
}
