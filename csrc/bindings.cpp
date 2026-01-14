#include <torch/extension.h>

// Forward declarations
void rms_norm(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight, 
                       double epsilon, bool gemma);

// Forward declarations - Rotary Embedding
void rotary_embedding(torch::Tensor& positions, torch::Tensor& query, 
                      torch::Tensor& key, int64_t head_size,
                      torch::Tensor& cos_sin_cache, bool is_neox);

// Forward declarations - Activation Kernels
void silu_and_mul(torch::Tensor& out, torch::Tensor& input);
void gelu_and_mul(torch::Tensor& out, torch::Tensor& input);
void gelu_tanh_and_mul(torch::Tensor& out, torch::Tensor& input);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
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


}

