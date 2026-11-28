# eole.modules — layout and layering

Model building blocks, grouped by concern. Import from the subpackage
directly (e.g. `from eole.modules.moe.moe import MoE`).

```
eole/modules/
├── attention/     multi_headed_attn, rope, rmsnorm, relative/alibi_position_bias,
│                  global_attention, conv_multi_step_attention, fp8_kv, prefill_cache
├── mlp/           transformer_mlp, embeddings, estimator, representation, contextgate, lora
├── moe/           moe (layer), expert_execution (ExpertExecutor interface + registry),
│                  moe_quant_utils, fused_moe_marlin, cached_experts, expert_cache,
│                  host_experts, hybrid_experts, hybrid_graph, expert_placement
├── quant/         autoround/awq/bnb/gguf/compressed_float_linear, marlin_*, int8_gemv,
│                  dense_quantization, quantization_loading, quantization_selection,
│                  vocabulary_quantization, and the QUANT_BACKENDS registry (__init__.py)
├── speculation/   mtp, dflash2, dflash2_graph (draft heads + CUDA-graph adapters)
├── gdn/           gated_delta_net.py (facade), kernels.py (fast paths + PyTorch
│                  reference kernels + dispatch + gated RMSNorm), layer.py (the
│                  GatedDeltaNet module), gdn_ab.py (Triton ABI glue)
└── legacy/        stacked_rnn, weight_norm, sparse_activations, sparse_losses
```

## Layering rules

Data flows through a decoder layer roughly as:

```
attention/ (self/cross attention, KV caches)  →  mlp/ or moe/ (feed-forward)
                                               →  speculation/ (draft/verify, wraps the decoder)
gdn/ (linear attention) replaces attention/ in hybrid layers
quant/ swaps nn.Linear for a quantized backend at load time (registry in quant/__init__.py)
```

- **attention/, mlp/, gdn/** are the compute primitives — no checkpoint-format knowledge.
- **moe/** owns routing *and* execution: `MoE` routes and combines; an `ExpertExecutor`
  backend (registry: `EXPERT_EXECUTORS` in `moe/expert_execution.py`, keyed by
  `moe_expert_backend`) runs the routed experts. Add a backend by registering a factory.
- **quant/** owns weight-format replacement, selected by `quant_type` through
  `QUANT_BACKENDS` in `quant/__init__.py` (`apply_quant_backend`, called from
  `Model.maybe_quantize`). Add a format by registering a handler.
- **speculation/** wraps a whole decoder (draft → target verify); it is not a layer.
- **legacy/** is kept for the RNN/CNN/scorer training paths; new inference code
  should not add files here.

## modules/ vs utils/

- `eole/modules/` = things that live **inside a model** (nn.Modules, kernels called
  by layers, executor/quant registries used at load time).
- `eole/utils/` = **runtime orchestration around models**: CUDA-graph capture
  (`decoder_graph.py`), verifier/request caches, tensor-parallel and distributed
  helpers, logging. Notably `utils/decoder_graph.py` orchestrates graph
  capture/replay for the decoder, while `modules/moe/hybrid_graph.py` holds the
  per-layer graph pools used by hybrid expert execution — same "graph" word,
  different layers; keep new graph code on the side that owns the captured call.

When adding a file: if a layer or the loader calls it, it belongs here (pick the
subpackage by concern); if the engine/server/scheduler calls it around a model,
it belongs in `eole/utils/`.
