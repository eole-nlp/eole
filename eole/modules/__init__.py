"""eole.modules: model building blocks, grouped in subpackages:

- attention/: Attention backends, position encodings, KV caches (incl. FP8 KV and prefill cache).
- mlp/: Feed-forward blocks, embeddings, LoRA and auxiliary heads.
- moe/: Mixture-of-experts layer, executor interface/registry and expert placement/caching.
- quant/: Quantized linear backends (GPTQ/AWQ/INT8/FP8/NVFP4/GGUF) and quantization selection.
- speculation/: Speculative-decoding drafters (native MTP, DFlash2) and their graph adapters.
- gdn/: Gated delta net (linear attention) layer and Triton ABI glue.
- legacy/: Training-era components kept for RNN/CNN/scorer paths (StackedRNN, weight norm, sparsemax).

Import from the subpackage directly, e.g. eole.modules.moe.moe.
"""
