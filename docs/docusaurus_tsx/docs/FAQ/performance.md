# Performance tips

## Training

* Use `compute_dtype: bf16` or `fp16` for mixed precision training
* Use `batch_size_multiple: 8`
* Use `vocab_size_multiple: 8`
* Depending on the number of GPUs, use `num_workers: 4` (for 1 GPU) or `num_workers: 2` (for multiple GPUs)
* To avoid averaging checkpoints you can use the "during training" average decay system
* For pure BF16 training (lower memory footprint), use `compute_dtype: bf16` with `use_amp: false` and `optim: adamw` — this leverages [Kahan Summation](https://optimi.benjaminwarner.dev/kahan_summation/) for stable updates

## Inference

* Use Flash Attention for fast attention computation (install with `pip install flash-attn --no-build-isolation`)
* Enable `torch.compile` for maximum inference speed — set `EOLE_TORCH_COMPILE=1` (or `EOLE_COMPILE_MODE=2/3` for CUDA graph capture). See [TORCHCOMPILE_README](https://github.com/eole-nlp/eole/blob/main/TORCHCOMPILE_README.md) for details
* Use quantization (`quant_type: bnb_NF4` or `awq_gemm`) to reduce VRAM usage
* For tensor parallel inference across multiple GPUs, set `parallel_mode: tensor_parallel` with appropriate `world_size` and `gpu_ranks`
