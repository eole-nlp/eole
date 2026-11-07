# Qwen3.5: text, vision, streaming, and serving

Run from the repository root. Follow the [installation guide](../../README.md#installation)
for Eole CUDA kernels, FlashAttention, and `fla-core` in your CUDA-enabled environment.
The inference scripts use `Qwen/Qwen3.5-4B`; the server example uses a separate
27B INT4 checkpoint. These are different model paths, not interchangeable steps.

## Convert and run the 4B examples

```bash
export EOLE_MODEL_DIR="$PWD/models"
mkdir -p "$EOLE_MODEL_DIR"
eole convert HF --model_dir Qwen/Qwen3.5-4B \
  --output "$EOLE_MODEL_DIR/qwen3.5-4B"
```

The scripts select FlashAttention; install `flash-attn --no-build-isolation`, or
change `self_attn_backend="flash"` to `"pytorch"` to use the fallback backend.

```bash
python recipes/qwen35/test_inference.py   # image questions with bundled fixtures
python recipes/qwen35/test_inference2.py  # text generation
python recipes/qwen35/test_stream.py      # live text streaming
```

The first script prints one response per image in `eole/tests/data/images`.
The second prints a generated passage. Adjust GPU ranks, generation length, and
model paths in the scripts to fit your hardware.

## Serve a model

For the small-model starting point, use the [server recipe](../server/README.md).
`serve.yaml` in this directory is an advanced Qwen3.5-27B INT4 example: it
requires a converted checkpoint at `$EOLE_MODEL_DIR/qwen3.5-27B-int4`, quantization
kernels, and FlashAttention. Its large context/cache settings must be adjusted
to available VRAM; it is not a conversion or download recipe.

```bash
eole serve -c recipes/qwen35/serve.yaml --host 127.0.0.1
```

For Qwen3.8 MTP and a reproducible baseline comparison, use the
[Qwen3.8 recipe](../qwen38/README.md).

## MoE execution and the CPU/GPU prototype

`moe_expert_backend: resident` remains the default. It uses the existing fused
GPU expert kernels, with the same checkpoint keys and routing mathematics.
Expert kernel caches/dispatch now live in `eole/modules/expert_execution.py`;
`MoE` retains routing and shared-expert behavior. The loader asks a separate
placement policy where each module belongs instead of tying placement to the
checkpoint reader.

`moe_expert_backend: hybrid_int8` is an **experimental capacity-oriented path**:

- Start from an **unquantized converted checkpoint**, not the GPTQ/NF4 checkpoint.
- The first `moe_gpu_experts` experts in each layer stay on CUDA in BF16. Remaining
  experts stay on the CPU during loading and are packed to per-channel INT8
  weights. Source weights are released expert by expert; they are not staged
  through CUDA. CPU activations are BF16 using PyTorch's native CPU matrix kernel.
- All selected experts are evaluated. CPU and GPU results are merged in routed
  order with the original weights; no experts are pruned or skipped. Runtime
  weight quantization changes numerical results compared with the BF16 source.
- One shared CPU worker overlaps CPU expert work with resident GPU experts.
  This is a fixed partition, **not** an adaptive expert cache or SSD-offload engine.
- Currently requires one CUDA GPU, BF16 compute, gated-SiLU experts, ordinary
  decoding and `EOLE_TORCH_COMPILE=0`. MTP, DFlash2, tensor parallelism and whole
  decoder CUDA graphs are rejected for this backend. The default resident backend
  keeps its existing capabilities. No new extension build or dependency is needed.

Use a checkpoint converted with current Eole (older Qwen conversions may still
have an incorrect RoPE layout). For example:

```bash
export QWEN35_MOE_MODEL=/path/to/converted/qwen3.5-35B-A3B
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 EOLE_TORCH_COMPILE=0 \
  python recipes/qwen35/benchmark_moe.py \
  --model "$QWEN35_MOE_MODEL" --backend hybrid_int8 --gpu-experts 64 \
  --tokens 64 --repeats 3 --output /tmp/qwen35-hybrid.json
```

Alternatively use [predict-hybrid.yaml](predict-hybrid.yaml), whose conservative
16-expert default uses less VRAM. This is a count per layer, not a memory budget;
choose it for the checkpoint and available VRAM. CPU thread settings are explicit
process settings; Eole does not change global thread counts. Keep the original
checkpoint for reload/training; the CPU packs are inference-only runtime state.

### Validation on Qwen3.5-35B-A3B

RTX 5090 and RTX 4090, Ryzen 9 5900X (AVX2), 128 GB RAM; PyTorch 2.15.0/CUDA
13.2, Triton 3.9, FLA 0.6.0, FlashAttention 2.8.3, fusion enabled. Batch-one greedy,
4,096-token cache, binary-search prompt from `benchmark_moe.py`, discarded
16-token warmup, three measured repetitions. Decode timing excludes the first
output token and prefill.

The resident regression control uses `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4`, with its
unused MTP head disabled and quantization selection matching the stored tensors.
Fixed-token replay feeds both versions the same 128-token history (127 timed
positions), removing differences in generated text and expert locality:

| RTX 5090 resident execution | Before extraction tok/s | After tok/s |
| --- | ---: | ---: |
| Eager, fixed-token replay | 25.84 | 25.69 |
| Compile mode 0, fixed-token replay | 161.81 | 161.53 |

The measured differences are under 1%. No material slowdown was observed in
these controls; they are not a guarantee across all architectures or workloads.
The replay check adds identical token-forcing overhead to both versions and is
an execution regression control, not a production throughput claim.

The BF16-source hybrid run on the **24 GB RTX 4090**, with 64 GPU experts per
layer, used **20.45 GiB peak PyTorch allocation** and generated 64 tokens:

| Hybrid execution | Median decode tok/s |
| --- | ---: |
| CPU/GPU overlap, six CPU threads | 15.63 |
| Same weights, serial execution | 15.48 |

All six runs produced identical text. The overlap gain is too small to call
meaningful here; the fixed partition is largely CPU-limited. One/two CPU threads
measured 11.87/14.29 tok/s. Six real expert projection checks measured
0.78–0.94% relative RMS error from INT8 weights/BF16 computation; this is not a
model-quality evaluation. The hybrid run and GPTQ control use different weight
quantization and are not accuracy-equivalent comparisons.

For ordinary inference on this particular model, the GPTQ checkpoint already
fits on the 4090 with the unused MTP head disabled: **about 116 tok/s**, compile
mode 0, at **21.55 GiB** peak PyTorch allocation (128 output tokens, three runs).
This prototype is groundwork for larger models, not the recommended speed path
when all expert weights already fit on the GPU.
