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

### Adaptive INT4/INT8 GPU cache

`moe_expert_backend: adaptive` uses one bounded GPU expert cache with two
explicit weight paths:

| Converted source checkpoint | CPU experts | GPU experts | Weight changes |
| --- | --- | --- | --- |
| GPTQ INT4 | Original INT4 | Original INT4 (Marlin) | Codes/scales preserved |
| Floating-point (BF16/FP16) | Runtime INT8 | Same runtime INT8 (Marlin) | Per-output-channel RTN quantization |

The floating-point path quantizes routed experts once at load. CPU and GPU share
the same signed INT8 codes and rounded BF16 scales; changing residency does not
change quantization. The GPU repeats each per-channel scale across 128-value
groups to use the existing Marlin group layout. Attention, shared experts,
vocabulary heads and FP32 normalization retain their existing precision. Runtime
quantization changes numerical behavior versus the floating-point checkpoint.

The existing `hybrid_int8` backend remains the fixed CPU INT8/GPU BF16 variant.
`hybrid_gptq` remains an alias restricted to GPTQ INT4; use `adaptive` for either
supported source. Loading an already-quantized GPTQ INT8 checkpoint is not yet
supported. GPTQ inputs require symmetric INT4, group128, BF16 scales, direct zero
point 8, no activation ordering or bias; unsupported layouts are rejected.

**Rebuild the Eole CUDA extension after checking out this PR**. It adds the
BF16-activation INT8 MoE specializations; older extensions support only INT4 MoE.
From the repository root with the intended Python environment active:

```bash
MAX_JOBS=4 python setup.py build_ext --inplace
```

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 EOLE_TORCH_COMPILE=0 \
  python recipes/qwen35/benchmark_moe.py \
  --model /path/to/converted/Qwen3.5-35B-A3B-GPTQ-Int4 \
  --backend adaptive --gpu-experts 64 --cache-policy frequency \
  --cache-dir /path/to/local/ssd/cache \
  --tokens 128 --repeats 3 --output /tmp/qwen35-adaptive.json
```

For a floating-point source, set `QWEN35_MOE_MODEL` and use
[predict-adaptive.yaml](predict-adaptive.yaml). For GPTQ INT4, set
`QWEN35_MOE_GPTQ_MODEL` and use [predict-hybrid-gptq.yaml](predict-hybrid-gptq.yaml).
Equivalent inference settings:

```yaml
moe_expert_backend: adaptive
moe_gpu_experts: 64
moe_expert_cache_policy: frequency  # fixed keeps the initial residency
moe_expert_cache_interval: 32
moe_expert_cache_swaps: 4
moe_expert_cache_dir: /path/to/local/ssd/cache
moe_expert_cuda_graph: true
compute_dtype: bf16
self_attn_backend: flash
dynamic_shapes: false
self_speculative_decoding: false
```

The slot count is **per layer, not a VRAM limit**. Leave room for dense weights,
KV/recurrent state, graph pools, prefill workspace and one spare expert per layer.
Start conservatively and measure process GPU memory with your intended context
and prompts. Increasing batch size or context changes the required headroom.

Frequency admission learns demand over successive requests, replaces at most four
experts every 32 layer calls, and decays old counts. It changes residency only:
every selected expert is still evaluated with its original routing weight. Cache
misses run on the CPU while cached experts run on the GPU. A frozen domain-specific
cache can be faster than adaptation for a stable workload; changing domains needs
time to learn. No policy guarantees that adaptation improves every request.

GPU execution uses Eole's existing Marlin kernels. Single-token expert graphs use
a persistent GPU ID-to-slot table; supported decoder layers also capture attention,
routing and shared-expert stages around the CPU handoff. Prefill and unsupported
layer layouts stay eager. Static KV/GDN buffers are owned by the hybrid runtime
and reset in place at the next request. Compatible requests reuse their graphs;
a batch/shape/dtype/device change invalidates the bindings before buffers are
replaced. The first two real decode calls warm a new stage and the third captures
it. Initial capture costs remain part of the first request's latency.
`--no-hybrid-graph` disables all these hybrid GPU stages
for comparison. When every expert fits in the cache, no CPU routing readback or
CPU expert job is needed. This does not enable whole-decoder `torch.compile`: the backend
still requires `EOLE_TORCH_COMPILE=0`, one GPU and ordinary decoding.

CPU native INT4/INT8 layouts stay in RAM. Separate Marlin layouts are temporarily
memory-mapped on the selected SSD, so they do not require a second anonymous RAM
bank. Allow disk space for the routed expert weights at execution precision
(INT4 or INT8). Preparation can take minutes and is excluded from decode timings.
Packing occurs at model load, one expert at a time on CUDA. Temporary files are
removed when the engine is terminated or preparation fails. This is an in-process GPU cache,
not an SSD-streaming inference engine or a reusable converted checkpoint format.
CPU/GPU kernel rounding can differ near ties; this is not a promise of bit-identical
generation to the fully resident backend. FP32 normalization is unchanged.

#### Floating-point and GPTQ path confirmation

Both formats were rerun after adding INT8 MoE kernels: eight selected experts,
native vocabulary head, no MTP, 4096-token context, prefill chunk 32 and 128 output
tokens (127 timed), six CPU threads on the RTX 5090/Ryzen 5900X. Identical token
histories are forced; these are practical checkpoint configurations, not an
accuracy evaluation or an isolated expert-bitwidth ablation. Non-expert weights
retain each source checkpoint's precision. Unused vision modules were on CPU.

The table shows the third successive adapted request per task, with learning
retained across code → English → translation:

| Source / expert storage | Slots/layer | Harness cap | Process peak | Code | English | Translation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GPTQ / original INT4 | 91 | 16 GiB | 15.57 GiB | 45.4 tok/s | 48.0 tok/s | 51.1 tok/s |
| BF16 / runtime INT8 | 48 | 20 GiB | 15.94 GiB | 34.0 tok/s | 33.4 tok/s | 32.8 tok/s |

80 exact layer/output/KV/GDN checks passed for each source (160 total). Kernel
packing/migration/replay tests passed on the RTX 5090 and RTX 4090. CPU and GPU
kernel rounding may differ despite identical codes/scales; cache warming can
therefore change generated text even with fixed sampling settings.

The BF16 checkpoint first failed the artificial 20 GiB allocator cap with 80
cache slots while validating GPU graph workspaces; the 48-slot run above passed.
Capacity is not a byte budget: floating-point dense weights and graph workspaces
need headroom beyond the expert banks. Successful BF16 preparation on local NVMe
with warm source data took about 215 seconds; a cold run using the source SSD for
backing took about 696 seconds before that cap failure. Packing is excluded from
the table's decode timings, and temporary backing files are removed on close.

Normal resident compile-mode-0 throughput after the rebuild was **160.6 tok/s**,
within 1% of the prior matched main control (160.7 tok/s).

#### Earlier GPTQ cache-capacity measurements

Original Qwen3.5-35B-A3B-GPTQ-Int4 codes/scales, eight routed experts per token,
BF16 compute, native vocabulary head and no speculation. PyTorch 2.15.0/CUDA
13.2, Ryzen 5900X with six CPU threads, static 4096-token KV cache, 32-token
prefill chunks and 128 output tokens (127 timed). These controlled decode runs
force fixed code/English/translation token histories; they are **not quality
scores or unconstrained-generation benchmarks**. Unused vision blocks were on CPU.

| GPU budget | Cached experts/layer | Code | English | Translation | Measured process peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 GiB | 91 | 46.5 tok/s | 49.7 tok/s | 49.9 tok/s | 15.57 GiB |
| 20 GiB | 157 | 62.2 tok/s | 67.9 tok/s | 70.6 tok/s | 19.57 GiB |
| All experts on GPU | 256 | 129.3 tok/s | 129.0 tok/s | 128.4 tok/s | 25.43 GiB |

Each task runs three successive requests; the table shows the **third**, after
frequency admission has adapted. Tasks run code → English → translation without
resetting learning. Cold requests are slower; there is no universal speedup for
arbitrary workloads. The harness enforced the byte caps and sampled process
memory; the configuration's slot count alone does not enforce a byte limit.
Graphs reuse compatible KV/GDN storage between requests. Initial capture is
included in the first request, rather than subtracted from timings.

With all experts cached on GPU, there is no CPU expert computation and the
executor bypasses the CPU routing readback. Hybrid GPU stages retain more launch
boundaries than whole-decoder compilation; use the normal resident backend when
the checkpoint fits entirely on GPU.

A fresh ordinary resident compile-mode-0 code control measured **160.67
tok/s on main** and **160.74 tok/s with this backend available**, using
the same checkpoint and fixed history. It retained the default resident executor
and native vocabulary head; no throughput regression was observed in this control.

A separate code control froze residency after the translation workload, then ran
three repeats per execution mode with the same assignments and token histories:

| GPU budget | Eager GPU phases | All hybrid GPU stages |
| --- | ---: | ---: |
| 16 GiB | 15.9 tok/s | 33.7 tok/s |
| 20 GiB | 19.1 tok/s | 53.2 tok/s |
| All experts on GPU | 24.2 tok/s | 127.6 tok/s |

These medians isolate dispatch costs from adaptation; the frozen translation
cache is not the adapted-code cache in the first table. Exact output and KV/GDN
state checks against eager execution passed for 80 layer/token pairs at each
capacity (240 total), including graph reuse across requests. This is not a
promise of bit-identical end-to-end generation across CPU and GPU arithmetic.

### BF16 source with CPU INT8

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
  decoder compilation is rejected for this backend. The default resident backend
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
meaningful here; both CPU work and the eager execution path contribute to the limit. One/two CPU threads
measured 11.87/14.29 tok/s. Six real expert projection checks measured
0.78–0.94% relative RMS error from INT8 weights/BF16 computation; this is not a
model-quality evaluation. The hybrid run and GPTQ control use different weight
quantization and are not accuracy-equivalent comparisons.

For ordinary inference on this particular model, the GPTQ checkpoint already
fits on the 4090 with the unused MTP head disabled: **about 116 tok/s**, compile
mode 0, at **21.55 GiB** peak PyTorch allocation (128 output tokens, three runs).
This prototype is groundwork for larger models, not the recommended speed path
when all expert weights already fit on the GPU.

### Matched 4090 / 5090 follow-up

Same script and settings on both GPUs: 128 output tokens (127 timed), 16-token
warmup, three measured repetitions, six CPU threads, 4,096-token cache, native
vocabulary heads and no MTP. Hybrid uses 64 BF16 GPU experts/layer plus CPU INT8
experts; resident uses the prepared GPTQ checkpoint. Different quantization means
these are practical configurations, not accuracy-equivalent execution backends.

| Configuration | RTX 4090 tok/s | RTX 5090 tok/s |
| --- | ---: | ---: |
| Hybrid BF16/CPU INT8, eager | 15.6 | 16.1 |
| GPTQ, resident eager | 25.8 | 25.7 |
| GPTQ, resident compile mode 0 | 118.9 | 161.9 |

The near-identical resident eager speeds suggest host launch/synchronization
overhead limits GPU scaling; compiled execution exposes the hardware difference.
It does not establish which individual operations cause the eager overhead.

Lightweight host timing of three additional hybrid runs on the 5090 measured
62.8 ms/decode token overall. CPU expert work took 22.6 ms/token, including
15.2 ms/token in native CPU linear calls. The executor boundary took 29.1 ms/token
inclusive of CPU work; about 33.7 ms/token remained outside it. These nested,
potentially overlapping wall-clock measurements are not additive GPU kernel
profiling. They indicate that optimizing only CPU matvecs is insufficient:
GPU segments, Python dispatch and synchronization also need investigation.
Whole-decoder compilation is currently rejected by this prototype; compiling
GPU segments around the host-execution boundary is a possible next experiment.

### GPU stages around the host expert boundary

For `hybrid_int8`, `moe_expert_cuda_graph: true` (default) captures two GPU
phases per supported decoder layer: attention/normalization/routing, then the
shared expert and residual combination. CPU expert execution and the existing
resident-expert dispatch remain outside these graphs. This does not compile the
whole decoder or change expert selection, quantization or FP32 normalization.

The graph path requires ordinary single-token inference, a static cache
(`dynamic_shapes: false`), FlashAttention and the supported sequential residual
layout. Prefill, batches larger than one, attention/alignment output and other
unsupported layouts retain the eager path. The first two eligible calls run
eagerly; the third captures and executes once. Compatible requests reset owned
KV/GDN buffers in place and reuse the captures. A changed cache shape, dtype or
device invalidates the graphs before replacing their buffers. Termination
releases the graphs and CPU worker; a capture failure warns and falls back.

To compare the same backend without these GPU graphs, set
`moe_expert_cuda_graph: false` or pass `--no-hybrid-graph` to
`benchmark_moe.py`. Whole-decoder `EOLE_TORCH_COMPILE` remains disabled.

The CPU single-token worker also avoids regrouping already-distinct top-k
expert IDs and redundant input copies. Multi-token prefill retains grouped
expert execution and the original route ordering.

### Fixed-partition hybrid graph confirmation (RTX 5090)

BF16 source, 64 BF16 GPU experts and 192 CPU INT8 experts per layer, native
vocabulary head, no MTP, six CPU threads, context 4096 and prefill chunk 32.
After a discarded 16-token warmup, three fixed-history runs generated 64 output
tokens each (63 timed). These compare the **same updated executor**, changing
only `moe_expert_cuda_graph`; the CPU worker optimization is present in both.
Unused vision modules were placed on CPU for this text-only measurement.

| Routing | GPU phases eager, tok/s | GPU phases graphed, tok/s |
| --- | ---: | ---: |
| Legacy four selected experts | 17.34 | 30.89 |
| Corrected eight selected experts | 12.96 | 20.15 |

The BF16 metadata used in the older hybrid measurements above selected four
experts per token. The official model selects eight; use the corrected row for
that workload. This check changed the routing count explicitly without altering
the source checkpoint. Peak measured PyTorch allocation was 19.71 GiB, excluding
unused vision modules, in both graph modes. Captures were retained in both comparison arms, so this
is not a measurement of incremental graph memory. It is not a model-quality
evaluation.

160 exact layer/token checks (two per layer at each routing count) compared GPU
graph execution with the existing eager layer, including KV and GDN states.
Each of the 40 attention/router and 40 shared-expert graphs was captured once
per routing configuration and reused across the measured requests.

### Resident regression confirmation after GPU-stage extraction

Fresh controls against clean main `3d6a170d` used the same GPTQ checkpoint,
compiled extension, forced 128-token history and settings described above.
Each cell is the median of three runs; the eager comparison was repeated in
reverse process order because the first pass showed a small difference.

| Resident mode / run order | Main tok/s | This PR tok/s |
| --- | ---: | ---: |
| Eager, main then PR | 26.10 | 25.17 |
| Eager, PR then main | 25.13 | 25.47 |
| Compile mode 0 | 161.49 | 160.51 |

The eager ordering reverses which version is faster; these measurements do not
establish a repeatable eager regression. Compiled throughput differs by less
than 1%. Both versions measured 21.55 GiB peak PyTorch allocation. This is a
limited regression control, not a guarantee across all models and workloads.
