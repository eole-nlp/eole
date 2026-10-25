# Marlin register and launch scheduling benchmark

The change lets the shared Marlin GEMM choose its register budget using `__launch_bounds__(threads, 1)`. Other kernels keep the build-wide register policy. The previous 64-register limit caused substantial accumulator spills for larger M tiles.

Dense launches now allocate the tile's required dynamic shared memory, rather than the device maximum. For M <= 8, the dispatcher prefers the existing 128-thread N=64/K=128 tile and chooses CTAs per SM using hardware occupancy and output tile count. The CTA count also respects the existing FP32 reduction scratch allocation. Occupancy queries are cached by device, dimensions, dtypes and quantization layout. Larger M keeps the existing tile preference. Activation-order index storage is counted in bytes when computing shared memory.

## Full inference

RTX 5090, converted `RedHatAI/Qwen3.8-27B-INT4`, main baseline `960c027b`, PyTorch 2.12.1+cu132, rebuilt CUDA extension using CUDA 13.3. Single GPU, batch 1, greedy, compile mode 0, 64,000 context allocation, maximum 512 output tokens, 1,024-token prefill chunks, packed attention KVQ on, gate fusion off, no prefill cache. FP32 normalization and reduction settings are unchanged.

Prompt: `Generate a 500 word text talking about George Orwell.` with the Qwen user/assistant template and an empty thinking block (25 input tokens). Two full requests warm up each process, followed by three measured requests with seed 42. The table reports median decode tokens/second, using Eole's first-to-last-token timing and excluding the first token. Per-phase MTP CUDA events are disabled in this benchmark process to avoid their measurement overhead. The final extension was rebuilt for both SM89 and SM120.

INT8/INT4 heads use group 128 and the implementation from the separate PR #481, temporarily applied for benchmarking. That feature is **not included in this change**. `none` in the script means the original BF16 vocabulary head; the decoder weights remain INT4 in every row.

| Vocabulary head | MTP | Before tok/s | After tok/s | Change | Acceptance before / after |
|---|---|---:|---:|---:|---|
| BF16 | Off | 74.22 | 76.52 | +3.1% | — |
| BF16 | 3 drafts | 102.68 | 101.56 | -1.1% | 48.4% / 46.6% |
| INT8 | Off | 78.92 | 81.85 | +3.7% | — |
| INT8 | 3 drafts | 124.05 | 126.21 | +1.7% | 52.5% / 50.2% |
| INT4 | Off | 82.20 | 83.87 | +2.0% | — |
| INT4 | 3 drafts | 132.95 | 136.33 | +2.5% | 52.0% / 52.8% |

Changing tile and Stream-K partitioning changes floating-point reduction order. Generated outputs are not bit-identical across builds, although each measured configuration repeated its own output. MTP acceptance therefore changes too: these are observed end-to-end results, not a claim that the MTP kernel itself regressed or that all throughput changes come directly from the kernel. There is no substantial MTP speedup in this test. No task-level quality benchmark was run.

A longer **2,581-token input** (the paragraph below repeated 128 times, then a one-sentence summary request), with BF16 heads and MTP off, reduced the median logged first-step time from **6.74 s to 0.65 s (10.4×)**. These are warmed first-step timings, including prefill and first-token selection, rounded by the logger to 0.01 s; they exclude initial compilation.

## Fixed-input projection check

To separate kernel cost from different MTP acceptance, the six real decoder projection shapes were benchmarked with identical activations and existing packed weights, before/after. The control is an isolated rebuild of the original dense kernel with the 64-register cap using the same CUDA 13.3 compiler. Earlier checks matched its timings to the installed baseline. Weights rotate over more than five times L2 capacity; each captured graph runs two full rotations, with two warmup and nine timed replays. Times below sum the six shape timings weighted by their 368 decoder occurrences; these are projection estimates, **not full-decoder times**.

| M | Capped control ms | New launch ms | Ratio |
|---|---:|---:|---:|
| 1 | 9.119 | 8.595 | 1.06× |
| 4 | 9.079 | 8.595 | 1.06× |
| 32 | 67.756 | 11.924 | 5.68× |
| 256 | 686.566 | 56.994 | 12.05× |

Maximum relative RMS output difference across the 24 fixed-input checks was 0.00005609. The large-M benefit mainly comes from removing register spills. The small-M benefit is much smaller and cannot be multiplied directly into whole-model MTP throughput.

## Reproduce

Rebuild the extension for the GPUs being tested (stop inference processes before replacing it):

```bash
TORCH_CUDA_ARCH_LIST="8.9;12.0" MAX_JOBS=2 python setup.py build_ext --inplace
```

Run the same command on the baseline and changed builds, keeping the checkpoint and environment unchanged:

```bash
CUDA_VISIBLE_DEVICES=0 EOLE_TORCH_COMPILE=1 EOLE_COMPILE_MODE=0 \
  python recipes/qwen38/benchmark_marlin.py \
  --model "$QWEN38_MODEL" --head none --output /tmp/marlin-no-mtp.json
```

Add `--mtp` for three drafts. Repeat with `--head int8` and `--head int4` when PR #481 is available. The benchmark refuses these options if that implementation is missing. Preserve the script outside the checkout when comparing with a baseline that predates it.

For the longer prefill case, add `--prompt-repeat 128 --tokens 32`. The repeated paragraph is:

> George Orwell wrote novels and essays about politics, language, individual freedom, and the abuse of power.

The script appends `Summarize this in a single sentence.` and the same Qwen template. It saves full outputs, hashes, decode counts/times, first-step time, wall time and MTP acceptance. Raw run metrics and fixed-input projection checks are in [marlin-benchmark-results.json](marlin-benchmark-results.json).

## Validation and limits

- Numerical CUDA tests passed on RTX 5090 and RTX 4090: FP16/BF16, INT4/INT8, per-channel and groups 32/64/128, M=1 through 256 including partial tiles, full-K activation permutation, repeated dispatch, wide outputs, CUDA-graph replay and the shared MoE kernel. Existing packed-KVQ tests also passed: 18 tests per GPU.
- Focused CPU tests passed: 31 tests, eight CUDA-dependent skips. The broader CPU suite passed 732 tests (29 skips) with optional FLA disabled in the test process. With FLA installed but GPUs hidden, unrelated FLA/Triton tests fail; the same failures were reproduced on unchanged main. The recipe CLI is run separately rather than passing unittest discovery arguments into its parser. Black 24.10.0 and flake8 pass for the added Python files.
- Full-model throughput was measured on the 5090. Other GPU architectures and full MoE models have not been performance-benchmarked.
