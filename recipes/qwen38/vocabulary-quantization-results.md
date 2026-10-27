# Vocabulary-head quantization benchmark

## Current main after packed projections

Measured on 2026-10-26 after rebasing #481 onto main `8f60c58d` (#490).
Both `fuse_kvq` and `fuse_gate` are enabled for every head format. This includes
Marlin GDN QKV/Z and MLP gate/up packing, single-token paired A/B dispatch and
the cache/fusion allocation fixes. Unmerged #489 (accepted-state commit),
#491 (bounded cache views), and restricted draft-vocabulary work are excluded.

PyTorch 2.12.1+cu132, FLA 0.5.2, Triton 3.7.1, FlashAttention 2.8.3. The same
existing extension (CUDA 13.3, SM89/SM120) is used without rebuilding it. Model,
decoder weights, tokenizer, normalization and fixtures are unchanged between
head formats. Only the shared vocabulary projection changes: original BF16,
symmetric RTN INT8 or symmetric RTN INT4, group 128 with BF16 scales. The same
head serves ordinary decoding, MTP drafting and target verification.

[Post-GDN results](vocabulary-quantization-post-gdn-results.md),
[post-Marlin results](vocabulary-quantization-post-marlin-results.md),
[pre-Marlin results](vocabulary-quantization-pre-marlin-results.md) and their
raw files are retained. The previous post-GDN campaign kept gate fusion off;
this campaign enables it equally in all arms. Cross-campaign differences
therefore do not isolate any single optimization. Quantization can change
continuations, lengths and acceptance: these are sampled workload timings,
not same-token comparisons or equivalent-quality claims.

## RTX 5090: MTP sampling

Original Eole code/story/translation fixtures (`eole-v1`), seeds 42/43. Compile
mode 0, 64,000 reserved context, batch/beam one, three MTP drafts, output cap
4096, prefill chunks 1024, both fusion flags enabled, prefill cache disabled.
Sampling: temperature 0.6, top-k 20, top-p 0.95, presence penalty 1, frequency
penalty 0. One 128-token code warmup precedes the measured requests, preserving
the earlier procedure. MTP phase-timing instrumentation remains enabled.

Rates are arithmetic means of two measured decode rates. Timing excludes
loading, head quantization, compilation, prefill and the first generated token;
EOS is included. Acceptance pools accepted/drafted counts across both seeds.

| Workload | BF16 tok/s | INT8 tok/s | Gain | INT4 tok/s | Gain |
|---|---:|---:|---:|---:|---:|
| Python code | 151.82 | 178.70 | +17.7% | 188.31 | +24.0% |
| English story | 100.14 | 114.06 | +13.9% | 122.24 | +22.1% |
| Chinese→English | 140.52 | 165.71 | +17.9% | 179.32 | +27.6% |

MTP acceptance:

| Workload | BF16 | INT8 | INT4 |
|---|---:|---:|---:|
| Python code | 83.1% | 84.4% | 83.0% |
| English story | 42.7% | 41.3% | 42.4% |
| Chinese→English | 71.1% | 71.8% | 73.1% |

### Per-request RTX 5090 results

| Head | Workload | Seed | Decode tokens | tok/s | Accepted / drafted | Same text as BF16 |
|---|---|---:|---:|---:|---:|---|
| BF16 | Python code | 42 | 4095 | 149.34 | 2901 / 3582 | yes |
| BF16 | English story | 42 | 2934 | 99.26 | 1638 / 3888 | yes |
| BF16 | Chinese→English | 42 | 672 | 139.47 | 456 / 651 | yes |
| BF16 | Python code | 43 | 4095 | 154.30 | 2944 / 3450 | yes |
| BF16 | English story | 43 | 2583 | 101.02 | 1460 / 3372 | yes |
| BF16 | Chinese→English | 43 | 651 | 141.58 | 446 / 618 | yes |
| INT8 | Python code | 42 | 4095 | 174.65 | 2908 / 3558 | no |
| INT8 | English story | 42 | 2934 | 113.97 | 1623 / 3936 | no |
| INT8 | Chinese→English | 42 | 643 | 164.87 | 439 / 615 | no |
| INT8 | Python code | 43 | 4095 | 182.75 | 2961 / 3399 | no |
| INT8 | English story | 43 | 2708 | 114.15 | 1498 / 3630 | no |
| INT8 | Chinese→English | 43 | 651 | 166.55 | 446 / 618 | yes |
| INT4 | Python code | 42 | 4095 | 190.13 | 2936 / 3474 | no |
| INT4 | English story | 42 | 3004 | 121.97 | 1675 / 3990 | no |
| INT4 | Chinese→English | 42 | 660 | 178.01 | 451 / 627 | no |
| INT4 | Python code | 43 | 4095 | 186.48 | 2907 / 3564 | no |
| INT4 | English story | 43 | 2656 | 122.51 | 1496 / 3480 | no |
| INT4 | Chinese→English | 43 | 661 | 180.63 | 457 / 615 | no |

[Raw 5090 metadata and measurements](vocabulary-quantization-post-packing-results.jsonl). Full texts and logs are
retained locally under `/tmp/eole-vocab-post490`.

## RTX 4090: repeated 32K checks

Same model, fixtures, head group size and sampling settings. Reserved context
32,768, output cap 512. Each fixture receives a full 512-token warmup, then
three measured repeats at seed 42, preserving the previous 4090 procedure.
Rates are arithmetic means of three decode rates. Each head/workload repeats
its output hash across the three requests; this checks repeatability, not quality.

| Workload | BF16 tok/s | INT8 tok/s | Gain | INT4 tok/s | Gain |
|---|---:|---:|---:|---:|---:|
| Python code | 92.89 | 106.30 | +14.4% | 127.96 | +37.8% |
| English story | 61.62 | 72.62 | +17.8% | 81.98 | +33.0% |
| Chinese→English | 86.87 | 105.64 | +21.6% | 114.05 | +31.3% |

MTP acceptance:

| Workload | BF16 | INT8 | INT4 |
|---|---:|---:|---:|
| Python code | 75.5% | 72.1% | 81.7% |
| English story | 40.1% | 39.1% | 40.6% |
| Chinese→English | 70.2% | 72.3% | 68.7% |

All three head formats completed these 32K short-prompt checks without OOM.
The desktop uses about 1 GB of VRAM. Historical 64K OOM checks were not repeated;
these runs do not guarantee full-context capacity. The output caps differ
between GPUs, so their absolute rates are not directly comparable.

[Raw 4090 metadata and measurements](vocabulary-quantization-4090-post-packing-results.jsonl). GPU workloads were
serialized, including across GPUs, with no concurrent model or GPU-test process
during measured requests.

## Validation

- CPU suite: 760 tests run, 40 skipped; all executed tests passed, with optional FLA disabled and recipe integration excluded.
- 50 focused checks passed on each GPU, covering head packing/dequantized-head agreement and source-weight preservation, inference wiring, packed projections, GDN kernels, KV caches and CUDA graphs.
- Black with repository configuration, flake8 and whitespace checks passed.

## Reproduce

Use the refreshed #481 branch and the existing matching Eole extension. Run
each head with a separate output path:

```bash
CUDA_VISIBLE_DEVICES=0 EOLE_TORCH_COMPILE=1 EOLE_COMPILE_MODE=0 \
  python recipes/qwen38/benchmark_mtp_sampling.py \
  --model "$QWEN38_MODEL" --head none --fuse-gate --output /tmp/vocab-bf16.json
```

Repeat with `--head int8` and `--head int4`. The `--fuse-gate` option is needed
to reproduce this campaign; omitting it retains the earlier harness setting.
For RTX 4090, select its CUDA device and add:

```text
--context-length 32768 --tokens 512 --seeds 42 42 42 --warmup-tokens 512 --warmup-all-prompts
```

No output-quality benchmark or new 64K capacity test was performed.

Real non-MTP compiled model checks also passed for INT8 batch four, INT4
worker-thread streaming and INT4 beam four.
