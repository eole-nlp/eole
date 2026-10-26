# Vocabulary-head quantization benchmark

## Current main after the GDN fixes

Measured on 2026-10-26 after rebasing #481 onto main `84f99d40`, including #487 (paired GDN A/B projections) and #488 (native FP32 FLA launch boundary), as well as #485 (Marlin register/CTA changes). **The unmerged direct accepted-state commit PR #489 is excluded.**

PyTorch 2.12.1+cu132, FLA 0.5.2, Triton 3.7.1, FlashAttention 2.8.3. The existing extension was built with CUDA 13.3 for SM89/SM120; its SHA256 matches the previous campaign. Model, decoder weights, tokenizer, normalization and fixtures remain unchanged between head formats. Only the shared vocabulary projection changes: BF16, symmetric RTN INT8 or symmetric RTN INT4, group 128 with BF16 scales. The same head serves ordinary decoding, MTP drafting and target verification.

[Previous post-Marlin campaign](vocabulary-quantization-post-marlin-results.md), its JSONL files, and [pre-Marlin measurements](vocabulary-quantization-pre-marlin-results.md) are retained. Comparisons across heads or campaigns do not establish equivalent output quality or isolate a kernel-only speedup: quantization and floating-point changes can alter continuations, lengths and acceptance.

## RTX 5090: MTP sampling

Original Eole code/story/translation fixtures (`eole-v1`), seeds 42/43. Compile mode 0, reserved context 64,000, batch/beam 1, three MTP drafts, output cap 4096, prefill chunks 1024, packed attention KVQ enabled, gate fusion disabled, prefill cache disabled. Sampling: temperature 0.6, top-k 20, top-p 0.95, presence penalty 1, frequency penalty 0. One 128-token code warmup precedes the measured requests, preserving the original procedure. MTP phase-timing instrumentation remains enabled.

Rates are arithmetic means of the two measured decode rates. Timing excludes loading, head quantization, compilation, prefill and the first generated token; EOS is included. Acceptance pools accepted/drafted counts across both seeds.

| Workload | BF16 tok/s | INT8 tok/s | Gain | INT4 tok/s | Gain |
|---|---:|---:|---:|---:|---:|
| Python code | 149.82 | 173.50 | +15.8% | 183.98 | +22.8% |
| English story | 99.06 | 115.98 | +17.1% | 121.08 | +22.2% |
| Chinese→English | 140.20 | 160.17 | +14.2% | 160.87 | +14.7% |

MTP acceptance:

| Workload | BF16 | INT8 | INT4 |
|---|---:|---:|---:|
| Python code | 83.3% | 83.2% | 84.0% |
| English story | 43.1% | 43.2% | 43.4% |
| Chinese→English | 72.6% | 70.3% | 65.5% |

### Per-request RTX 5090 results

| Head | Workload | Seed | Decode tokens | tok/s | Accepted / drafted | Same text as BF16 |
|---|---|---:|---:|---:|---:|---|
| BF16 | Python code | 42 | 4095 | 147.35 | 2906 / 3564 | yes |
| BF16 | English story | 42 | 2855 | 98.02 | 1606 / 3750 | yes |
| BF16 | Chinese→English | 42 | 669 | 141.91 | 462 / 624 | yes |
| BF16 | Python code | 43 | 4095 | 152.29 | 2942 / 3456 | yes |
| BF16 | English story | 43 | 2404 | 100.10 | 1360 / 3135 | yes |
| BF16 | Chinese→English | 43 | 670 | 138.49 | 457 / 642 | yes |
| INT8 | Python code | 42 | 4095 | 171.14 | 2903 / 3573 | no |
| INT8 | English story | 42 | 2599 | 117.07 | 1476 / 3372 | no |
| INT8 | Chinese→English | 42 | 861 | 160.74 | 585 / 831 | no |
| INT8 | Python code | 43 | 4095 | 175.85 | 2944 / 3453 | no |
| INT8 | English story | 43 | 2713 | 114.89 | 1525 / 3567 | no |
| INT8 | Chinese→English | 43 | 635 | 159.59 | 431 / 615 | no |
| INT4 | Python code | 42 | 4095 | 185.12 | 2930 / 3492 | no |
| INT4 | English story | 42 | 2826 | 120.91 | 1585 / 3726 | no |
| INT4 | Chinese→English | 42 | 1058 | 159.22 | 702 / 1071 | no |
| INT4 | Python code | 43 | 4095 | 182.84 | 2932 / 3488 | no |
| INT4 | English story | 43 | 3361 | 121.25 | 1915 / 4341 | no |
| INT4 | Chinese→English | 43 | 899 | 162.52 | 596 / 912 | no |

[Raw 5090 metadata and measurements](vocabulary-quantization-post-gdn-results.jsonl). Full generated texts and logs are retained locally under `/tmp/eole-vocab-post488`.

## RTX 4090: repeated 32K checks

Same model, fixtures, head group size and sampling settings. Reserved context 32,768, output cap 512. Every fixture receives a full 512-token warmup, then three measured repeats at seed 42, preserving the previous 4090 procedure. Rates are arithmetic means of three decode rates. Each head/workload repeats its own output hash across all three requests; this is a repeatability check, not a quality evaluation.

| Workload | BF16 tok/s | INT8 tok/s | Gain | INT4 tok/s | Gain |
|---|---:|---:|---:|---:|---:|
| Python code | 90.83 | 107.58 | +18.4% | 124.20 | +36.7% |
| English story | 64.72 | 72.46 | +12.0% | 80.03 | +23.6% |
| Chinese→English | 87.18 | 99.90 | +14.6% | 109.43 | +25.5% |

MTP acceptance:

| Workload | BF16 | INT8 | INT4 |
|---|---:|---:|---:|
| Python code | 75.5% | 74.5% | 81.7% |
| English story | 44.2% | 40.6% | 40.1% |
| Chinese→English | 71.5% | 68.8% | 66.9% |

All three head formats complete these 32K checks. The desktop uses approximately 1 GB of VRAM. Historical 64K OOM checks were not repeated; short-prompt success does not guarantee long-context capacity. The 4090 output cap differs from the 5090 campaign, so their absolute rates are not directly comparable.

[Raw 4090 metadata and measurements](vocabulary-quantization-4090-post-gdn-results.jsonl). All GPU campaigns run serially, including between GPUs. No other model process or GPU test runs concurrently with measured requests.

## Validation after rebasing

- 751 CPU tests passed, 39 skipped, with optional FLA disabled in the CPU test process; recipe integration tests are excluded as in the previous campaign.
- 29 CUDA checks passed on each GPU, covering vocabulary packing/dequantized-head agreement and source-weight preservation, Marlin launches/fusion, and the current FLA compile/recurrent adapters.
- Black with the repository configuration, flake8, and whitespace checks passed.

## Reproduce

Use the branch with vocabulary quantization rebased onto the current main and the matching Eole extension. Run each head with a separate output path:

```bash
CUDA_VISIBLE_DEVICES=0 EOLE_TORCH_COMPILE=1 EOLE_COMPILE_MODE=0 \
  python recipes/qwen38/benchmark_mtp_sampling.py \
  --model "$QWEN38_MODEL" --head none --output /tmp/vocab-bf16.json
```

Repeat with `--head int8` and `--head int4`. Defaults reproduce the 5090 procedure. For the 4090, select its CUDA device and add:

```text
--context-length 32768 --tokens 512 --seeds 42 42 42 --warmup-tokens 512 --warmup-all-prompts
```

No output-quality benchmark or new 64K capacity test was performed.
