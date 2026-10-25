# Vocabulary-head quantization benchmark

## Rerun after the Marlin changes

Measured on 2026-10-25 after rebasing #481 onto main `3a386c4e` (#485). The installed extension was rebuilt for SM89/SM120 using CUDA 13.3; PyTorch is 2.12.1+cu132 with FlashAttention 2.8.3. RedHat Qwen3.8-27B INT4 decoder, tokenizer, prompts and normalization are unchanged across heads. Only the shared vocabulary projection changes: BF16, symmetric RTN INT8 or symmetric RTN INT4, group 128 with BF16 scales. The same head serves ordinary decoding, MTP draft selection and target verification.

The tables below are the new campaign. [Previous measurements and their methodology](vocabulary-quantization-pre-marlin-results.md) are retained, along with the original JSONL files. Generated text can change with head quantization and the merged Marlin scheduling; neither comparisons between heads nor comparisons with historical results establish equivalent output quality or isolate a kernel-only speedup.

## RTX 5090: MTP sampling

Original Eole code/story/translation fixtures, revision `eole-v1`, seeds 42/43. Compile mode 0, 64,000 reserved context, batch/beam 1, three MTP drafts, 4,096-token output cap, 1,024-token prefill chunks, packed attention KVQ on, gate fusion off, prefill cache off. Sampling: temperature 0.6, top-k 20, top-p 0.95, presence penalty 1, frequency penalty 0. One 128-token code warmup precedes the measured requests, matching the original 5090 campaign. Per-phase MTP timing events remain enabled as in that campaign.

Rates are arithmetic means of two measured decode rates. Timing excludes the first token and includes EOS, and excludes loading, head quantization, compilation and prefill. Acceptance sums accepted/drafted counts across both seeds. Outputs and lengths can differ between heads.

| Workload | BF16 tok/s | INT8 tok/s | Gain | INT4 tok/s | Gain |
|---|---:|---:|---:|---:|---:|
| Python code | 137.09 | 159.76 | +16.5% | 167.99 | +22.5% |
| English story | 91.87 | 103.17 | +12.3% | 111.36 | +21.2% |
| Chinese→English | 127.93 | 147.25 | +15.1% | 150.84 | +17.9% |

| Workload | BF16 acceptance | INT8 acceptance | INT4 acceptance |
|---|---:|---:|---:|
| Python code | 82.5% | 84.5% | 84.1% |
| English story | 43.6% | 41.3% | 43.6% |
| Chinese→English | 71.2% | 70.8% | 67.6% |

### Per-request RTX 5090 results

| Head | Workload | Seed | Decode tokens | tok/s | Accepted / drafted | Same text as BF16 |
|---|---|---:|---:|---:|---:|---|
| BF16 | Python code | 42 | 4095 | 135.42 | 2895 / 3598 | yes |
| BF16 | English story | 42 | 3093 | 91.65 | 1751 / 4026 | yes |
| BF16 | Chinese→English | 42 | 675 | 128.40 | 461 / 645 | yes |
| BF16 | Python code | 43 | 4095 | 138.76 | 2936 / 3474 | yes |
| BF16 | English story | 43 | 2762 | 92.09 | 1568 / 3585 | yes |
| BF16 | Chinese→English | 43 | 671 | 127.46 | 457 / 645 | yes |
| INT8 | Python code | 42 | 4095 | 159.34 | 2927 / 3500 | no |
| INT8 | English story | 42 | 2718 | 102.67 | 1496 / 3666 | no |
| INT8 | Chinese→English | 42 | 648 | 147.67 | 441 / 621 | no |
| INT8 | Python code | 43 | 4095 | 160.18 | 2945 / 3448 | no |
| INT8 | English story | 43 | 2762 | 103.67 | 1538 / 3675 | no |
| INT8 | Chinese→English | 43 | 635 | 146.83 | 432 / 612 | no |
| INT4 | Python code | 42 | 4095 | 165.78 | 2912 / 3547 | no |
| INT4 | English story | 42 | 3187 | 112.80 | 1828 / 4080 | no |
| INT4 | Chinese→English | 42 | 672 | 156.42 | 460 / 636 | no |
| INT4 | Python code | 43 | 4095 | 170.20 | 2952 / 3426 | no |
| INT4 | English story | 43 | 2515 | 109.91 | 1403 / 3339 | no |
| INT4 | Chinese→English | 43 | 908 | 145.27 | 599 / 930 | no |

[Raw 5090 metadata, timings, token counts, acceptance and output hashes](vocabulary-quantization-post-marlin-results.jsonl). Full texts are retained locally under `/tmp/eole-vocab-post485`.

## RTX 4090: repeated 32K checks

Same model, prompts, head group size and sampling configuration. Reserved context is 32,768 and output cap is 512. Each fixture receives a full 512-token warmup, then three measured repeats at seed 42, matching the previous 4090 campaign. Each head/workload repeated its own output hash in all three runs. These are timing repeats, not independent quality samples. Rates below are arithmetic means; the desktop occupies approximately 1 GB of VRAM.

| Workload | BF16 tok/s | INT8 tok/s | Gain | INT4 tok/s | Gain |
|---|---:|---:|---:|---:|---:|
| Python code | 83.91 | 97.57 | +16.3% | 109.57 | +30.6% |
| English story | 54.31 | 68.06 | +25.3% | 70.63 | +30.0% |
| Chinese→English | 79.17 | 91.70 | +15.8% | 95.04 | +20.1% |

| Workload | BF16 acceptance | INT8 acceptance | INT4 acceptance |
|---|---:|---:|---:|
| Python code | 76.7% | 76.7% | 81.7% |
| English story | 38.3% | 43.9% | 40.3% |
| Chinese→English | 71.5% | 71.5% | 67.3% |

All three head formats completed the repeated 32K tests. The historical 64K OOM checks were not repeated; these short-prompt results are not a long-context capacity guarantee. Do not directly compare the 512-token 4090 numbers with the 4,096-token 5090 campaign.

[Raw 4090 metadata and per-request measurements](vocabulary-quantization-4090-post-marlin-results.jsonl).

## Reproduce

Use the merged Marlin kernels and rebuild the extension if necessary. Run the same checkpoint once per head, with separate output paths:

```bash
CUDA_VISIBLE_DEVICES=0 EOLE_TORCH_COMPILE=1 EOLE_COMPILE_MODE=0 \
  python recipes/qwen38/benchmark_mtp_sampling.py \
  --model "$QWEN38_MODEL" --head none --output /tmp/vocab-bf16.json
```

Repeat with `--head int8` and `--head int4`. Defaults reproduce the 5090 settings. For the 4090 campaign, select its CUDA device and add:

```text
--context-length 32768 --tokens 512 --seeds 42 42 42 --warmup-tokens 512 --warmup-all-prompts
```

The warmup options make the previous 4090 procedure reproducible without a separate script; default behavior remains the original 128-token first-prompt warmup. No output-quality benchmark was performed.
