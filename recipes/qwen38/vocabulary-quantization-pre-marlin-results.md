# Historical vocabulary-head quantization benchmark

These measurements predate the Marlin changes in #485. See the [current rerun](vocabulary-quantization-results.md) for the rebased PR results.

RTX 5090, PyTorch 2.12.1+cu132, FlashAttention 2.8.3; RedHat Qwen3.8-27B INT4 decoder unchanged across arms. Original Eole prompts revision eole-v1. Only the shared vocabulary projection changes: BF16, symmetric RTN INT8, or symmetric RTN INT4 (group 128, BF16 scales). The same head serves draft selection and target verification.

Compile mode 0, 64,000 reserved context, prefill chunk 1024, batch/beam 1, MTP with three drafts, 4096 output cap. Temperature 0.6, top-k 20, top-p 0.95, presence penalty 1, frequency penalty 0. Seeds 42/43, one measured request per seed/workload/head after 128-token warmup. Decode timing excludes the first token and includes EOS; loading, quantization and compilation are excluded. Rates below are arithmetic means of two runs; acceptance sums accepted/drafted counts.

These sampled workload timings are not same-token comparisons or equivalent-quality claims. Quantization can change final text and acceptance.

| Workload | BF16 tok/s | INT8 tok/s | Gain | INT4 tok/s | Gain |
|---|---:|---:|---:|---:|---:|
| scenario_code_python | 135.63 | 154.68 | +14.0% | 162.33 | +19.7% |
| scenario_story_en_mystery | 87.96 | 103.60 | +17.8% | 106.19 | +20.7% |
| scenario_translation_zh_en | 124.22 | 142.33 | +14.6% | 149.51 | +20.4% |

| Workload | BF16 acceptance | INT8 acceptance | INT4 acceptance |
|---|---:|---:|---:|
| scenario_code_python | 83.6% | 82.5% | 82.6% |
| scenario_story_en_mystery | 41.8% | 44.0% | 41.5% |
| scenario_translation_zh_en | 70.5% | 69.9% | 68.8% |

## Per-request results

| Head | Workload | Seed | Decode tokens | tok/s | Accepted / drafted | Same text as BF16 |
|---|---|---:|---:|---:|---:|---|
| BF16 | scenario_code_python | 42 | 4095 | 133.88 | 2911 / 3552 | yes |
| BF16 | scenario_story_en_mystery | 42 | 2680 | 87.20 | 1479 / 3606 | yes |
| BF16 | scenario_translation_zh_en | 42 | 677 | 124.51 | 460 / 651 | yes |
| BF16 | scenario_code_python | 43 | 4095 | 137.38 | 2943 / 3453 | yes |
| BF16 | scenario_story_en_mystery | 43 | 2906 | 88.73 | 1631 / 3828 | yes |
| BF16 | scenario_translation_zh_en | 43 | 634 | 123.92 | 431 / 612 | yes |
| INT8 | scenario_code_python | 42 | 4095 | 152.38 | 2897 / 3593 | no |
| INT8 | scenario_story_en_mystery | 42 | 3487 | 104.57 | 2003 / 4452 | no |
| INT8 | scenario_translation_zh_en | 42 | 677 | 143.27 | 460 / 651 | yes |
| INT8 | scenario_code_python | 43 | 4095 | 156.98 | 2935 / 3480 | no |
| INT8 | scenario_story_en_mystery | 43 | 2658 | 102.64 | 1493 / 3498 | no |
| INT8 | scenario_translation_zh_en | 43 | 692 | 141.38 | 467 / 675 | no |
| INT4 | scenario_code_python | 42 | 4095 | 159.01 | 2892 / 3608 | no |
| INT4 | scenario_story_en_mystery | 42 | 2912 | 108.17 | 1638 / 3822 | no |
| INT4 | scenario_translation_zh_en | 42 | 646 | 152.98 | 441 / 618 | no |
| INT4 | scenario_code_python | 43 | 4095 | 165.64 | 2943 / 3456 | no |
| INT4 | scenario_story_en_mystery | 43 | 3077 | 104.20 | 1685 / 4179 | no |
| INT4 | scenario_translation_zh_en | 43 | 725 | 146.04 | 484 / 726 | no |

## Reproduce

Run the [sampling benchmark](benchmark_mtp_sampling.py) three times, passing --head none, --head int8, and --head int4, with CUDA_VISIBLE_DEVICES=0 EOLE_TORCH_COMPILE=1 EOLE_COMPILE_MODE=0. Keep other options at their defaults and use separate --output JSON paths. See the [recipe](README.md) for the full command.

[Raw metadata and measurements](vocabulary-quantization-results.jsonl) include fixture hashes and output hashes. Full texts are retained locally in /tmp/eole-vocab-pr. No output-quality benchmark was performed.

## RTX 4090: repeated 32K compatibility and speed checks

Mode 0, MTP with three drafts, 32,768 reserved context, 512-token output cap, group 128. Same sampling configuration and original Eole prompts as above. Each fixture was warmed with a full 512-token request before three measured repeats using seed 42. This isolates timing variation on the same text within each head; the three repeats are not independent quality samples. The active desktop used approximately 1 GB of VRAM.

| Workload | BF16 tok/s | INT8 tok/s | Gain | INT4 tok/s | Gain |
|---|---:|---:|---:|---:|---:|
| scenario_code_python | 82.58 | 95.67 | +15.9% | 108.41 | +31.3% |
| scenario_story_en_mystery | 56.13 | 65.98 | +17.6% | 70.61 | +25.8% |
| scenario_translation_zh_en | 76.63 | 88.50 | +15.5% | 95.23 | +24.3% |

All three heads completed all repeated requests at 32K. INT8 and INT4 loaded successfully at 64,000 context but both ran out of memory during warmup generation. These short-prompt checks do not guarantee that every long-input request fits; prefill temporaries and other GPU users also affect available memory.

All measured repeats produced identical output hashes within each head/workload. Between heads, outputs can differ. An earlier isolated INT4 code result of 55.9 tok/s was not reproduced: the three fully warmed repeats measured 108.2–108.6 tok/s. That earlier run is excluded from this table, as is the restarted run interrupted because a video was playing. No definitive cause is assigned to the earlier outlier.

[Per-request 4090 metadata, timing, acceptance and output hashes](vocabulary-quantization-4090-results.jsonl). These 512-token results should not be directly compared with the 4096-token RTX 5090 campaign.
