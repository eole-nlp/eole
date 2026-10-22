# Restricted MTP draft-head experiment

Re-measured on **2026-10-22**, rebased onto main `69179d7a` (including #475).
Benchmark source: `ba2b9dd9`. One RTX 5090, driver 610.43.02,
PyTorch 2.12.1+cu132 / CUDA 13.2. The target is the converted RedHatAI
Qwen3.8-27B-INT4 checkpoint with a dense BF16 output generator.

## Method

`benchmark_draft_vocab.py` compares the full draft generator against an
additional dense head containing 40,960 of 248,320 vocabulary rows. Both arms
keep the extra 400 MiB weight allocation resident, use the same loaded target,
three MTP drafts, greedy decoding, FlashAttention, packed KVQ, a 64,000-token
context limit, and a 512-token output cap. Normalization is unchanged.

Each prompt is warmed in both arms before measurement. **Four measured repetitions**
alternate arm order for each prompt (24 measured requests per compile mode).
All measured outputs reached the cap. Decode throughput counts the 511 tokens
after the first token using synchronized first-to-last-token timing. Per-phase
GPU profiling is enabled in both arms. These are small workload samples, not a
quality evaluation or a benchmark against another engine.

Mode 0 uses main's full-chunk-only verifier compilation and local verifier cache.
The unchanged target verifier is reused by both draft heads; shortened tails
remain eager. Mode 2 retains layer compilation. Startup/cache loading is excluded
from the throughput figures. Maximum measured request warmup was 0.065 s in mode
0 and 0.256 s in mode 2. Both processes run separately on the same GPU.

The shortlist comes from the pinned external repository linked in the
[recipe](README.md#experimental-restricted-mtp-draft-vocabulary), SHA256
`b64b6dfcf5441eb995ddf77d3d37b018e91b88c56ad1b4c5774ad8fbfac1c388`.
It is an English/code-oriented example, not a recommended multilingual vocabulary.
It is downloaded separately and is not bundled into Eole.

## Current results

| Compile mode | Prompt | Full head tok/s | Restricted head tok/s | Change | Acceptance full → restricted |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | Orwell | 101.23 | 111.33 | +10.0% | 48.4% → 43.9% |
| 0 | Code | 144.51 | 166.41 | +15.2% | 82.3% → 82.3% |
| 0 | French | 103.60 | 92.83 | -10.4% | 49.9% → 31.8% |
| 2 | Orwell | 98.55 | 108.31 | +9.9% | 49.6% → 44.0% |
| 2 | Code | 133.69 | 151.64 | +13.4% | 80.0% → 76.7% |
| 2 | French | 98.20 | 89.65 | -8.7% | 49.5% → 30.6% |

Rates are arithmetic means of four repetitions. Acceptance is accepted/drafted
tokens aggregated across repetitions. Full request wall times include prefill,
warmup, decoding and Python overhead:

| Mode | Prompt | Full tok/s range | Restricted tok/s range | Mean request wall time, full → restricted |
| --- | --- | --- | --- | --- |
| 0 | Orwell | 100.64–101.84 | 110.87–112.02 | 5.377 → 4.898 s |
| 0 | Code | 144.03–144.90 | 165.00–167.44 | 3.881 → 3.418 s |
| 0 | French | 102.66–104.49 | 90.80–93.54 | 5.344 → 5.915 s |
| 2 | Orwell | 97.94–98.78 | 107.56–108.96 | 5.683 → 5.217 s |
| 2 | Code | 133.36–134.17 | 146.00–154.86 | 4.359 → 3.907 s |
| 2 | French | 98.06–98.48 | 87.87–90.50 | 5.802 → 6.296 s |

The isolated projection plus argmax measured **1.651 → 0.260 ms** in mode 0
and **1.657 → 0.260 ms** in mode 2 (about 6.4× faster), over 100 GPU-event-timed
calls after 10 warm calls. This excludes the MTP transformer and target verifier.
Faster projection does not guarantee faster generation: French acceptance falls
from about 50% to 31–32%, requiring enough extra verification cycles to lose throughput.

[Machine-readable measurements](draft-vocab-results-20261022.jsonl) include runtime
and source provenance, per-request token counts/timings, acceptance, GPU phase
measurements and output hashes. Original local logs and earlier results were retained.

## Correctness and limitations

Only proposals use the smaller head. Target verification, its full output
vocabulary, target weights and normalization are unchanged. Excluded tokens can
still be emitted by the target. Only dense `nn.Linear` output generators are
supported; other types are rejected. Tensor parallelism was not benchmarked.

Outputs are stable within every arm in all four repetitions and match the previous
2026-10-20 experiment. Mode-0 code outputs are also identical **between** arms.
Prose in both modes and mode-2 code differ between full and restricted heads.
Different text trajectories can affect acceptance and speed; only mode-0 code
holds the generated text identical in this comparison.

Fresh mode-0 target traces were collected after the new timing runs, outside the
benchmark. Their texts match the measured outputs. The first divergences again
involve a target-score tie in the full-head run:

| Prompt | Sequence position including prompt | Full-head target top two log probabilities | Restricted-head target top two log probabilities |
| --- | ---: | --- | --- |
| Orwell | 350 | -1.1796875, -1.1796875 | -1.1328125, -1.2578125 |
| French | 454 | -0.75, -0.75 | -0.6875, -0.8125 |

This is consistent with BF16 numerical differences when changed proposals alter
verification chunk boundaries. It is not a guarantee of bitwise-identical greedy
output. Keep this opt-in and measure the shortlist on the intended workload.

## Previous measurements (2026-10-20)

Preserved for reference: main `cc4b6b88`, same GPU/model/runtime family and 64k/512
limits, two alternating repetitions, with only Orwell warmed before measurement.
These are not a controlled isolation of a main-branch performance change: the
new runs warm every workload, use four repetitions, and record the current driver.

| Mode | Prompt | Full head tok/s | Restricted head tok/s | Change |
| --- | --- | ---: | ---: | ---: |
| 0 | Orwell | 101.68 | 111.37 | +9.5% |
| 0 | Code | 144.22 | 168.28 | +16.7% |
| 0 | French | 103.85 | 93.42 | -10.0% |
| 2 | Orwell | 98.45 | 107.51 | +9.2% |
| 2 | Code | 129.95 | 151.10 | +16.3% |
| 2 | French | 95.72 | 88.86 | -7.2% |
