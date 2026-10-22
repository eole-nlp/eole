# MTP target sampling: RTX 5090 measurements

Baseline main:69179d7a. This PR adds target sampling with unchanged greedy MTP proposals. Both arms use the original dense BF16 vocabulary heads; the decoder checkpoint remains the converted RedHat Qwen3.8-27B INT4. No quantized-head or FlashInfer changes. FP32 normalization unchanged.

## Settings

RTX5090,PyTorch2.12.1+cu132,FlashAttention2.8.3,compile mode0,64,000 context reservation,prefill chunk1024,batch/beam1,three drafts,4096 output ceiling. Sampling:temperature0.6,top-k20,top-p0.95,presence penalty1.0,frequency penalty0.0. Seeds42/43; two runs per fixture and arm,not a broad quality evaluation. Exact system/user messages and Qwen3.8 thinking-off serialization are retained in the benchmark fixtures. Each process warms with128 tokens before measurement. Decode rate excludes the first token and includes EOS. GPU phase profiling remains enabled for MTP.

## Results

|Fixture|Ordinary sampling tok/s|MTP sampling tok/s|Ratio|MTP draft acceptance|
|---|---:|---:|---:|---:|
|scenario_code_python|70.68|131.84|1.87×|80.4%|
|scenario_story_en_mystery|70.58|90.46|1.28×|45.4%|
|scenario_translation_zh_en|70.37|139.46|1.98×|83.4%|

## Per-request counts and rates

|Fixture|MTP|Seed|Completion tokens|tok/s|Accepted / drafted|
|---|---|---:|---:|---:|---|
|scenario_code_python|False|42|4096|70.63|0 / 0|
|scenario_story_en_mystery|False|42|4096|70.63|0 / 0|
|scenario_translation_zh_en|False|42|877|70.43|0 / 0|
|scenario_code_python|False|43|4096|70.72|0 / 0|
|scenario_story_en_mystery|False|43|4007|70.53|0 / 0|
|scenario_translation_zh_en|False|43|772|70.31|0 / 0|
|scenario_code_python|True|42|4096|133.78|2905 / 3568|
|scenario_story_en_mystery|True|42|4096|91.23|2377 / 5151|
|scenario_translation_zh_en|True|42|766|136.93|542 / 672|
|scenario_code_python|True|43|4096|129.89|2885 / 3630|
|scenario_story_en_mystery|True|43|4096|89.70|2344 / 5250|
|scenario_translation_zh_en|True|43|985|141.99|709 / 828|

## Interpretation and limits

MTP preserves the target sampling distribution for the supplied verification probabilities; it does not consume RNG identically to ordinary decoding. Consequently,seeded continuations and lengths differ. Finite-precision decoder chunking can also perturb target probabilities,as in greedy MTP. This is an observed workload comparison,not a same-token timing guarantee or answer-quality evaluation. Acceptance aggregates accepted/drafted counts across both seeds;speed averages per-request rates.

The published NInfer profile differs:it applies top-k before top-p,uses different quantized weights and FP8 KV,and uses five seeds. Eole retains its existing temperature→top-p→top-k order. Do not treat these as an exact NInfer replication. An initial131K-cache ordinary-sampling process ran out of VRAM;both arms were rerun at64K,and those are the only numbers reported above. The initial results remain in the local initial-131k archive.

## Reproduce

```bash
EOLE_TORCH_COMPILE=1 EOLE_COMPILE_MODE=0 python recipes/qwen38/benchmark_mtp_sampling.py \
  --model "$QWEN38_MODEL" --output /tmp/mtp-sampling.json
EOLE_TORCH_COMPILE=1 EOLE_COMPILE_MODE=0 python recipes/qwen38/benchmark_mtp_sampling.py \
  --model "$QWEN38_MODEL" --output /tmp/ordinary-sampling.json --no-mtp
```

Machine-readable measured counts,timings and output hashes are in [mtp-sampling-results.jsonl](mtp-sampling-results.jsonl). Full output texts are retained locally under /tmp/eole-mtp-sampling. Single-GPU mode0 is GPU-tested here;other compile modes and tensor-parallel sampling have not been GPU-qualified in this experiment.
