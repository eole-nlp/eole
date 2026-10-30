# Experimental DFlash2 decoding

This prototype runs the BF16 `incoai/Qwen3.8-27B-DFlash2` drafter with a
converted Qwen3.8-27B target. It supports single-GPU, batch-one, text-only
greedy inference. The seven-token proposal is verified in one eight-token
target pass, using Eole's existing attention/GDN state commit and rollback.

The drafter shares the target's embeddings and vocabulary projection. Its
five-layer backbone uses PyTorch SDPA. In compile mode 0, each pure MLP
branch (normalization, dynamic convolutions and MLP) replays through a CUDA
graph. The graphs are prepared during model loading, using the same eager
kernels. Attention/cache handling and the candidate selector stay eager;
target verification also uses the existing compile/CUDA graph path.
Intermediate target features and drafter context are bounded by its 2,048-token
sliding window. Chunked prefill is supported; persistent prefix reuse is disabled
for this path because existing prefix entries do not contain intermediate features.

## Try locally

Use the normal Eole environment and a converted target checkpoint. Download the
separate HF-format drafter without converting it:

```bash
hf download incoai/Qwen3.8-27B-DFlash2 \
  --revision 015e795645c74b1a0eeef3b570031fb62e769bc5 \
  --include config.json model.safetensors \
  --local-dir /path/to/Qwen3.8-27B-DFlash2

EOLE_TORCH_COMPILE=1 EOLE_COMPILE_MODE=0 \
python recipes/qwen38/dflash2.py \
  --model /path/to/Qwen3.8-27B-INT4 \
  --drafter /path/to/Qwen3.8-27B-DFlash2 \
  --mode dflash2 --max-tokens 512 --repeats 2
```

Use `--mode none` and `--mode mtp` for controls with the same target and prompt.
The recipe defaults to native BF16 vocabulary heads, a 4,096-token target cache,
and fused projections. `--context-length` and `--prompt` can be adjusted.

In a regular `PredictConfig`, set:

```yaml
dflash2_model: /path/to/Qwen3.8-27B-DFlash2
batch_size: 1
beam_size: 1
top_k: 1
compute_dtype: bf16
gpu_ranks: [0]
```

`dflash2_model` enables the external drafter and derives the proposal length from
its checkpoint. `self_speculative_num_tokens` continues to control native MTP.
The prototype has no workspace selector or deployment-template integration yet.

## Initial measurements

RTX 5090, PyTorch 2.14.1 / CUDA 13.2, native BF16 heads, RedHat INT4 target,
compile mode 0, 4,096-token context capacity. The recipe's binary-search prompt
contains 30 input tokens and each run generates 512 tokens. Timing excludes
warmup and prefill; decode throughput counts 511 tokens after the first token.
Two sequential requests per mode:

| Mode | Request 1 tok/s | Request 2 tok/s |
| --- | ---: | ---: |
| Ordinary greedy | 83.3 | 83.3 |
| Native MTP, 3 drafts | 162.4 | 166.2 |
| DFlash2, 7 drafts | 193.2 | 205.0 |

DFlash2 accepted 70.7% of proposed tokens. Peak PyTorch allocated memory was
22.8 GiB, including loading and warmup, versus 19.0 GiB for ordinary decoding.
These are short code-prompt prototype measurements, not a general benchmark or
a comparison with another engine. First-request graph capture still affects
throughput; use repeated requests when evaluating steady-state performance.

## Shared-position optimization on the longer fixtures

The drafter now computes RoPE tables and its attention mask once per proposal
block and shares them across all layers. This replaces 15 rotary-table
calculations and five equivalent masks in the five-layer drafter.

A before/after comparison uses the original
[`mtp-sampling-fixtures.json`](../mtp-sampling-fixtures.json) code, English and
translation tasks also used in the local Ninfer comparison: 64,000 context
limit, 2,048 output cap, greedy decoding, thinking disabled, shared INT8 heads
(group size 128), BF16 KV cache, fused projections and compile mode 0. Each
fixture receives a 128-token warmup before three measured repetitions; optional
phase-event profiling is disabled. Median decode tok/s:

| Workload | Before | Shared RoPE/mask | Gain | Earlier Ninfer DFlash2 |
| --- | ---: | ---: | ---: | ---: |
| Code | 165.9 | 176.9 | +6.6% | 243.4 |
| English | 99.7 | 106.7 | +7.0% | 156.9 |
| Translation | 137.1 | 147.2 | +7.3% | 245.3 |

All nine Eole before/after pairs have exactly identical token IDs and acceptance
counts. The Ninfer column is the earlier reference run, with different mixed
quantized weights, an optimized Q8 proposal head and FP8 dynamic KV cache;
physical cache behavior and generated sequences differ. Eole translation emits
1,189 tokens, versus 654 in that Ninfer run. These figures therefore establish
the Eole optimization's benefit, while the remaining cross-engine gap needs
further investigation.

## MLP-branch CUDA graphs

Compile mode 0 also records the cache-independent MLP branches and replays them
for full eight-token draft blocks. Shortened tails, training and gradient-enabled
calls use the eager branch. Preparation happens during loading (about 0.30 s on
the test 5090), avoiding capture work in the first request. This adds graph
recording without additional `torch.compile` programs or changes to normalization.

The same longer-workload benchmark above, shared INT8 vocabulary heads and
native BF16 drafter, gives these medians:

| Workload | Shared-position baseline | MLP graphs | Gain |
| --- | ---: | ---: | ---: |
| Code | 176.9 | 191.2 | +8.1% |
| English | 106.7 | 115.0 | +7.8% |
| Translation | 147.2 | 158.2 | +7.5% |

All nine before/after pairs have exactly identical token IDs and acceptance
counts. Dedicated tests exercise real GPU capture/replay, eager tails and
training/gradient fallbacks. The drafter backbone remains BF16 by default;
experimental INT8 and packed-projection trials changed output trajectories and
were not a consistent improvement across these workloads. The historical native
MTP INT8-head results remain faster on English and translation, and Ninfer remains
ahead on all three workloads.

## Correctness status

The small-model drafter calculation was compared with the upstream reference
across cache updates and sliding-window eviction (maximum FP32 difference
3e-8). CPU regressions cover split-half RoPE, dynamic convolution, candidate
transitions, bounded context, chunked feature collection, and target verification
with ordinary and GDN decoders, including deliberately incorrect proposals.

Repeated real-model DFlash2 runs produced identical token IDs to each other.
They matched ordinary greedy decoding through the first 499 tokens. At token
500, ordinary decoding assigned exactly equal BF16 log probabilities to its
leading two candidates; eight-token verification slightly separated those
scores and selected the other candidate. Native MTP matched ordinary decoding
for this particular prompt. The prototype therefore does **not** establish
bit-identical output to single-token decoding. Broader teacher-forced numerical
and quality checks remain necessary before production use.

Sampling, tensor parallelism, vision inputs, other compile modes, and drafter
quantization remain future work. FP32 normalization in the target is unchanged.

The adapted drafter implementation retains the upstream MIT notice in
[`licenses/dflash-MIT.txt`](../../../licenses/dflash-MIT.txt).
