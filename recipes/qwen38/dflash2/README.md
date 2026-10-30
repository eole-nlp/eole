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
quantized weights, an optimized INT4 proposal head and FP8 dynamic KV cache;
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
were not a consistent improvement at that stage, before full-drafter replay.
The historical native MTP INT8-head results were faster on English and translation,
and Ninfer was ahead on all three workloads. Updated results follow below.

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

Sampling, tensor parallelism, vision inputs and other compile modes remain future work. FP32 normalization in the target is unchanged.

The adapted drafter implementation retains the upstream MIT notice in
[`licenses/dflash-MIT.txt`](../../../licenses/dflash-MIT.txt).


## Full-drafter replay and INT8 projections

Compile mode 0 now prepares one CUDA graph for the complete eight-position
proposal block. Committed context uses fixed ring buffers, reset in place between
requests; prompt K/V is populated once outside replay. A valid-count mask prevents
padded updates from overwriting older context. Short final requests use only the
required prefix of the seven proposals. This does not compile new whole-decoder
programs or change the target model's FP32 normalization.

Drafter Q/K normalization and split-half RoPE use a fused Triton kernel. The
normalization reduction remains FP32 and preserves intermediate BF16 casts.
Different FP32 reduction association can nevertheless move rare values across a
BF16 rounding boundary. Fixed-size masked attention also changes reduction
ordering; outputs need not match the previous drafter path bit for bit.

The optional `dflash2_quantization: int8` setting quantizes the feature projection,
attention projections and MLP projections using group-32 runtime RTN Marlin
weights. It leaves normalization, convolution controls and the selector native.
It is independent of `vocabulary_quantization`, which controls the shared output
head. BF16 drafter projections remain the default. In the example script, use
`--drafter-quantization int8` to try it.

Sequential RTX 5090 measurements use the same code, English and translation
fixtures above, context 64,000, output cap 2,048, batch one, greedy decoding,
shared INT8 group-128 vocabulary heads, three measured requests per fixture and
no phase profiling. Medians are decode tokens/s, excluding the first token:

| Stage | Code | English | Translation |
| --- | ---: | ---: | ---: |
| Previous MLP graph baseline | 191.2 | 115.0 | 158.2 |
| Fused Q/K normalization and RoPE | 211.3 | 120.1 | 168.4 |
| Fixed-cache full-drafter graph, BF16 (integrated rerun) | 223.0 | 139.7 | 212.6 |
| Full graph with INT8 drafter projections | 229.2 | 134.7 | 220.7 |
| INT8 plus packed QKV/context KV and gate/up (integrated rerun) | 240.7 | 140.2 | 222.3 |

These are end-to-end observed gains, not isolated kernel speedups: acceptance
and generated trajectories change. Acceptance is respectively 55.3% / 29.7% /
51.6% for full-graph BF16 and 53.9% / 25.9% / 50.4% for INT8. Translation emits
719 and 720 tokens, versus 1,189 in the earlier baseline. No translation quality
claim follows from these timing results. Repeated requests are deterministic
within each tested configuration. The historical Ninfer reference remains
243.4 / 156.9 / 245.3 tokens/s with different mixed weights, proposal head and
FP8 KV cache, so this is not a controlled engine-only comparison.


With INT8 drafter projections, `fuse_kvq` packs the noise Q/K/V and context
K/V projections; `fuse_gate` packs MLP gate/up. Original projections are released
after packing. Context K/V needs separate Marlin packing, adding about 55 MB
versus the unpacked INT8 drafter. Packed acceptance is 56.2% / 27.0% / 50.4%.
BF16 is still the default: drafter quantization changes proposals and acceptance,
and should be chosen explicitly rather than inferred from target quantization.

GPU regressions cover real graph capture/replay, updating control values, ring
wraparound, partially valid appends, shortened proposal counts, deterministic
request resets with stable buffer pointers, and packed projection equivalence.


The final integrated BF16 and packed-INT8 reruns reproduced every experimental
output token ID across all nine requests per configuration. Full-drafter graph
preparation took 0.95 s (BF16) and 0.91 s (INT8); target verifier compilation is
separate. The integrated packed result is 25.9% / 21.9% / 40.5% faster than the
previous MLP-graph baseline, including the acceptance/trajectory differences
noted above.
