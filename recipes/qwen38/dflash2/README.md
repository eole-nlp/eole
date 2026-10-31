# Experimental DFlash2 decoding

This prototype runs the BF16 `incoai/Qwen3.8-27B-DFlash2` drafter with a
converted Qwen3.8-27B target. It supports single-GPU, batch-one, text-only
greedy inference. The seven-token proposal is verified in one eight-token
target pass, using Eole's existing attention/GDN state commit and rollback.

The drafter shares the target's embeddings and vocabulary projection. Its
five-layer backbone uses PyTorch SDPA. Compile mode 0 replays the complete
proposal block through one CUDA graph, prepared during model loading. Fixed
context K/V buffers retain only committed target states; speculative noise K/V
is temporary. Target verification uses the existing compile/CUDA graph path.
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

There are two independent quantization settings:

- `dflash2_quantization: none` keeps BF16 drafter projections; `int8` uses group-32
  runtime RTN Marlin projections. Norms, convolution controls and the selector stay native.
- `vocabulary_quantization: none`, `int8` or `int4` controls the shared vocabulary
  projection used by both the target and drafter. This changes target logits as
  well as proposals. The measured quantized heads use group size 128.

`fuse_kvq` and `fuse_gate` also pack the compatible INT8 drafter projections.
BF16 drafter gate/up packing is an experiment, not enabled by those flags in this PR.
The default HF checkpoint block size is 8, giving seven proposals. The graph path
uses the checkpoint block size; `self_speculative_num_tokens` does not override it.


## Matched benchmark: INT8 vocabulary heads, block size 8

All rows below use the same converted RedHat Qwen3.8-27B INT4 target and shared
INT8 group-128 vocabulary head. This is the configuration of the original
optimization table, not the INT4-head configuration used in the local Orwell script.

RTX 5090; Torch 2.14.1/CUDA 13.2; Triton 3.8.0; FLA 0.6.0; FlashAttention 2.8.3.
The code, English mystery and Chinese-to-English prompts are the same
[`mtp-sampling-fixtures.json`](../mtp-sampling-fixtures.json) fixtures throughout.
Batch one, greedy, thinking disabled, context capacity 64,000, output cap 2,048,
BF16 KV cache, prefix reuse off, fused target projections and compile mode 0.
Each fixture gets a 128-token warmup and three measured repetitions. Optional
phase-event profiling is off. Values are median decode tokens/s, excluding the
first token, prefill and warmup.

| Variant | Code | English | Translation |
| --- | ---: | ---: | ---: |
| BF16, current graph | 223.6 | 140.1 | 212.5 |
| BF16, compilation | 230.5 | 135.8 | 213.1 |
| BF16, CUDA RMSNorm + SiLU | 226.4 | 132.0 | 213.5 |
| BF16, compilation + packed gate/up | 231.7 | 136.6 | 214.7 |
| BF16, CUDA kernels + packed gate/up | 227.0 | 132.4 | 214.6 |
| INT8 drafter, packed attention + MLP | 239.8 | 139.6 | 221.0 |

The current BF16 graph baseline reproduces all nine original output token
sequences. Its original measurement was 223.0 / 139.7 / 212.6 tokens/s; the small
rerun difference is measurement variation, not an implementation improvement.

Only the current graph path and optional packed INT8 drafter are implemented by
this PR. Drafter `torch.compile`, extra CUDA RMSNorm/SiLU substitutions, and BF16
gate/up packing are benchmark-only experiments. Target verification is compiled
and graph-replayed in every row. The current drafter graph records eager kernels;
it does not itself use `torch.compile`.

Compilation and kernel substitutions can change rounding, proposal acceptance
and generated sequences. These are observed end-to-end results, not isolated
kernel speedups. No quality evaluation was performed.

## INT4 vocabulary-head benchmark, block size 8

This separate comparison keeps the same RedHat INT4 target, **BF16 drafter**,
block size 8 and benchmark protocol above, but uses **INT4 group-128 shared
vocabulary heads**. The current implementation reaches **272.4 code tokens/s**
in this configuration:

| BF16 drafter variant | Code | English | Translation |
| --- | ---: | ---: | ---: |
| Current graph (implemented in this PR) | **272.4** | 132.7 | 193.1 |
| Compilation (experiment) | 278.1 | 143.1 | 192.1 |
| CUDA RMSNorm + SiLU (experiment) | 274.6 | 145.2 | 192.7 |
| Compilation + packed gate/up (experiment) | 279.5 | 143.7 | 193.2 |
| CUDA kernels + packed gate/up (experiment) | 275.5 | 146.0 | 193.5 |

For the implemented baseline row, configure the shared head independently of
DFlash2 projection quantization:

```yaml
dflash2_model: /path/to/Qwen3.8-27B-INT4/dflash2
dflash2_quantization: none
vocabulary_quantization: int4
vocabulary_quantization_group_size: 128
fuse_kvq: true
fuse_gate: true
```

The drafter checkpoint's `dflash_config.block_size` must be 8 for this table.
Use the same batch-one greedy settings, 64,000-token context and 2,048-token
output cap described above. Changing the head affects target logits and
acceptance, so 272.4 versus the INT8-head baseline's 223.6 is a configuration
comparison, not an implementation-only speedup.

## Why the other tables had different numbers

The following are graph-only BF16 drafter baselines, not successive optimization
stages. Only the shared head setting and/or checkpoint block size changes:

| Shared vocabulary head | Block size | Code | English | Translation |
| --- | ---: | ---: | ---: | ---: |
| INT8 group 128 | 8 | 223.6 | 140.1 | 212.5 |
| INT4 group 128 | 8 | 272.4 | 132.7 | 193.1 |
| INT4 group 128 | 4 | 205.3 | 144.9 | 172.9 |

Switching INT8 to INT4 heads at block size 8 changes acceptance from
55.3% / 29.7% / 51.6% to 67.3% / 25.7% / 43.3%, respectively. Translation
length changes from 719 to 1,065 generated tokens. The 272.4 code figure is
therefore not a further optimization gain over 223.0. Head projection cost and
output/acceptance changes both contribute. A shorter translation is not evidence
of better quality.

The block-size-4 runs use a temporary four-position drafter configuration with
three proposals; block size 8 proposes seven. Longer blocks help when later
proposals are accepted often enough to offset verification cost. These fixtures
do not establish one universally optimal block size.

## Earlier implementation checkpoints

These historical rows all used INT8 group-128 vocabulary heads and block size 8:

| Earlier stage | Code | English | Translation |
| --- | ---: | ---: | ---: |
| MLP-only graph baseline | 191.2 | 115.0 | 158.2 |
| Fused Q/K normalization and RoPE | 211.3 | 120.1 | 168.4 |
| Full-drafter graph, BF16 | 223.0 | 139.7 | 212.6 |
| Full graph, unpacked INT8 drafter projections | 229.2 | 134.7 | 220.7 |
| Full graph, packed INT8 drafter projections | 240.7 | 140.2 | 222.3 |

The MLP-only graph implementation has been removed; full-drafter replay replaces
it. The initial short binary-search measurements used a different prompt, native
BF16 heads and a 4,096-token context, so they are omitted from this comparison.
Earlier Ninfer results used different mixed weights, proposal heads and FP8 KV
cache; they do not isolate an engine-only difference and are not included here.

## Fusion and startup conclusions

Packing BF16 gate/up replaces two projections with one concatenated-weight
projection. The existing CUDA SiLU kernel can consume the packed result directly.
It saves launches and an intermediate concatenation, but does not reduce weight
bytes read. The down projection remains after the nonlinear activation. In these
runs packing adds less than 1% to the compiled path; it is not a large remaining
speedup by itself.

Graph-only drafter preparation is about one second. The first compile-before-
capture experiments took about 8-10 seconds, while a cached matching case took
about two seconds. This is startup work, separate from target-verifier compilation
(about a minute cold in these tests, or a few seconds to load a saved verifier).
Warmup is excluded from the throughput tables. Whole-drafter compilation is not
currently enabled in the PR.

## Correctness and scope

The small-model drafter calculation was compared with the upstream reference
across cache updates and sliding-window eviction (maximum FP32 difference 3e-8).
Tests cover split-half RoPE, dynamic convolution, selector transitions, bounded
context, chunked feature collection, real CUDA graph replay, ring wraparound,
partial appends, stable-buffer request resets, packed projections, and target
verification with ordinary and GDN decoders, including incorrect proposals.

Target FP32 normalization is unchanged. Drafter Q/K normalization remains FP32
and preserves intermediate BF16 casts. Reduction ordering can still affect
rounding, acceptance and later trajectories. Repeated runs are deterministic
within each tested configuration, but bit-identical output to single-token
ordinary decoding has not been established. Broader numerical and quality
checks remain necessary before production use.

The prototype supports single-GPU, batch-one, beam-one, text-only greedy
inference. Sampling, tensor parallelism, vision inputs and other compile modes
remain outside its tested scope.

The adapted drafter implementation retains the upstream MIT notice in
[`licenses/dflash-MIT.txt`](../../../licenses/dflash-MIT.txt).
