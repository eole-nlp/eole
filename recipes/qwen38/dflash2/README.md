# Experimental DFlash2 decoding

This prototype runs the BF16 `incoai/Qwen3.8-27B-DFlash2` drafter with a
converted Qwen3.8-27B target. It supports single-GPU, batch-one, text-only
greedy inference. The seven-token proposal is verified in one eight-token
target pass, using Eole's existing attention/GDN state commit and rollback.

The drafter shares the target's embeddings and vocabulary projection. Its
five-layer backbone, dynamic convolutions and candidate selector run eagerly
in PyTorch; target verification can use compile mode 0 and CUDA graphs.
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
