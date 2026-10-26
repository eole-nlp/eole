# Request-bounded FlashAttention KV-cache views

This change keeps the configured cache allocation and its storage addresses,
but gives FlashAttention a prefix view that safely covers the whole request.
FA2's automatic split heuristic and split partitions depend on the cache view's
capacity. Passing all 64k reserved slots for a short request can therefore pick
less effective partitions, even though the kernel respects actual sequence
lengths and does not attend to unused tokens.

The generator computes an upper bound from the original padded prompt width,
the decoding strategy's maximum output length, and one extra slot (or the MTP
draft count plus one). It rounds up to a power of two, with a minimum of 512,
and clamps to the configured capacity. This is a request-wide bound, not a
per-token length read. It does not reduce reserved KV-cache memory.

A CPU scalar stores the bound. The opaque FlashAttention operation reads it
and creates the view outside the compiled decoder, avoiding recompilation just
because a request's bound changes. Captured graphs record that request's view;
graph bindings are reset before the scalar changes. No GPU-to-CPU length read,
cache copy, fixed split count, or additional token-loop synchronization is added.

Static decoder-only text requests use this automatically in eager execution
and supported bound CUDA graphs in modes 0/2, including MTP and batches.
Images, cross-attention, dynamic cache sizes, compiled beam search and other
legacy graph-tree paths retain the full view. This keeps CPU scalar inputs out
of legacy Inductor graph-tree capture, including modes 1/3. Paged cache
arguments also retain their original behavior.

Changing the split policy can change floating-point rounding and generated
continuations or MTP acceptance. This is not a bit-exact output guarantee or a
quality benchmark. FP32 normalization, projection packing, vocabulary heads,
and GDN state updates are unchanged from main.

## RTX 5090 results

Baseline main `8f60c58d` versus implementation `f8e63aa9`, RedHat Qwen3.8-27B
INT4, BF16 compute/vocabulary heads, compile mode 0, FlashAttention, greedy
batch/beam one, 64k reserved context, 1024-token prefill chunks. Both versions
have `fuse_kvq` and `fuse_gate` enabled. Each process runs two warm requests and
three measured requests, seed 42. All measured outputs deliver 511 decode
tokens after the first token, with a 512-token output cap.

| Case | Prompt tokens | Main tok/s | Bounded view tok/s | Change |
|---|---:|---:|---:|---:|
| No MTP, short | 25 | 81.66 | 84.23 | +3.2% |
| No MTP, long | 5145 | 78.69 | 81.57 | +3.7% |
| MTP, three drafts, short | 25 | 112.47 | 117.74 | +4.7% |

The short cases use a 1024-token logical view, the long case an 8192-token view.
MTP acceptance changes from 296/643 (46.0%) to 303/623 (48.6%) per measured
request. The MTP speed difference therefore includes changed continuations and
acceptance; it is not an isolated measurement of attention savings. Timing
excludes prefill, the first token, compilation and capture, but includes MTP
drafting, verification and state commit. Phase-specific CUDA events are disabled
on both sides. Results repeat within each configuration; hashes differ between
versions. No task-quality evaluation is included.

[Raw results](bounded-kv-cache-results.json) retain the common configuration,
per-case overrides, exact prompts, outputs, hashes and individual measurements.
These are single-GPU measurements on one model, not a general hardware claim.

## Reproduce

Use the same benchmark harness from this branch for both revisions. Its long
case now requests a 500-word response rather than one sentence, so it reaches
the same output cap as the short case:

```bash
cp recipes/qwen38/benchmark_marlin.py /tmp/eole-cache-benchmark.py
CUDA_VISIBLE_DEVICES=0 EOLE_TORCH_COMPILE=1 EOLE_COMPILE_MODE=0 \
  python /tmp/eole-cache-benchmark.py --model "$QWEN38_MODEL" \
  --fuse-gate --output /tmp/cache-short.json
```

Add `--prompt-repeat 256` for the long case, or `--mtp` for the short MTP case.
Run the copied harness on main for the control and this branch for the bounded
view. The other optimizations already merged into main are unchanged. No
checkpoint conversion or CUDA-extension rebuild is required.

## Validation

The CPU suite passed 759 tests (40 skipped). The focused kernel/cache/graph
suite passed 16 tests on each of RTX 5090 and RTX 4090, including FP16/BF16,
one/four query tokens, left padding, boundary writes and changing view bounds
without compiling another program. Tests preserve backing storage and compare
all cache writes with the full-view reference.

Real compiled model checks passed for batch four, beam four (unchanged fallback),
worker-thread streaming, MTP sampling, two-GPU tensor parallel and mode 2.
A persisted MTP verifier loaded successfully on restart and served 512- and
1024-token view bounds without rebuilding its verifier function. These are
compatibility checks, not performance comparisons for those configurations.

The measured implementation revision is `f8e63aa9`. Follow-up `aa9f5fb1` adds
legacy-path eligibility guards and regression coverage; the timed bound-graph
paths retain the same view selection and attention operation.
