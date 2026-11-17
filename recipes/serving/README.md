# Greedy Responses serving: decode speed and agent latency

This is an opt-in fixed-slot continuous backend for a **dedicated single-model,
single-GPU, greedy text Responses server**. Existing YAMLs stay serialized.
The workload uses the [Codex integration](../codex/README.md) protocol without
launching Codex, executing generated code or reading user files. It measures
serving performance, not coding-task quality.

## Shared server behavior

The shared server changes are already on main via [#518](https://github.com/eole-nlp/eole/pull/518).
Independently of continuous batching, streaming routes signal
cooperative generation cancellation on disconnect/early consumer exit, instead
of only stopping delivery. Responses, Chat, Anthropic and workspace callers
using the bridge are affected. Cleanup drains the worker. Completed segments
leave a caller-provided shared event unset. Supplying the event also enables
existing request-setting restoration on routes that previously supplied none,
so streaming overrides no longer leak into subsequent requests.

## Run locally

Use a dedicated GPU/server and converted checkpoint:

```bash
export EOLE_MODEL_DIR=/mnt/InternalCrucial4/LLM_work
export QWEN38_MODEL="$EOLE_MODEL_DIR/Qwen3.8-27B-INT4"
export CUDA_VISIBLE_DEVICES=0
export EOLE_TORCH_COMPILE=1 EOLE_COMPILE_MODE=0
export TRITON_DEFAULT_FP_FUSION=1

# Serialized baseline:
python -m eole.bin.main serve -c recipes/serving/serve.yaml \
  --host 127.0.0.1 --port 5019

# After stopping the baseline with Ctrl+C, start continuous serving instead:
python -m eole.bin.main serve -c recipes/serving/serve_continuous.yaml \
  --host 127.0.0.1 --port 5019
```

Wait for readiness, then run in another terminal (`aiohttp` required):

```bash
python recipes/serving/benchmark_responses.py \
  --concurrency 1 2 4 8 --repeats 2 --max-tokens 128 \
  --output /tmp/eole-short.json

python recipes/serving/benchmark_agentic.py \
  --concurrency 1 2 4 8 --duration 30 --context-lines 120 \
  --tool-rounds 3 --max-tokens 128 --output /tmp/eole-agentic.json
```

Both scripts default to `http://127.0.0.1:5019/v1`, model `qwen3.8-27B` and a
600-second HTTP timeout. For authentication use `--api-key-env NAME`; its value
is not saved. Stop the server with Ctrl+C to release its GPU model.

## What each metric measures

The scripts request the Eole-specific `eole_metrics: true` extension. Terminal
Responses then contain server-measured decode durations and counts; default Responses payloads remain
unchanged. A mismatched old server or missing timings fails the benchmark.
Model token counts and timestamps include terminal EOS when generated.

- **Per-request decode p50:** the median token-delivery rate across individual
  model requests, including tool-call and final-answer requests in multi-turn
  workloads. Each rate is `(generated tokens - 1) / (last server token time - first
  server token time)`. It excludes that request's initial queue/prefill and first
  token, but includes pauses while other requests are prefilled or decoded.
  It describes typical token delivery under load, not isolated decoder capacity.
  Serialized requests can decode quickly after a long wait; continuous requests
  share the GPU and can finish whole workflows sooner despite slower delivery.
  Percentiles use the nearest-rank convention; p10 and individual records are saved.
- **First visible output p95:** client HTTP submission to first text/reasoning or
  tool-argument delta. This includes queue/prefill/delivery; tool blocks are
  buffered, so it is **not** pure first-model-token or prefill latency.
- **Jobs/min:** complete workflows finished per minute across all clients;
  higher is better. Wall time includes queueing, all prefills, generation,
  simulated tool delays, HTTP delivery and final drain. A short-text job is one
  request, a short-tool job is a tool call followed by an answer, and a multi-turn
  or 12K-input job is three model/tool exchanges followed by a final answer.
- **Job p95:** the completion time within which 95% of measured jobs finished,
  from each job's submission through its final response. Lower is better.
  For example, 31.60 seconds means 95% finished within about 32 seconds. Small
  samples do not establish production latency guarantees.

Combined decode-window throughput is omitted from comparisons: interleaved
prefills enter the active windows in continuous mode but are outside them in
serialized mode, so that metric does not isolate decoder capacity consistently.

JSON additionally retains `end_to_end_output_tokens_per_second`, explicitly named
as such. It is **not decode speed and is not shown as the headline tok/s metric**.
Earlier figures such as 270 short-prompt or 43 agent-loop tok/s used that
end-to-end definition; they must not be read as decode-only results.
Inter-chunk gaps are also not per-token latency: a tool call may arrive in one
SSE event. Server timings avoid that buffering error.

## Fresh measurement setup

Measured 17 November 2026: RTX 5090, PyTorch 2.15.0+cu132, Triton 3.9.0, FLA 0.6.0, FlashAttention 2.8.3,
compile mode 0, FP fusion enabled, INT4 Qwen3.8-27B target and **native BF16
vocabulary head**, no MTP/DFlash2 or prefix reuse, gate/KVQ fusion enabled.
Short and multi-turn sweeps use **4,096-token capacity in every slot**. Serialized
prefill uses 256 tokens; continuous uses **256 active / 512 idle**. Changing slot
count requires a YAML copy with `continuous.slots` set accordingly.

Each profile has one unmeasured short HTTP warmup. Startup/model graph preparation
is outside the measured cells. All cells run on localhost, with no other compute
model on GPU0. These are initial samples, not statistical speed guarantees.
Different prompts, contexts and concurrency remain different workloads even when
the metric definition is identical to the front-page decode benchmark.

## Short text: one generation per job

Two jobs per client; 128-token output cap. Prompts request a small Python function
and tests, with distinct job IDs. This is the decode-heavy concurrency probe.

| Profile | Clients | Per-request decode p50 (tok/s) | First visible output p95 (s) | Job p95 (s) | Jobs/min |
| --- | ---: | ---: | ---: | ---: | ---: |
| Serialized | 1 | 78.4 | 0.22 | 1.90 | 31.76 |
| Serialized | 2 | 78.5 | 2.14 | 3.80 | 31.73 |
| Serialized | 4 | 78.4 | 5.93 | 7.61 | 31.81 |
| Serialized | 8 | 78.7 | 13.53 | 15.18 | 31.88 |
| 4 slots | 1 | 81.8 | 0.20 | 1.75 | 34.35 |
| 4 slots | 2 | 58.4 | 0.39 | 2.41 | 50.10 |
| 4 slots | 4 | 54.0 | 0.80 | 3.11 | 85.43 |
| 4 slots | 8 | 46.0 | 3.80 | 6.56 | 86.46 |
| 8 slots | 1 | 82.2 | 0.20 | 1.75 | 34.40 |
| 8 slots | 2 | 55.5 | 0.41 | 2.54 | 47.76 |
| 8 slots | 4 | 51.4 | 0.80 | 3.20 | 82.31 |
| 8 slots | 8 | 41.5 | 1.62 | 4.82 | 128.56 |

## Short tool loop: two requests per job

Request `read_fixture(counter.py)`, check its name/arguments/call ID, wait
50 ms, return an in-memory fixture, then request a patch and tests. No code
is executed. Two jobs per client, 128-token cap per request.

| Profile | Clients | Per-request decode p50 (tok/s) | First visible output p95 (s) | Job p95 (s) | Jobs/min |
| --- | ---: | ---: | ---: | ---: | ---: |
| Serialized | 1 | 65.9 | 0.70 | 2.79 | 21.68 |
| Serialized | 2 | 66.0 | 2.65 | 5.42 | 22.26 |
| Serialized | 4 | 66.2 | 6.51 | 10.81 | 22.31 |
| Serialized | 8 | 65.9 | 14.08 | 21.86 | 22.04 |
| 4 slots | 1 | 82.8 | 0.48 | 2.28 | 26.28 |
| 4 slots | 2 | 56.0 | 0.89 | 3.56 | 34.92 |
| 4 slots | 4 | 43.0 | 1.59 | 5.24 | 51.83 |
| 4 slots | 8 | 18.5 | 4.48 | 10.55 | 51.64 |
| 8 slots | 1 | 82.6 | 0.49 | 2.30 | 26.16 |
| 8 slots | 2 | 52.9 | 0.90 | 3.76 | 33.04 |
| 8 slots | 4 | 39.8 | 1.88 | 5.68 | 48.56 |
| 8 slots | 8 | 15.7 | 4.02 | 9.16 | 65.57 |

## Longer multi-turn agents

Reported inputs span 454–2,938 tokens. Mixed short/long repository prompts,
three tool calls plus final answer,
100–400 ms variable tool delays and 150 ms staggered client arrivals. Each
client replenishes work for a 30-second admission window, then all admitted
jobs drain. Every fifth admitted job also opens a separate stream after
completion and disconnects after real text. Cancelled partial tokens are
excluded from rates; their elapsed work remains in job wall time and may
stall other clients. Tool/protocol failures fail the run.

| Profile | Clients | Per-request decode p50 (tok/s) | First visible output p95 (s) | Job p95 (s) | Jobs/min |
| --- | ---: | ---: | ---: | ---: | ---: |
| Serialized | 1 | 64.6 | 1.54 | 8.12 | 8.86 |
| Serialized | 2 | 64.6 | 3.37 | 14.58 | 9.58 |
| Serialized | 4 | 64.5 | 7.78 | 25.88 | 9.82 |
| Serialized | 8 | 64.6 | 14.71 | 47.78 | 9.79 |
| 4 slots | 1 | 82.0 | 1.15 | 6.53 | 10.76 |
| 4 slots | 2 | 38.4 | 2.32 | 11.50 | 12.58 |
| 4 slots | 4 | 18.0 | 4.52 | 19.03 | 13.91 |
| 4 slots | 8 | 11.1 | 7.72 | 34.49 | 14.16 |
| 8 slots | 1 | 81.9 | 1.16 | 6.54 | 10.73 |
| 8 slots | 2 | 45.2 | 2.32 | 11.59 | 12.61 |
| 8 slots | 4 | 17.2 | 4.42 | 19.12 | 13.87 |
| 8 slots | 8 | 8.6 | 7.98 | 31.60 | 14.75 |

Active generations pause while the GPU prefills other requests. That lowers
per-request decode speed under continuous load, while improved prefill/decode
overlap can complete more whole jobs per minute. The short-text probe above is
the closer measure of decode-heavy capacity.

## Larger-context stress: up to 12K input tokens

Copies of the YAMLs use `context_length: 16384`; continuous uses four slots.
This retains the same three tool rounds and variable delays. Reproduce with:

```bash
python recipes/serving/benchmark_agentic.py --concurrency 4 --duration 30 \
  --context-lines 600 --max-tokens 128 --output /tmp/eole-stress.json
```

| Profile | Clients | Per-request decode p50 (tok/s) | First visible output p95 (s) | Job p95 (s) | Jobs/min |
| --- | ---: | ---: | ---: | ---: | ---: |
| Serialized | 4 | 63.2 | 13.13 | 47.43 | 5.01 |
| 4 slots | 4 | 9.6 | 11.04 | 39.17 | 6.07 |

## Memory and fixed context capacity

Sampled whole-device serving peaks below cover each profile’s measured
workloads; nvidia-smi samples every 0.5 seconds and may miss brief spikes.
Weights, native BF16 head, KV/GDN state, graphs and temporary storage are
included. These are observations, not arbitrary-prompt memory bounds.

| Profile | Context per slot | Sampled serving peak (GiB) |
| --- | ---: | ---: |
| Serialized | 4096 | 20.31 |
| 4 slots | 4096 | 21.58 |
| 8 slots | 4096 | 22.94 |
| Serialized | 16384 | 21.93 |
| 4 slots | 16384 | 25.25 |

For this checkpoint, 16 full-attention layers × 4 KV heads × 256 dimensions ×
K/V × 2-byte BF16 = **64 KiB/context token/slot**. Eight slots use 2 GiB of KV
at 4K, 16 GiB at 32K or 32 GiB at 64K, before weights/GDN/graphs/temporaries.
Storage is not paged. Do not extrapolate these results to long-context Codex
sessions or assume the tested configurations fit a 24-GiB display GPU.

## Why the recipe uses 256 active / 512 idle prefill

A retained 8K-context experiment exposed a regression with 128-token active
chunks. The table below is **end-to-end jobs/min**, recalculated from saved job
counts and wall time; decode timings were not saved in that older experiment.
It is a historical policy comparison, not part of the fresh 4K decode tables.

| Eight-client profile, 8K capacity | Admission window (s) | Jobs/min | Job p95 (s) |
| --- | ---: | ---: | ---: |
| Serialized, 256 prefill | 45 | 9.49 | 50.02 |
| 4 slots, 128 active / 512 idle | 45 | 8.38 | 56.20 |
| 8 slots, 128 active / 512 idle | 45 | 8.41 | 56.02 |
| 4 slots, 256 active / 512 idle | 30 | 14.24 | 34.39 |

The windows differ and these are small samples; they are not repeated statistical
comparisons. The larger active chunk improved whole jobs in that workload.
Prefills still run FIFO, one at a time, yielding between decode rounds. This is
not priority scheduling around queued long prompts.

A retained idle-only phase probe encoded a 1,817-token prompt once and timed
cache allocation, prefill/admission and first token, excluding HTTP/tokenization.
One warmup plus two measured repeats per size:

| Idle chunk | Warmed seconds to first token |
| --- | ---: |
| 128 | 1.270 |
| 256 | 0.677 |
| 512 | 0.466 |
| 1024 | 0.460 |

This is a **phase latency** table, not prefill or decode tok/s; 512 captured most
of this measured gain. Reproduce with:

```bash
EOLE_TORCH_COMPILE=1 EOLE_COMPILE_MODE=0 \
python recipes/serving/check_continuous.py --compiled-step \
  --model "$QWEN38_MODEL" --prefill-sweep --output /tmp/eole-prefill-sweep.json
```
Omitting `continuous.prefill_idle_chunk_size` preserves the active chunk size;
the idle value must be at least the active value.

## Retained sustained reuse/cancellation check

The earlier four-slot/eight-client, 8K-context soak used 256 active / 512 idle.
A 180-second admission window plus drain took 200.58 seconds: **47 jobs / 188
ordinary requests / nine intentional disconnects**, or **14.06 jobs/min**.
All tool/protocol checks passed; no ordinary response hit its cap. Clients
completed `6, 6, 6, 6, 6, 6, 6, 5` jobs; short/long first-visible-output p95 was
7.98 / 8.32 seconds and overall job p95 was 37.48 seconds.

Sampled GPU memory stayed at **22.68 GiB** after the first 40 serving seconds
until shutdown; PyTorch peak allocated/reserved was 21.41 / 21.82 GiB. This is
finite evidence of stable memory and client progress, not a universal leak or
fairness proof. Only saved job/latency/memory metrics are retained here; a decode
speed has not been invented retrospectively from buffered SSE events.

Reproduce with `benchmark_agentic.py --concurrency 8 --duration 180`, using an
8K YAML copy. This reliability run need not be repeated to relabel its metrics.

## Worker ownership and supported settings

One dedicated worker owns the engine. RequestCache transfers ownership of KV,
GDN recurrent/convolution state, padding and positions; admission copies a
separately prefilled request into its slot once. Singleton row views share bank
storage. Explicit decoder steps use per-row RoPE positions; ordinary predictor
routing retains its existing path. Startup prepares bank/singleton graphs before
readiness; failed capture fails preparation. Shutdown stops the worker, closes
graphs and unloads the model. A worker failure fails affected streams and requires
reload. Queued, prefilling and decoding cancellation retire/release capacity.

Only greedy text Responses are supported. Sampling, penalties, images,
MTP/DFlash2, prefix reuse, tensor parallelism and general prediction/scoring are
rejected. Use `temperature: 0` or greedy defaults. The server must contain one
model and must not use `model_policy: single`, whose lease would serialize it.
Another ordinary predictor must not run on the worker-owned engine concurrently.

The core probe below tests late arrivals, mixed lengths, cancellation and reuse.
It reports actual chunk boundaries, widths, tokens and mismatch diagnostics:

```bash
CUDA_VISIBLE_DEVICES=0 EOLE_TORCH_COMPILE=0 \
python recipes/serving/check_continuous.py --model "$QWEN38_MODEL" \
  --output /tmp/eole-continuous-core.json
```

## Correctness: solo bit-identity is not guaranteed

The earlier 92-token check was too small to support a broad equality claim.
The expanded test ran ten requests, cancelled three in queued/prefill/decode
states, and completed seven for **1,558 generated tokens**. All references replay
the request's actual prefill chunks.

- Against compiled **single-row** free generation, four of seven continuations
  diverged, first at token indices 67, 147, 177 and 64 (zero-based). Against an
  eager singleton reference, first divergences also differed. Thus selecting
  matching compilation alone does not ensure cross-batch identical output.
- At the first compiled-solo divergences, final-hidden relative RMS differences
  were roughly 0.7–1.2%; each decision involved the same two leading candidates,
  with rounded log-probability ties in one path. This is evidence consistent
  with width-dependent numerical differences, not a model-quality measurement.
- A separate **state-isolation control** replayed the actual token prefixes with
  the same slot, batch width and prefill boundaries, but removed other live
  requests. All **1,558 target decisions matched**. This checks isolation under
  matching arithmetic; it is not independent single-row free generation and
  does not prove universal absence of state bugs.

The probe records widths, tokens and first-mismatch diagnostics. By default a
mismatch fails the run. `--allow-divergence` explicitly permits saving a numerical
comparison without a failing exit; do not treat that flag as an equality pass.

```bash
# Cross-batch comparison: numerical divergence can occur.
EOLE_TORCH_COMPILE=1 EOLE_COMPILE_MODE=0 \
python recipes/serving/check_continuous.py --compiled-step --slots 4 --tokens 256 \
  --prefill-idle-chunk-size 512 --allow-divergence \
  --model "$QWEN38_MODEL" --output /tmp/eole-solo-comparison.json

# Strict isolated-state comparison with matching arithmetic schedule.
EOLE_TORCH_COMPILE=1 EOLE_COMPILE_MODE=0 \
python recipes/serving/check_continuous.py --compiled-step --reference-mode isolation \
  --slots 4 --tokens 256 --prefill-idle-chunk-size 512 \
  --model "$QWEN38_MODEL" --output /tmp/eole-state-isolation.json
```

The uniform-table RoPE regressions compare legacy 2-D and repeated 3-D tables
for multiple rows and tokens, in CUDA FP16/BF16 and the PyTorch/compiled-dispatch
branch. Greedy selection still uses the existing `log_softmax().argmax()` rule:
removing log-softmax is mathematically valid but can change decisions through
low-precision rounding/ties, so it is not folded into this serving change.
Tokenization remains on the worker and decisions still synchronize with the
CPU once per tick; those remain profiling opportunities.

The qualified workload here is single-GPU greedy text Responses serving on the
measured RTX 5090 configurations. Larger contexts, other hardware and actual
Codex coding-task quality remain outside this qualification. No native Eole
extension rebuild is required; startup still prepares runtime-specific
torch.compile programs and CUDA graphs.


## Automated GPU correctness tests

`eole.tests.test_continuous_gpu` uses small, explicitly initialized Eole decoders
with real FlashAttention and FLA kernels, not a fake runner or downloaded weights.
Both attention-only and GDN models compare mixed-position batched logits and
KV/recurrent/convolution state against independent ordinary decoding with BF16
tolerances. Tests exercise admission into occupied rows, late arrivals, partial
prefill cancellation, slot reuse, compiled replay and the inference-mode contract.
The random-weight scheduler fixture also checks its emitted tokens against solo
inference; this does not assert universal bit-identity on arbitrary checkpoints.

```bash
CUDA_VISIBLE_DEVICES=0 EOLE_TORCH_COMPILE=1 EOLE_COMPILE_MODE=0 \
EOLE_REQUIRE_GPU_TESTS=1 python -m unittest -v \
  eole.tests.test_continuous_gpu \
  eole.tests.test_request_cache.RequestRotaryCudaTests
```

This requires matching Eole CUDA ops, FlashAttention and FLA in the environment.
`EOLE_REQUIRE_GPU_TESTS=1` fails if CUDA or the required compile configuration is
missing, rather than allowing the GPU job to pass with skipped tests. Ordinary
CPU CI still skips the CUDA integration cases.

The **Continuous decoder GPU tests** Actions workflow is prepared for a self-hosted
Linux x64 runner labelled `gpu`, with a compatible Torch/FlashAttention/FLA Python
environment and CUDA toolkit. It builds this checkout's ops in an isolated venv
for the attached GPU. No GPU runner is currently configured: the tests have run
locally, and GPU CI is not yet an active check. After registering that runner,
maintainers can dispatch reviewed refs manually. Set the repository Actions
variable `EOLE_GPU_CI=true` to enable automatic same-repository PR runs. Fork code
never runs automatically on the self-hosted host; manual dispatch is for reviewed
code only. The workflow must be available on the default branch before dispatch.

## Runtime contracts

Compilation keeps normal Dynamo guards. Captured CUDA graphs require model flags,
parameter/cache storage and layouts to remain fixed for the worker lifetime;
change model configuration through unload/reload, not mutation during replay.
The bank still runs all rows when at least two requests are active. Idle-row cost
is a documented trade-off; per-step KV gathering/compaction is outside this change.

Requests with `0 < top_p < 1` are rejected; 0 disables nucleus filtering and 1
keeps the full distribution before greedy selection. Admission requires exactly
one nonempty unpadded source sequence, checked against the collator's `srclen`.
Formatted prompt EOS/turn-boundary IDs remain valid. Other generation routes,
including Chat, Anthropic, native inference and workspace POST routes, return a
clear 409 before model loading or SSE headers on a continuous-only server.

Cooperative cancellation cannot interrupt a hard-hung CUDA kernel. The worker
must retain ownership of state while it runs; a timeout must not release that
state for another request. A hard device hang requires an external process
supervisor to restart the server. There is no in-process CUDA-hang recovery here.
