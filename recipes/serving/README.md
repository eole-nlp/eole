# Codex-shaped serving baseline

This recipe compares the existing serialized Responses API with an experimental,
opt-in continuous backend. Default execution stays serialized; disconnect cancellation changes as described
below. The new backend
requires a dedicated single-model, single-GPU, ordinary-greedy Responses server;
it is not a general replacement for other Eole prediction/scoring/UI workflows.

The workload uses the same Responses protocol as the [Codex integration](../codex/README.md),
without launching Codex itself. It checks streaming load and a synthetic coding
tool round trip. It is not a coding-quality benchmark or a full agent evaluation.

## Streaming cancellation change

This PR also intentionally changes **ordinary streaming routes**, independently
of the continuous option: a disconnect or early consumer exit now signals
cooperative generation cancellation, instead of only discarding delivery while
the model continues generating. It applies to Responses, Chat, Anthropic and
workspace callers using the shared streaming bridge. Cleanup still drains the
worker; completed segments leave a caller-provided shared event unset. Supplying
the event also enables the existing engine request-setting restoration for routes
that previously supplied none, so their streaming overrides no longer leak into
later requests.

## Run locally

Use a dedicated server and converted Qwen3.8 checkpoint. The baseline deliberately
disables speculation and prefix reuse; its native vocabulary head stays BF16.
The model path and server port below are independent of your workspace server.

```bash
export EOLE_MODEL_DIR=/mnt/InternalCrucial4/LLM_work
export QWEN38_MODEL="$EOLE_MODEL_DIR/Qwen3.8-27B-INT4"
export CUDA_VISIBLE_DEVICES=0
export EOLE_TORCH_COMPILE=1 EOLE_COMPILE_MODE=0
export TRITON_DEFAULT_FP_FUSION=1
python -m eole.bin.main serve -c recipes/serving/serve.yaml \
  --host 127.0.0.1 --port 5019
```

After startup preparation finishes, in another terminal with `aiohttp` installed:

```bash
python recipes/serving/benchmark_responses.py \
  --base-url http://127.0.0.1:5019/v1 --model qwen3.8-27B \
  --concurrency 1 2 4 8 --repeats 2 --max-tokens 128 \
  --output /tmp/eole-serving-baseline.json
```

The benchmark performs one unmeasured text warmup, then two scenarios:

- **Text:** concurrent requests for a small Python function and tests.
- **Tool:** request a `read_fixture` call, validate its arguments, wait 50 ms to
  simulate tool work, return an in-memory `counter.py` fixture, then request a
  patch and tests. No generated code is executed and no user file is read.

Each cell runs `concurrency * repeats` jobs with that many agents in flight.
A tool job has two HTTP requests and retains its client slot during tool work.
The two-turn history is resent explicitly (`store: false`), as required by Eole's
stateless endpoint. Each job has a distinct identifier; instruction/tool prefixes
are shared, but complete prompts are not identical. The short fixture is well
below Codex's typical large system/tool prompts.

Output records include aggregate output tokens/second, first-output p50/p95,
request p95, whole-job p50/p95, and inter-chunk p95. First output excludes protocol
announcements such as `response.created` and includes text or function-argument
deltas. Chunk gaps **are not per-token latency**: one SSE event can contain many
tokens. Token totals come from terminal response usage, not chunk counts.
Missing terminal events, HTTP errors, missing usage and invalid tool calls fail
the run rather than silently inflating throughput. `incomplete_requests` records
responses truncated at the configured output budget.

The JSON retains per-job/per-turn timing for later comparisons. A cell's wall
clock includes prefill, queueing, response delivery and simulated tool delays;
it excludes model loading, initial compilation and the unmeasured warmup.
These rates must not be compared directly with the front-page decode-only table.

For an authenticated server, pass `--api-key-env NAME_OF_KEY_VARIABLE`; the key
value is neither printed nor saved. HTTP timeouts default to 600 seconds.
Use Ctrl+C in the server terminal to release its GPU model afterward.

## Initial RTX 5090 baseline

Measured 17 November 2026 from main `50b4371e` plus this recipe/cancellation
fixes: PyTorch 2.15.0+cu132, Triton 3.9.0, FLA 0.6.0, FlashAttention 2.8.3,
compile mode 0, FP fusion enabled, native BF16 vocabulary head, no MTP/DFlash2,
4,096-token context capacity, 256-token prefill chunks, gate/KVQ fusion enabled.
Two jobs per concurrent client, 128-token output cap per request, localhost.
These are small baseline samples, not a statistical performance guarantee.

| Workload | Clients | Aggregate output tok/s | First-output p95 (s) | Job p95 (s) |
| --- | ---: | ---: | ---: | ---: |
| Text | 1 | 67.9 | 0.23 | 1.90 |
| Text | 2 | 68.3 | 2.10 | 3.76 |
| Text | 4 | 67.3 | 5.97 | 7.66 |
| Text | 8 | 67.3 | 13.66 | 15.35 |
| Read-fixture tool loop | 1 | 55.3 | 0.69 | 2.79 |
| Read-fixture tool loop | 2 | 56.0 | 2.70 | 5.52 |
| Read-fixture tool loop | 4 | 55.6 | 6.69 | 11.19 |
| Read-fixture tool loop | 8 | 56.2 | 14.14 | 22.02 |

All text and final patch requests reached the short output cap. The first tool
call completed and its name/arguments/call ID were checked. Increasing client
count did not increase GPU throughput; it increased waiting time. That establishes
a scheduling bottleneck rather than evidence for another projection-kernel change.
See the valid live cancellation/fairness checks for the continuous backend below.

## Scheduler design

The scheduler owns the one inference worker rather than running simultaneous
forwards against a shared decoder. Request-owned KV/GDN state, positions, token
histories, stopping criteria and cancellation allow admission and retirement at
decode boundaries. An explicit step path supplies per-row rotary positions;
ordinary predictor dispatch retains its existing behaviour.

Start with single-GPU, greedy, text-only ordinary decoding, bounded admission
and prefill chunks that yield between decode rounds. Retire completed/cancelled
slots and admit waiting work at step boundaries. Preserve the existing fast
single-request path and reject unsupported settings explicitly. Paged KV storage,
batched speculation and broader sampling can follow after isolation, mixed-length
correctness, cancellation and fairness are demonstrated. Reuse this recipe to
compare latency and aggregate throughput at every milestone.


## Eager continuous core probe

`eole.predict.continuous.ContinuousGreedy` is an isolated fixed-slot runner owned
by one inference worker. Requests can join, finish or cancel between ticks;
freed slots admit waiting work. A `RequestCache` transfers ownership of KV,
convolution and GDN recurrent tensors, positions and padding metadata without
copying the bank on each decode step. Admission copies a separately prefilled
request into its slot once, overwriting the previous occupant's complete state.
Completed handles can be released with `forget()` after output consumption.
Ordinary predictor dispatch and HTTP serialization remain unchanged.

The explicit decoder `forward_step` uses each row's cache position for rotary
embeddings. Existing uniform-position calls retain their previous table path.
Both CUDA rotary and the PyTorch/compilation branch accept per-row tables.
No kernel rebuild is required for these Python changes.

```bash
CUDA_VISIBLE_DEVICES=0 EOLE_TORCH_COMPILE=0 \
python recipes/serving/check_continuous.py --model "$QWEN38_MODEL" \
  --benchmark --output /tmp/eole-continuous-core.json
```

The probe now submits ten requests with 256-token budgets, including delayed
arrivals, long prefills, queued/prefill/decode cancellation and repeated slot
reuse. Seven complete, generating 1,558 tokens in the recorded run. It records
actual prefill boundaries, arithmetic widths, token sequences and first-mismatch
hidden/log-probability diagnostics. See the exactness findings below.

The dedicated service owns its engine worker for the entire model lifetime;
another predictor must never run against that engine concurrently.

## Experimental compiled Responses server

Start this instead of the baseline server, on the same localhost port:

```bash
EOLE_TORCH_COMPILE=1 EOLE_COMPILE_MODE=0 \
python -m eole.bin.main serve -c recipes/serving/serve_continuous.yaml \
  --host 127.0.0.1 --port 5019
```

Use the same benchmark command above. The YAML adds a model-level `continuous`
section with four decode slots, a bounded 32-request pending budget and
128-token active prefill chunks and a configurable 512-token idle chunk. Startup
compiles/captures the fixed bank and per-slot
singleton views before readiness. One active request uses a singleton graph;
multiple active requests share the bank graph. Singleton views share cache
storage rather than duplicating the KV bank. Initial preparation took 62 seconds
in the measured run, excluding model loading. Failed graph capture fails
preparation rather than advertising a compiled backend while falling back.

The worker alternates one decode round with at most one bounded prefill chunk.
`continuous.prefill_chunk_size` caps chunks while any decode request is active;
`continuous.prefill_idle_chunk_size` permits larger chunks when none is active.
Omitting the idle option preserves the active chunk size. Set idle equal to active
to keep prefill boundaries independent of whether another request is decoding;
batching can still change floating-point results. It must be at least as
large as the active value. The recipe uses 128 active / 512 idle tokens.
Active requests continue decoding while a long prompt is prepared; incoming
prefills currently run FIFO, one at a time. This is not priority scheduling of
short prompts around other queued prefills. Cancellation handles queued,
partially prefilled and decoding requests. Completion returns admission capacity.
A worker failure fails all affected streams and requires a model reload. Server
shutdown stops the worker, closes graphs and unloads the model before process exit.

Only text-only greedy Responses requests are qualified. Sampling, penalties,
MTP/DFlash2, prefix reuse, tensor parallelism and general scoring/prediction calls
are unsupported in this mode. Use `temperature: 0` or the greedy server default;
unsupported request settings are rejected. The server must contain one model and
must not use `model_policy: single`, whose global lease would serialize requests.
Existing server YAMLs without `continuous` keep their current paths.

This small-context recipe is for protocol/load testing. Actual Codex sessions can
carry much larger system/tool prompts; align context/output metadata and verify
VRAM before running real coding tasks. The tests here do not establish coding
quality or successful end-to-end Codex task completion.

### Latest four-slot versus eight-slot HTTP measurements

Same RTX 5090, target, native BF16 vocabulary head, 4,096-token context capacity,
no speculation/prefix reuse, 128-token output cap and two jobs per client.
Both continuous profiles use 128 active / 512 idle prefill tokens. The earlier
serialized baseline uses 256-token chunks. All figures include queueing, prefill,
detokenization and HTTP delivery. These are small initial samples, not a general
speed guarantee or coding-quality evaluation.

| Workload | Clients | Serialized tok/s | 4 slots tok/s | 8 slots tok/s | First-output p95, 4 / 8 slots (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Text | 1 | 67.9 | 73.6 | 73.8 | 0.19 / 0.20 |
| Text | 2 | 68.3 | 106.6 | 101.9 | 0.40 / 0.42 |
| Text | 4 | 67.3 | 182.6 | 174.1 | 0.77 / 0.82 |
| Text | 8 | 67.3 | 184.5 | 270.2 | 3.81 / 1.66 |
| Tool loop | 1 | 55.3 | 67.2 | 66.9 | 0.48 / 0.49 |
| Tool loop | 2 | 56.0 | 85.7 | 82.3 | 0.98 / 1.00 |
| Tool loop | 4 | 55.6 | 114.7 | 109.6 | 2.30 / 2.48 |
| Tool loop | 8 | 56.2 | 115.3 | 141.9 | 5.22 / 5.19 |

Eight slots help when eight clients are busy, but are slightly slower at two or
four clients because partly occupied multi-request banks still execute all rows.
Singleton graphs preserve the one-client path. Keep four slots as the recipe
default; use eight for measured higher concurrency and adequate memory.

### Measured VRAM: context and head precision matter

These numbers include the INT4 target and its **native BF16 vocabulary head**,
with a **4,096-token context allocated in every slot**. Serving peaks cover the
short text/tool workload above; they are not bounds for arbitrary long prompts.

| Slots | Startup allocated (GiB) | Serving peak allocated (GiB) | Serving peak reserved (GiB) | Sampled whole-device peak (GiB) |
| --- | ---: | ---: | ---: | ---: |
| 4 | 19.61 | 20.13 | 20.61 | 21.47 |
| 8 | 20.91 | 21.44 | 21.95 | 22.82 |

PyTorch allocated/reserved figures exclude some CUDA/library allocations;
whole-device values were sampled with nvidia-smi every 0.5 seconds and may miss
brief spikes. Leave headroom for display use, library versions and larger
prefills. Eight slots were tested on the 32-GiB RTX 5090; a 24-GiB device would
be tight and was not validated by this run.

For this checkpoint, 16 full-attention layers × 4 KV heads × 256 dimensions ×
K/V × 2-byte BF16 = **64 KiB per context token per slot**. Eight slots therefore
allocate 2 GiB of KV at 4K context, 16 GiB at 32K, or 32 GiB at 64K, before
weights, GDN state, graphs and temporary buffers. Do not use the 4K memory result
to size a long-context Codex server. This backend is not paged.

### Idle prefill sizing

A separate idle probe encoded a 1,817-token prompt once, then timed cache
allocation, prefill/admission and the first token, with no HTTP/tokenization.
Each size had one warmup and two measured repeats; means are below:

| Idle chunk | Warmed seconds to first token |
| --- | ---: |
| 128 | 1.270 |
| 256 | 0.677 |
| 512 | 0.466 |
| 1024 | 0.460 |

512 removes most of the small-chunk overhead; 1024 saved only about another 1%
here. The active-decode chunk remains 128. Reproduce with `--prefill-sweep` on
`check_continuous.py`. This short-context result does not establish performance
for tens of thousands of tokens.

### Longer correctness findings: solo bit-identity is not guaranteed

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

The HTTP path remains experimental. Broader contexts, hardware and actual Codex
coding tasks remain unqualified. No native Eole extension rebuild is required;
startup still prepares runtime-specific torch.compile programs and CUDA graphs.
