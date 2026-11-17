# Codex-shaped serving baseline

This recipe compares the existing serialized Responses API with an experimental,
opt-in continuous backend. Default server behaviour is unchanged. The new backend
requires a dedicated single-model, single-GPU, ordinary-greedy Responses server;
it is not a general replacement for other Eole prediction/scoring/UI workflows.

The workload uses the same Responses protocol as the [Codex integration](../codex/README.md),
without launching Codex itself. It checks streaming load and a synthetic coding
tool round trip. It is not a coding-quality benchmark or a full agent evaluation.

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

The probe checks four completed short Qwen requests against independent solo
decoding, including different prompt lengths, a late arrival, queued admission,
cancellation and reuse of the cancelled request's slot. On the RTX 5090 all four
matched token-for-token (92 completed output tokens total). This finite check
is not a universal cross-batch bit-identity guarantee.

The optional `--benchmark` probe measures core capacity without HTTP delivery;
its results must not be compared directly with the compiled HTTP table below.

The dedicated Responses backend below adds compiled step graphs, per-slot
singleton graphs and bounded prefill chunks to this core. It owns the engine
worker for the entire model lifetime; another predictor must never use the same
engine concurrently.

## Experimental compiled Responses server

Start this instead of the baseline server, on the same localhost port:

```bash
EOLE_TORCH_COMPILE=1 EOLE_COMPILE_MODE=0 \
python -m eole.bin.main serve -c recipes/serving/serve_continuous.yaml \
  --host 127.0.0.1 --port 5019
```

Use the same benchmark command above. The YAML adds a model-level `continuous`
section with four decode slots, a bounded 32-request pending budget and
128-token prefill chunks. Startup compiles/captures the fixed bank and per-slot
singleton views before readiness. One active request uses a singleton graph;
multiple active requests share the bank graph. Singleton views share cache
storage rather than duplicating the KV bank. Initial preparation took 62 seconds
in the measured run, excluding model loading. Failed graph capture fails
preparation rather than advertising a compiled backend while falling back.

The worker alternates one decode round with at most one bounded prefill chunk.
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

### Compiled HTTP results

Same RTX 5090, target, native BF16 vocabulary head, prompts, 128-token output cap,
client counts and two jobs per client as the baseline. The new backend uses
128-token prefill chunks versus the baseline's 256. No speculation or prefix
reuse. Values below include queueing, prefill, detokenization and HTTP delivery.
These are initial small samples, not a broad throughput guarantee.

| Workload | Clients | Serialized tok/s | Continuous tok/s | First-output p95, old → new (s) | Job p95, old → new (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Text | 1 | 67.9 | 74.4 | 0.23 → 0.19 | 1.90 → 1.72 |
| Text | 2 | 68.3 | 107.3 | 2.10 → 0.39 | 3.76 → 2.48 |
| Text | 4 | 67.3 | 183.4 | 5.97 → 0.79 | 7.66 → 3.07 |
| Text | 8 | 67.3 | 184.7 | 13.66 → 3.84 | 15.35 → 6.60 |
| Tool loop | 1 | 55.3 | 59.2 | 0.69 → 0.65 | 2.79 → 2.60 |
| Tool loop | 2 | 56.0 | 78.4 | 2.70 → 1.14 | 5.52 → 4.19 |
| Tool loop | 4 | 55.6 | 111.8 | 6.69 → 2.51 | 11.19 → 6.44 |
| Tool loop | 8 | 56.2 | 113.9 | 14.14 → 5.22 | 22.02 → 12.42 |

All tool-call arguments and call IDs passed the benchmark checks. A live fairness
check recorded 20 short-request output events during a long request's 1.92-second
prefill window. The long stream was then closed after its first text event,
and a following request completed in 0.31 seconds. Unlike the earlier baseline
probe, this used a valid output budget and a confirmed live stream. A live
non-greedy request was rejected with HTTP 400.

The compiled mixed-arrival diagnostic uses the same prefill chunk boundaries in
its solo oracle. Chunking itself can change floating-point results; an unchunked
oracle is not a valid exact-output reference for this test. Run it with:

```bash
EOLE_TORCH_COMPILE=1 EOLE_COMPILE_MODE=0 \
python recipes/serving/check_continuous.py --compiled-step \
  --model "$QWEN38_MODEL" --output /tmp/eole-continuous-check.json
```

The HTTP path remains experimental. Broader long-context/long-output correctness,
real Codex workloads, more hardware and larger latency samples remain necessary
before making it a default. No native Eole extension rebuild is required; startup
still prepares runtime-specific torch.compile programs and CUDA graphs.
