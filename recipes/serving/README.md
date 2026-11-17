# Codex-shaped serving baseline

This recipe measures the existing Responses API under concurrent load before
adding continuous batching to HTTP. **It does not enable HTTP continuous batching.**
An isolated eager core probe is available below. Streaming
requests currently wait on a per-model Responses lock and a single inference
worker. Non-streaming Chat requests can use whole-request batching; that is not
a scheduler that admits new requests between decode steps.

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
A live cancellation check closed a 1,024-token stream after its first text delta;
the following short request completed in 0.33 seconds.

## Next scheduler milestone

The scheduler must own the one inference worker, not run simultaneous forwards
against the current shared decoder. Introduce request-owned KV caches, GDN
convolution/recurrent state, token history, positions, stopping criteria and
cancellation before removing endpoint serialization. The current decoder builds
ordinary rotary positions from `cache_seqlens[0]`; mixed-age decode rows need
per-request positions and equivalence tests before they can share a batch.

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

An initial **eager-only**, single-pass bank-capacity probe gave:

| Slots | Total output tokens | Aggregate output tok/s |
| --- | ---: | ---: |
| 1 | 128 | 24.9 |
| 2 | 207 | 37.3 |
| 4 | 414 | 71.0 |
| 8 | 779 | 120.1 |

Each cell submits twice as many jobs as slots, cycling three short prompts with
a 64-token output budget. EOS changes total output counts. These rates include
prefill/admission, exclude model loading and initial correctness checks, and
have no HTTP or detokenization cost. They are **not comparable to the compiled
HTTP baseline or the front-page decode table**. The eight-slot runner processes
all eight rows even when some slots are inactive; this establishes batching
and state ownership, not an optimized scheduler or a latency improvement.

Remaining before HTTP integration: compiled step graphs with stable bank
bindings, bounded/fair chunked prefill, streaming delivery and request-local
settings on the worker, and lifecycle/error handling with model unload. A
single-request regression benchmark and mixed-age correctness checks must pass
before replacing endpoint locks. The eager runner is not automatically used
by any public endpoint and must not run alongside another predictor forward.
