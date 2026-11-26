# Codex coding workload benchmark

Codex CLI edits two small, disposable Python repositories through Eole's
Responses API. One task changes integer discount arithmetic. The other repairs
closed-interval merging. Both require tool calls, a file edit, and a final
response. The benchmark reruns the original tests independently of Codex's
reported success.

## Run a profile

Use a CUDA-enabled Eole installation with FlashAttention and FLA, a converted
[Qwen3.8-27B INT4 checkpoint](../qwen38/README.md#download-the-pre-converted-eole-checkpoint),
and its [DFlash2 drafter](../qwen38/dflash2/README.md) for that profile. The local
runner needs the Codex CLI, Git, Bubblewrap with working Linux user namespaces,
Python at `/usr/bin/python3`, and `aiohttp`. Use a GPU with room for four 12,288-token slots and the drafter.

From the repository root, select `serialized`, `greedy`, `mtp`, or `dflash2`:

```bash
export EOLE_MODEL_DIR=/path/to/eole-models
export QWEN38_MODEL="$EOLE_MODEL_DIR/Qwen3.8-27B-INT4"
export QWEN38_DFLASH2="$QWEN38_MODEL/dflash2"
export CUDA_VISIBLE_DEVICES=0
export EOLE_TORCH_COMPILE=1 EOLE_COMPILE_MODE=0 TRITON_DEFAULT_FP_FUSION=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

profile=serialized
python recipes/codex/benchmark_profile.py "$profile" > "/tmp/eole-codex-$profile.yaml"
eole serve -c "/tmp/eole-codex-$profile.yaml" --host 127.0.0.1 --port 5019
```

After the server is ready, run from a second terminal:

```bash
python recipes/codex/benchmark.py --base-url http://127.0.0.1:5019/v1 \
  --concurrency 1 4 --timeout 100 --output /tmp/eole-codex-serialized
```

Use a fresh output directory and restart the server for each profile. Pass
`--codex /path/to/codex` if your CLI is outside `PATH`. The runner gives each
job its own Codex home and repository, executes commands with Codex's
workspace-write sandbox, and grades with read-only source and test mounts in
an offline Bubblewrap sandbox. It saves the CLI events, grading logs, and
per-request observations locally. The saved events can contain tool arguments
and output from these fixtures.

## Metrics

- **Successful jobs** have passing independent tests, a clean CLI exit, and
  completed Responses streams. The JSON preserves these three checks separately
  so an edited file can pass even if the agent times out or the API fails later.
- **Successful jobs/min** divides successful jobs by the full wave wall time.
  Each wave starts up to the requested number of clients and waits for all of
  them before starting the next. One warmup job is excluded.
- **First content** is measured per HTTP request, from relay submission to the
  first text or tool-input delta. It includes queueing and prefill. Eole buffers
  a tool call until the parser can emit it, so this can also include some
  generation. Use this metric for client-visible latency.
- **Job p95** includes the entire Codex process, its tool commands, and failed
  jobs. This uses nearest rank; with four or fewer jobs it is the slowest job.

All profiles use the same target, native BF16 vocabulary head and KV cache,
greedy selection, a 2,048-token per-request generation budget, and 256-token
prefill chunks. Continuous profiles use four slots without retained prompt
prefixes. The drafter config determines the DFlash2 draft count. Startup time
and cold compiler cost are outside the measured job waves. These small fixtures
check live tool execution and request handling on a fixed batch of jobs.

## Reference results

RTX 5090 at a 400 W power limit, Eole `0.6.8` runtime, Codex CLI
`0.171.0-alpha.13`, Torch 2.15.0+cu132, Triton 3.9.0, FlashAttention 2.8.3,
and FLA 0.6.0. The target was RedHat Qwen3.8-27B INT4. MTP used three drafts.
DFlash2 used seven drafts and an INT8 drafter. One warmed run per profile,
with both tasks once at one client and twice at four clients.

| Profile | Clients | Jobs passed | Successful jobs/min | First content p50 / p95 (s) | Job p95 (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Serialized greedy | 1 | 2/2 | 2.49 | 3.19 / 4.68 | 24.44 |
| Continuous greedy | 1 | 2/2 | 2.68 | 3.05 / 4.63 | 22.68 |
| Continuous MTP | 1 | 2/2 | 3.08 | 2.58 / 3.50 | 19.87 |
| Continuous DFlash2 | 1 | 2/2 | 3.34 | 2.35 / 3.09 | 18.40 |
| Serialized greedy | 4 | 4/4 | 2.66 | 13.12 / 16.98 | 90.05 |
| Continuous greedy | 4 | 4/4 | 3.42 | 9.49 / 16.80 | 69.96 |
| Continuous MTP | 4 | 4/4 | 3.62 | 9.50 / 13.26 | 66.23 |
| Continuous DFlash2 | 4 | 4/4 | 3.80 | 8.79 / 11.64 | 62.98 |

All 24 jobs passed and each used six Responses requests. Rendered input sizes
were 5,232–6,695 tokens per request. The four-client waves generated 2,468 /
2,609 / 2,211 / 2,068 output tokens respectively for serialized, greedy,
MTP, and DFlash2. These are whole-job measurements with varying generated
output lengths. The small sample limits how precisely we can rank MTP and
DFlash2.

At four clients, DFlash2 completed 43% more jobs per minute than serialized
decoding and reduced the slowest job by 30%. Compared with MTP, it completed
about 5% more jobs per minute. Peak allocated GPU memory during serving was
22.39 GiB for continuous greedy, 24.23 GiB for MTP, and 26.47 GiB for DFlash2,
as reported by PyTorch. Greedy batching improved completed jobs while leaving
request p95 first-content latency close to the serialized baseline.
