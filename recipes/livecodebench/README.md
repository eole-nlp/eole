# Coding correctness with LiveCodeBench

Generate one greedy Python solution per problem through Eole's
OpenAI-compatible API, then grade saved solutions with the official
[LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) executable tests.
This measures model code generation, not repository editing or agent tool use.
The initial recipe uses `release_v6`, February–April 2025. This is a fixed
historical window, not a contamination-free claim for newer models. Select and
report a window after the model's training cutoff when one is available.

## Install the evaluator separately

Keep evaluator dependencies out of the Eole serving environment. Python 3.11
or 3.12 is suitable. The upstream revision is checked by the runner; do not use
an unpinned checkout or modify it.

```bash
export LCB_DIR=/tmp/LiveCodeBench
git clone https://github.com/LiveCodeBench/LiveCodeBench.git "$LCB_DIR"
git -C "$LCB_DIR" checkout 28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24
python -m venv /tmp/eole-lcb-venv
/tmp/eole-lcb-venv/bin/pip install 'datasets>=3.2,<4' numpy tqdm PyYAML
```

The pinned upstream dataset loader requires `trust_remote_code=True`; datasets
4 removed support for that loader. We import upstream code directly without
installing its optional vLLM/cloud-provider backends. Dataset preparation may
execute Hugging Face dataset loader code and deserialize private test data;
prepare only with trusted upstream sources in an isolated environment. The first
download can require several GB even for a small selected subset.

## Configure and prepare

Run from the Eole repository root. Copy `benchmark.yaml` locally to select
output directory, server/model ID, release, date window, output budget, and CPU
checker concurrency. Paths resolve against the invocation directory. Start with
`limit: 5` for a smoke run; a subset score is not a full benchmark score.
Preparation downloads the dataset, uses upstream's generic chat prompt, and
saves prompts and test cases in `tasks.json`. It does not load a model.

```bash
/tmp/eole-lcb-venv/bin/python recipes/livecodebench/run.py prepare \
  -c recipes/livecodebench/benchmark.yaml
```

## Generate with Eole

Start a converted model using an Eole server YAML, for example
[Qwen3.8 serve.yaml](../qwen38/serve.yaml). Its configured context must fit the
prompt plus output budget; the benchmark requests up to 8192 output tokens.
Configure greedy decoding (`beam_size: 1`, `top_k: 1`, `n_best: 1`) in the server.
For compiled inference:

```bash
EOLE_TORCH_COMPILE=1 EOLE_COMPILE_MODE=0 \
  eole serve -c recipes/qwen38/serve.yaml --host 127.0.0.1 --port 5010
```

In another terminal, from the repository root:

```bash
/tmp/eole-lcb-venv/bin/python recipes/livecodebench/run.py generate \
  -c recipes/livecodebench/benchmark.yaml
```

Requests contain only the official prompt, never hidden test cases. One request
is sent at a time with temperature zero; raw text, request wall time, API usage,
and finish reason are saved after every successful request. A request failure
stops generation and preserves the partial artifact. This first version does
not resume partial runs; use a new run directory. Existing artifacts are not
overwritten during preparation/generation; grading can be rerun. Record server YAML, model provenance, Eole commit, compilation,
MTP, chat-template/thinking settings, GPU and software versions alongside runs.

Eole's current OpenAI endpoint estimates token usage and reports `stop` even
when a token limit may have been reached. Saved API metadata therefore cannot
establish exact tokens/s or truncation. Inspect outputs and server logs; retain
incomplete solutions as failures rather than excluding them. Wall times include
API/prefill overhead and first-request warmup; they are not steady-state decode
benchmarks.

## Grade saved solutions

**The upstream checker executes generated Python and is not a security sandbox.**
Run grading inside a disposable container/VM with no network, credentials, host
home mounts, or GPU access, and with CPU/memory/process limits. Copy the prepared
artifacts, recipe, pinned upstream checkout, and evaluator dependencies into
that environment. Then run there (adjust YAML paths to those copies):

```bash
python recipes/livecodebench/run.py evaluate \
  -c recipes/livecodebench/benchmark.yaml --allow-code-execution
```

Grading does not call the server or require a GPU. It verifies dataset hashes
and task ordering, extracts the last fenced code block using upstream's
extractor, and executes public and private tests using upstream's checker.
Missing/malformed code is evaluated as a failed solution. It writes
`evaluation.json` with pass@1, difficulty breakdown, per-task test results,
extracted code, and total request wall time. With one completion per task,
pass@1 is the fraction of problems solved. The initial greedy protocol differs
from upstream's default multi-sample protocol; compare only matching release,
window, prompt, decoding, thinking and token-budget settings.

For eager/compiled or MTP comparisons, prepare the same selection in separate
run directories, change the server configuration between runs, and evaluate
all selected tasks. Compare correctness and latency together; greedy text can
differ across backends. There are no published Eole scores yet.

## Validation status

Tested dataset preparation on two real `release_v6` problems in the configured
window. A local HTTP fixture exercised generation and grading end to end with
the pinned upstream checker: a known correct solution passed, an incorrect
solution failed, difficulty aggregation matched, and a changed dataset artifact
was rejected. Unit tests cover partial-result preservation and execution opt-in;
recipe YAML validation passes. A live Eole GPU benchmark has not yet been run.
