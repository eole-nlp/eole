# Qwen3.8-27B and native MTP speculative decoding

Run from the repository root on a checkout containing MTP support (after the
0.6.0 release). This recipe uses the model's native multi-token prediction head
to draft tokens, then verifies a chunk with the main model. No separate draft
model is needed. Qwen3.8 is loaded through the vision-language architecture:
its MTP heads support text inference, but auxiliary MTP training loss is not
implemented for that architecture. The checkpoint-compatibility training warning
does not mean its inference heads are missing.

## Hardware and checkpoint

An RTX 5090 has 32 GB VRAM. Full BF16 27B weights exceed that capacity; use a
compatible INT4 checkpoint with its **MTP weights retained**, or multiple GPUs
for an appropriately configured BF16 deployment. This example is single-GPU;
it does not establish MTP support for tensor-parallel inference.

Install Eole in a CUDA-enabled environment with the CUDA toolkit and build tools
from the [installation guide](../../README.md#installation). For this configuration
install FlashAttention and the hybrid-attention kernels used by Qwen:

```bash
MAX_JOBS=2 python -m pip install -e . --no-build-isolation
MAX_JOBS=2 python -m pip install flash-attn --no-build-isolation
python -m pip install fla-core
MAX_JOBS=2 python -m pip install causal-conv1d --no-build-isolation
```

INT4 AutoRound/GPTQ checkpoints require a supported packing layout and Eole's
Marlin CUDA extension. Build/install Eole while CUDA and the CUDA toolkit are
available; inspect backend diagnostics before measuring. Installing a package
does not prove its kernels are selected.

The RTX 5090 tests used an Eole conversion of
`Frozenlock/Qwen3.8-27B-int4-Autoround`. To prepare that checkpoint from Hugging
Face, run from the repository root in your Eole environment:

```bash
export EOLE_MODEL_DIR=/path/to/models
export QWEN38_MODEL="$EOLE_MODEL_DIR/Qwen3.8-27B-int4-Autoround"
python eole/bin/main.py convert HF \
  --model_dir Frozenlock/Qwen3.8-27B-int4-Autoround \
  --output "$QWEN38_MODEL" \
  --token "$HF_TOKEN"
```

Use a new output directory for conversion. `HF_TOKEN` is your Hugging Face access
token; omit `--token "$HF_TOKEN"` when authentication is unnecessary. The
converter accepts either a Hugging Face repository ID or a local HF model
directory as `--model_dir`. It converts the existing INT4 checkpoint into Eole's
format; this command does not quantize BF16 weights.

If you already have that converted checkpoint, skip conversion and set the two
path variables above to its location. The GPU tests used an existing converted
artifact; conversion itself was not rerun as part of those tests.

### Compressed-tensors INT4 checkpoints

The HF converter also accepts the symmetric grouped W4A16 `pack-quantized`
format used by `RedHatAI/Qwen3.8-27B-INT4`:

```bash
python eole/bin/main.py convert HF \
  --model_dir RedHatAI/Qwen3.8-27B-INT4 \
  --output "$EOLE_MODEL_DIR/Qwen3.8-27B-RedHatAI-INT4" \
  --dtype bf16 \
  --check-tensors
```

The converter transposes packed INT4 words and scales into Eole's GPTQ layout;
it does not dequantize and re-quantize the weights. Inference uses the existing
AutoRound/GPTQ backend selection, including Marlin when available. Exact module
paths from the checkpoint preserve unquantized projections, vision weights, and
MTP heads. GDN kernels and normalization are unchanged.

Supported exports have one `Linear` weight-only configuration group, symmetric
4-bit integer weights, and group size 32, 64, or 128. Static activation ordering
without saved group indices is accepted. Asymmetric weights, activation
quantization, reordered groups, transformed/sparse exports, and packed weights
requiring sliced or unmapped module transformations are rejected.

FP8 KV-cache scales in the source are not used: conversion logs a warning and
Eole retains its floating-point KV cache. `--check-tensors` may list those cache
scale tensors as unused. This is weight-format support, not FP8-cache support.
CPU tests verify packed-value equivalence, sharded conversion, and mixed-precision
module selection. A full RedHatAI checkpoint GPU run has not yet been validated.

For a BF16 deployment with sufficient memory, use `Qwen/Qwen3.8-27B` as the
source and a separate output directory. Quantized checkpoints are not all
equivalent: some omit MTP tensors. Conversion cannot recreate missing trained
heads. Check the converted artifact before loading:

```bash
python recipes/qwen38/check_checkpoint.py "$QWEN38_MODEL"
```

The checker requires decoder `num_mtp_heads > 0`, saved `mtp_heads.*` tensors,
and a chat template. It prints tensor names and quantization metadata without
loading weights onto the GPU. A successful check establishes artifact contents,
not GPU kernel compatibility or model quality.

## Serve with MTP

```bash
eole serve -c recipes/qwen38/serve.yaml --host 127.0.0.1
```

`top_k: 1` deliberately selects greedy decoding, including for API clients that
send a nonzero temperature. This recipe does not use sampling. It sets one
sequence/hypothesis and disables prefix caching/chunked prefill to provide a
simple baseline. Start with the conservative 32K context and lower it if VRAM
is tight; the model's advertised maximum context is not a VRAM guarantee.

```bash
curl --fail-with-body http://127.0.0.1:5000/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27B","max_tokens":256,"messages":[{"role":"user","content":"Write a short Python function that reverses a string."}]}'
```

Look for `MTP draft acceptance` in server logs. A fallback warning means the
request used ordinary decoding; merely setting the flag does not demonstrate
speculation. Image inputs, batches larger than one, beam search, sampling,
forced prefixes, and unsupported decoding constraints currently fall back.

For a real coding client continue with [Claude Code](../claude-code/README.md).

## Compare against ordinary greedy decoding

Stop the server first so only one process owns the checkpoint. The two inference
YAML files use the same model, prompt, precision, and decode settings; only MTP
and the output filename differ. The prompt fixture is one preformatted Qwen chat
per line, with Eole's `｟newline｠` marker for embedded newlines.

```bash
eole predict -c recipes/qwen38/predict.yaml
eole predict -c recipes/qwen38/predict-mtp.yaml
```

These save `/tmp/qwen38-baseline.txt` and `/tmp/qwen38-mtp.txt`. Compare the text
and inspect `report_time` / draft-acceptance logs. For repeated warm measurements,
the benchmark reads those **same inference YAML files**:

Run all four configurations explicitly, so inherited environment variables do
not change the comparison. Stop other GPU model workloads first. From the
repository root, with `QWEN38_MODEL` set:

```bash
BENCH_DIR="/tmp/eole-qwen38-bench-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BENCH_DIR"
nvidia-smi > "$BENCH_DIR/gpu.txt"
git rev-parse HEAD > "$BENCH_DIR/commit.txt"

for COMPILE in 0 1; do
  if [ "$COMPILE" = 0 ]; then MODE=eager; else MODE=compiled; fi
  for PATH_MODE in baseline mtp; do
    if [ "$PATH_MODE" = baseline ]; then
      CONFIG=recipes/qwen38/predict.yaml
    else
      CONFIG=recipes/qwen38/predict-mtp.yaml
    fi
    EOLE_TORCH_COMPILE="$COMPILE" EOLE_COMPILE_MODE=0 \
      python recipes/qwen38/benchmark.py -c "$CONFIG" --runs 5 \
      --output "$BENCH_DIR/$MODE-$PATH_MODE.json" \
      > "$BENCH_DIR/$MODE-$PATH_MODE.log" 2>&1 || exit 1
  done
done
echo "$BENCH_DIR"
```

Each invocation loads one engine, excludes one warmup, and measures five runs
of the same prompt. Compile mode 0 requests CUDA graphs; backend diagnostics do
not prove graph capture. Check measured runs for remaining warmup or timing
outliers. Outputs include wall time, peak allocated VRAM, decoded text, and
approximate retokenized output counts. Use Eole's logs for internal generated
counts, decode timing, and acceptance; retokenization can change token counts.
EOS can end a completion early, so verify equal generated lengths. Request wall
time includes prefill, decoding, and other overhead. `report_time` also enables
MTP GPU phase profiling, whose instrumentation is included in these timings.
The harness fails if an MTP run reports fallback or never logs draft acceptance.

Record GPU/driver, PyTorch/kernel versions, checkpoint ID, precision,
prompt/output lengths, compile mode, and draft acceptance with shared results. Try `self_speculative_num_tokens: 1`,
`2`, and `3` in a local YAML copy; more drafts can be slower when rejected.

## Run the complete local validation

`validation.yaml` names the checkpoint, baseline/MTP inference YAML files,
server YAML, loopback port, output directory, and optional Claude Code check.
Set `QWEN38_MODEL` as above or edit `model_path` in a local copy. Run from the
repository root in a terminal that can access the GPU:

```bash
python recipes/qwen38/run_validation.py -c recipes/qwen38/validation.yaml
```

This runs eager baseline/MTP benchmarks sequentially, starts the configured
loopback server, runs the API smoke checks, and optionally asks Claude Code to
read a sentinel file in an isolated directory (`claude: true`). It stops only
the server it started. Logs, JSON results, and a `SUCCESS` marker are saved in
the configured output directory. A failure preserves logs and exits nonzero.
Set `claude: false` for API-only checks. Use a fresh output directory per run.
Inspect `comparison.json`, MTP acceptance in `server.log`, and the Claude Read
tool block before calling the integration validated. The sentinel exercise does
not establish a complete file-editing workflow; use the Claude recipe's coding
task for that additional check.

## Measured RTX 5090 example (2026-10-07)

The recipe's prompt fixture was run at commit
`ff2ab2eb3edfe5c09e449284b62c16f050973081` with the local converted
`Frozenlock/Qwen3.8-27B-int4-Autoround` checkpoint, INT4 weights / BF16 compute,
PyTorch 2.12.1+cu132, FlashAttention 2.8.3, FLA 0.5.2, and Triton 3.7.1.
The driver was 610.43.02 and the pre-run power limit was 575 W; sustained power
and clocks were not recorded. One warmup was excluded per configuration; each
of five measured runs generated 256 tokens with batch size one. Compiled runs
used mode 0; CUDA graph capture was requested but not verified.

| Configuration | Median request wall time | Median reported decode throughput | End-to-end throughput | Maximum peak allocated GPU memory |
|---|---|---|---|---|
| Eager, ordinary greedy | 11.12 s | 23.3 tokens/s | 23.0 tokens/s | 19.25 GB |
| Eager, MTP (three drafts) | 3.85 s | 69.2 tokens/s | 66.4 tokens/s | 19.65 GB |
| Compiled, ordinary greedy | 4.24 s | 63.0 tokens/s | 60.3 tokens/s | 23.59 GB |
| Compiled, MTP (three drafts) | 2.64 s | 115.1 tokens/s | 97.0 tokens/s | 23.92 GB |

End-to-end throughput is 256 divided by median request wall time; reported
decode throughput comes from Eole's internal timing and excludes prefill and
some request overhead. Memory is PyTorch peak allocated memory in decimal GB,
not total process VRAM. Medians include all five runs. Compiled baseline run 2
was slower (5.22 s versus 4.20–4.26 s for the others); it was retained.
Compiled MTP wall time ranged from 2.636 to 2.648 s.

**Compare MTP against the baseline with the same compilation setting.**
Compilation alone brings baseline decode throughput from 23.3 to 63.6 tokens/s.
Adding MTP to the compiled baseline gives about **1.81× reported decode
throughput** and **1.61× end-to-end throughput** on this fixture. In eager mode,
the corresponding gains are 2.97× and 2.89×. The eager comparison alone does
not describe the benefit over compiled ordinary decoding.

MTP accepted 186/204 drafts (91.2%) in every eager run and 184/211 (87.2%) in
every compiled run. Text was repeatable within each configuration, but MTP and
baseline outputs differed in both modes. These fixed-length timings do not
establish exact greedy parity or complete/correct generated code. Use longer
output limits for coding tasks and validate the resulting files.

The [recorded measurements](benchmark-results.json) preserve all five timings,
decode rates, output hashes, backend diagnostics, and comparison limits. This
is one short prompt, not a general ranking across workloads or engines.

34 CPU fallback tests exercise native MTP and greedy verification; those tests
alone do not certify GPU behavior. The live API run confirmed text generation
and active MTP, and exposed XML numeric tool arguments being returned as
strings. That server conversion bug has a schema-aware fix and regression
tests. The live text, numeric tool round-trip, text SSE, and tool SSE checks
passed after the fix. Claude Code 2.1.79 passed Read, Write, and Bash checks,
including a generated three-test suite. Its configured output budget was
2048 tokens. See the
[implementation review](https://github.com/eole-nlp/eole/blob/main/docs/mtp-inference-review.md)
for rounding and backend limitations.

Quantization module selection uses the existing `quant_layers` and
`quant_exclude_modules` settings. Bare include names such as `down_proj` select
that leaf name anywhere in the model. Dotted entries select exact paths from the
model root, and shell-style globs such as
`decoder.transformer_layers.*.mlp.down_proj` select decoder projections only.
`*` can span path components. Exclusions take precedence; excluding a parent
skips its entire subtree. Bare exclusions retain their legacy behavior of
matching parent names anywhere in the model.

For compressed-tensors conversion, `quant_layers` is populated with exact Eole
paths from the packed weight inventory, preserving floating-point decoder and
MTP modules. Loading rejects selections that conflict with stored packed weights.
All quantization backends share these matching rules. Bits and group size remain
global settings; per-module quantization parameters are not supported.
