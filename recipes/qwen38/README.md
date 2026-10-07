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

Install Eole in a CUDA-enabled environment. For this configuration install
FlashAttention and the hybrid-attention kernels used by Qwen:

```bash
pip install -e .
pip install flash-attn --no-build-isolation
pip install fla-core causal-conv1d
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

```bash
python recipes/qwen38/benchmark.py -c recipes/qwen38/predict.yaml \
  --output /tmp/qwen38-baseline.json
python recipes/qwen38/benchmark.py -c recipes/qwen38/predict-mtp.yaml \
  --output /tmp/qwen38-mtp.json
```

Each invocation loads one engine, excludes one warmup, and repeats the same
prompt three times. Compilation is disabled for the initial comparison.
Outputs include wall time, peak allocated VRAM, decoded text, and approximate
retokenized output counts. Use Eole's logs for internal timing and acceptance.
EOS can end a completion early; inspect output lengths before comparing.
Latency includes prefill and decoding. The harness fails if an MTP run reports
fallback or never logs draft acceptance.

To try compilation after establishing the eager baseline:

```bash
EOLE_TORCH_COMPILE=1 EOLE_COMPILE_MODE=0 \
  python recipes/qwen38/benchmark.py -c recipes/qwen38/predict-mtp.yaml \
  --output /tmp/qwen38-mtp-compile.json
```

Repeat the baseline with the same compile settings. Record GPU/driver,
PyTorch/kernel versions, checkpoint ID, precision, prompt/output lengths, and
draft acceptance with any shared result. Try `self_speculative_num_tokens: 1`,
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

The recipe's prompt fixture was run with the local converted
`Frozenlock/Qwen3.8-27B-int4-Autoround` checkpoint, INT4 weights / BF16 compute,
PyTorch 2.12.1+cu132, FlashAttention 2.8.3, and FLA gated-delta kernels. Compilation
was disabled. One warmup was excluded; each of three measured runs generated
256 tokens with batch size one.

| Path | Mean request wall time | Reported decode throughput | Peak allocated GPU memory |
|---|---|---|---|
| Ordinary greedy | 10.85 s | 23.6–24.5 tokens/s | 19.25 GB |
| MTP, three drafts | 3.89 s | 66.2–70.5 tokens/s | 19.65 GB |

MTP accepted 186/204 drafted tokens (91.2%). This is about **2.79× faster by
request wall time for this prompt**. However, all three MTP outputs differed
from the corresponding baseline text. At the 256-token cap, extra comments in
the MTP answer also left the code incomplete. This is a throughput measurement,
not evidence of exact greedy parity or complete/correct generated code. Use
longer output limits for coding tasks and validate the resulting files.

The [recorded measurements](benchmark-results.json) preserve the fixture hash,
per-run timings, backend versions, and comparison limits. Do not generalize
this single-prompt result to other workloads, draft counts, or checkpoints.

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
