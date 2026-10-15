# Eole local workspace

`eole serve` now includes a browser interface at `/`. No Gradio, Node runtime,
frontend build, external fonts, or CDN is required. API routes and `/docs`
remain available. The original `apps/chatbot.py` stays a small Gradio example.

![Eole workspace](images/workspace.jpg)

[Watch the short UI tour](images/showcase.mp4). The recording shows real local
Qwen inference, hardware telemetry, and a completed benchmark, with navigation
pauses shortened. The animated preview is on the repository front page.

## Start with Qwen3.8 INT4

Use the converted **RedHatAI/Qwen3.8-27B-INT4** checkpoint, including MTP weights.
See [the Qwen3.8 recipe](../qwen38/README.md) for conversion.
Run from the repository root:

```bash
export EOLE_MODEL_DIR=/path/to/models
export QWEN38_MODEL="$EOLE_MODEL_DIR/Qwen3.8-27B-INT4"
CUDA_VISIBLE_DEVICES=0 EOLE_TORCH_COMPILE=1 EOLE_COMPILE_MODE=0 \
  eole serve -c recipes/ui/serve.yaml --host 127.0.0.1 --port 5000
```

Open **http://127.0.0.1:5000**. The supplied config uses BF16 compute,
Marlin projection packing, three MTP drafts, a 28,672-token context, and
chunked prefill. Context and output limits should be adjusted for your GPU;
this recipe is intended for an RTX 5090 with the converted INT4 model.
The chat output limit defaults to 4,096 tokens, capped by the selected model's limit.
It does not alter FP32 normalization.

The first compiled request can spend several minutes warming all full and
shortened verifier shapes. The UI shows preparation and reports warmup
separately. Subsequent requests reuse the compiled programs. Stop is cooperative:
it takes effect after the current forward or compilation finishes.

For optional GPU/CPU/RAM telemetry, install:

```bash
pip install nvidia-ml-py psutil
```

Alternatively install Eole with the `ui` extra. Missing drivers or optional
packages display unavailable readings; they do not prevent chat.

For a dedicated translation workflow, including Hy-MT2-7B, see the
[translation workspace recipe](../translation/README.md). Chat and Translate share
the same interface, monitoring and local server.

## Workspace features

- **Chat:** streaming responses, browser-local conversation history, edit/resend,
  regenerate, continuation after an output limit, basic Markdown/code rendering,
  copy/export, text-file attachments, separate thinking, and light/dark themes.
- **Generation settings:** Fast uses greedy selection and configured MTP when
  eligible. Creative uses sampling and turns MTP off. The response footer shows
  actual MTP activity and acceptance, not just the configured option.
- **Monitor:** cached hardware readings, active workspace requests, recent
  request timings, and a decode-speed chart. It tracks this UI's requests;
  external API requests are not listed. GPU cards are physical NVML devices.
- **Benchmark:** one discarded warmup followed by three 128-token requests.
  Reports the median post-first-token rate and exports settings and measurements.
  Early-EOS responses are excluded from the fixed-length median.
- **API & setup:** endpoint addresses, a curl example, and runtime information.

![Measured benchmark](images/benchmark.jpg)

## What the numbers mean

Output usage counts **actual generated model token IDs, including terminal EOS**,
not SSE events or whitespace. Decode speed comes from the inference worker's
synchronized timer from after its first output token to completion. It excludes
prefill, warmup, final detokenization and telemetry formatting, and excludes the
first token from its numerator. It includes decode dispatch and bookkeeping.

First-token latency includes server preparation and queue time. Prompt usage
uses the model tokenizer when available (the generic model counting fallback
is an estimate). MTP acceptance is the accepted/drafted token ratio. Cancelled
requests may not have a completed decode-time measurement; the UI displays a
missing rate rather than inventing one.

The benchmark uses one sequence, the selected mode/model/thinking setting, and
actual output counts. It is not whole-request throughput. Compare against other
engines using matching token histories, context sizes, output limits, stopping
policies, and timing definitions. Results vary with prompt and MTP acceptance.

## Scope and storage

Interactive streaming currently supports **single-process decoder-only models**.
Use Translate with a configured language pair for native encoder-decoder models,
or `/infer` for general inference. Use the inference recipes for tensor-parallel workloads.
Text attachments are inserted into the prompt, not indexed as a RAG collection.
The UI does not execute generated code or tools, and has no image-input flow.

Chats are saved in this browser's local storage; export them before clearing
browser data. No chat database is created on the server. The server retains at
most 50 UI request metric records in memory, without storing prompt content.
UI presets are restored after inference so they do not change later API defaults.

This is a local interface. The server has no application-level authentication;
keep it bound to loopback, or place it behind your authenticated proxy before
exposing it to other machines.

## Development

Assets live in `eole/server/web/` and are included as package data in wheels.
Edit them and reload the browser. No build step is needed. Formatting can be
checked with Prettier; backend routing, cancellation and metrics are covered by
`test_server_ui.py`, `test_server_streaming.py` and the speculative-decoding tests.
