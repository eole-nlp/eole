# Serve a model with Eole

Run all commands from the repository root. The example serves
`Qwen/Qwen3.5-0.8B` on one NVIDIA GPU using BF16 and PyTorch attention.
Install Eole with `pip install -e .` in a CUDA-enabled PyTorch environment.

## Download and convert

```bash
export EOLE_MODEL_DIR="$PWD/models"
mkdir -p "$EOLE_MODEL_DIR"
eole convert HF --model_dir Qwen/Qwen3.5-0.8B \
  --output "$EOLE_MODEL_DIR/qwen3.5-0.8B"
eole serve -c recipes/server/serve.example.yaml --host 127.0.0.1 --port 5000
```

`preload: true` loads the model at startup. `path` can point to a local converted
checkpoint or a Hugging Face repository containing an Eole checkpoint; the
latter is downloaded beneath `models_root`. A raw HF checkpoint is a different
case: consult the supported-HF inference path or convert it explicitly first.
Change both `id` and `path` in the YAML to serve another checkpoint.

## Check the server

```bash
curl --fail-with-body http://127.0.0.1:5000/health
curl --fail-with-body http://127.0.0.1:5000/models
```

Open `http://127.0.0.1:5000/docs` for interactive API documentation.

## OpenAI-style chat

```bash
curl --fail-with-body http://127.0.0.1:5000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5-0.8B","messages":[{"role":"user","content":"Say hello in French."}],"max_tokens":128,"temperature":0}'
```

Read `choices[0].message.content`. Add `"stream":true` and use `curl -N` to
receive server-sent events. This endpoint also accepts tool definitions; actual
model tool-use quality depends on the checkpoint and its chat template.

## Anthropic-style Messages

```bash
curl --fail-with-body http://127.0.0.1:5000/v1/messages \
  -H 'Content-Type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"qwen3.5-0.8B","messages":[{"role":"user","content":"Say hello in French."}],"max_tokens":128,"temperature":0}'
```

Read the `content` blocks. The server also exposes `/v1/messages/count_tokens`
and Anthropic-format `/v1/models` discovery. Token usage is estimated; this is
an API compatibility layer, not full Anthropic feature parity. The Anthropic
SSE implementation currently buffers generation before emitting typed content
blocks, so `stream:true` does not provide live token-by-token display.

For a coding client and a larger checkpoint, follow the
[Claude Code recipe](../claude-code/README.md). For speculative decoding and
performance comparisons, follow [Qwen3.8 / MTP](../qwen38/README.md).

## Responses API for Codex

`POST /v1/responses` supports text and client-executed function/custom tools,
including SSE events and replayed tool results. It is stateless: use
`store:false` and send the conversation history in `input`. Text streams during
generation; individual tool blocks are parsed before being emitted. Hosted
tools and server-side continuation/compaction are unsupported.

Follow the [Codex recipe](../codex/README.md) for model metadata, CLI/Linux app
configuration, and live protocol checks.

## Native inference

`/infer` accepts already formatted model inputs rather than a chat history:

```bash
curl --fail-with-body http://127.0.0.1:5000/infer \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5-0.8B","inputs":["<|im_start|>user\nSay hello.<|im_end|>\n<|im_start|>assistant\n"],"max_length":128,"beam_size":1,"top_k":1}'
```

Use the checkpoint's chat template for real applications.

## Decoding and deployment notes

The example fixes `top_k: 1` for greedy output. Request temperature alone does
not enable sampling under that policy. To experiment with sampling, change the
model configuration deliberately; for MTP keep the recipe's greedy settings.
Batch size is one for this example. MTP additionally requires one sequence,
one hypothesis, text-only input, and no unsupported decoding constraints.

The server has no application-level authentication. Bind to loopback for local
use, as above. Use your own authenticated reverse proxy before exposing it to
other machines. Port publishing and a placeholder client token do not add
server authentication.
