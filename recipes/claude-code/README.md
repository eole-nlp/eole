# Use Claude Code with local Qwen3.8 served by Eole

This experimental recipe connects the **Claude Code CLI** to Eole's
Anthropic-style Messages endpoint. The answering model is Qwen3.8-27B.
Anthropic does not support routing Claude Code to non-Claude models through a
gateway; this is an Eole compatibility workflow, not an official Anthropic
integration. See the [gateway documentation](https://code.claude.com/docs/en/llm-gateway).

Run Eole commands from the repository root. First complete the
[Qwen3.8 checkpoint and MTP setup](../qwen38/README.md). Use an INT4 checkpoint
retaining MTP weights for a 32 GB RTX 5090. The shared YAML fixes greedy decoding
with `top_k: 1`, single-sequence requests, and a conservative 32K context.

## Start and verify Eole

```bash
export EOLE_MODEL_DIR=/path/to/models
export QWEN38_MODEL="$EOLE_MODEL_DIR/Qwen3.8-27B-INT4"
python recipes/qwen38/check_checkpoint.py "$QWEN38_MODEL"
eole serve -c recipes/qwen38/serve.yaml --host 127.0.0.1 --port 5000
```

In another terminal run the live API checks:

```bash
python recipes/claude-code/smoke_test.py --base-url http://127.0.0.1:5000 \
  --model qwen3.8-27B
```

The script checks health, model discovery, token counting, a plain Messages
response, a tool-use/tool-result round trip, and typed text/tool-use SSE events. It executes
only a synthetic `add` tool locally; it does not run model-proposed shell
commands. A successful HTTP response alone is not a passing tool-use test.
The script fails on error responses, missing tool blocks, or an incorrect sum.

Verify that server logs contain `MTP draft acceptance`. A fallback warning means
MTP was not active. The script cannot prove speculation through API responses;
acceptance is a server-side diagnostic.

## Configure the Claude Code CLI

Install Claude Code using its [official instructions](https://code.claude.com/docs/en/overview).
Use a fresh terminal for the following variables so they do not change your
normal Claude setup:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:5000
export ANTHROPIC_AUTH_TOKEN=eole-local
export ANTHROPIC_MODEL=qwen3.8-27B
export ANTHROPIC_DEFAULT_OPUS_MODEL=qwen3.8-27B
export ANTHROPIC_DEFAULT_SONNET_MODEL=qwen3.8-27B
export ANTHROPIC_DEFAULT_HAIKU_MODEL=qwen3.8-27B
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=32768
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=2048
claude --model qwen3.8-27B
```

The placeholder token selects the client credential path; **Eole does not
validate it or add authentication**. Keep the server bound to loopback.
The default-model mappings keep client model aliases pointed at the same Eole
model. Existing Claude settings or provider variables can override this setup;
check `/status` for the actual URL and model before continuing.
The explicit context setting prevents Claude Code from assuming that the custom
model has a 200K window. The 2048-token output budget leaves room for the client
system prompt and tool history within the server’s 32K context; the client’s
default budget for a custom model may otherwise be 32K. Increase server context
and client output budget together only after checking VRAM. See the
[environment variable reference](https://code.claude.com/docs/en/env-vars).
See [gateway connection](https://code.claude.com/docs/en/llm-gateway-connect) and
[model configuration](https://code.claude.com/docs/en/model-config).

For a fully isolated client test, set `CLAUDE_CONFIG_DIR` to an empty writable
scratch directory before launching Claude. This also avoids read-only home
errors from the Bash tool's session files. The validation harness does this
automatically and does not modify your normal Claude configuration.

## Try an isolated coding task

Start a session in a scratch directory, not your real project:

```bash
mkdir -p /tmp/eole-claude-demo
cd /tmp/eole-claude-demo
claude --model qwen3.8-27B
```

Ask:

> Create palindrome.py with an is_palindrome function that ignores case and
> non-alphanumeric characters. Add a small unittest file, run it, and report
> the result.

Review and approve file/shell operations through the normal client permission
flow. Check the files and test output; a convincing final message is not proof
that tools ran. This deliberately small exercise tests repeated tool calls,
results returning to the model, and an actual edit/test cycle.

## Behavior and troubleshooting

- **Response appears all at once:** Anthropic SSE currently buffers the complete
  model output before emitting typed blocks. This is expected; it is not live
  token streaming. MTP can reduce generation time but does not remove buffering.
- **MTP fallback:** Check batch size, greedy selection, loaded MTP tensors,
  text-only inputs, and decoding constraints. Client temperature does not undo
  `top_k: 1`; this recipe intentionally sacrifices sampling diversity.
- **Unknown model:** Use `qwen3.8-27B`, the YAML `id`, rather than the checkpoint
  directory or a Claude model name. Check `/v1/models` and `/status`.
- **Context or out-of-memory errors:** Reduce context/generation length. Claude
  Code adds system instructions and tool definitions beyond your visible prompt.
  A custom model name does not necessarily tell the client the real context
  window; keep sessions short and verify the client's context configuration.
- **Tool parsing fails:** Run the smoke test first; confirm the checkpoint's
  chat template has tool support. Model-generated arguments can be malformed.
- **Prompt caching:** Current Claude Code clients can send system-role cache
  markers within `messages`. Eole accepts their text as system instructions but
  does not implement Anthropic prompt-cache storage or accounting.
- **Unsupported features:** Extended thinking, prompt-cache accounting, beta
  request features, and structured output are not guaranteed. Unrecognized
  request fields may be ignored. Token usage is estimated.

## Validation status

On 2026-10-07 the live API checks passed on an RTX 5090 with the converted
Qwen3.8-27B INT4 checkpoint and native MTP: discovery, token counting, text,
numeric tool-use/tool-result round trip, typed text SSE, and typed tool SSE.
The run exposed an XML argument-type bug; the schema-aware server fix is covered
by regression tests and passed live revalidation.

Claude Code **2.1.79** passed a sentinel-file Read check, created the palindrome
implementation and unittest file through Write calls, and ran all three tests
through Bash after its configuration directory was redirected to writable
scratch storage. Independent test execution and additional cases also passed.
The read/write and shell checks used separate isolated client sessions with
explicit tool permissions. Pin and record the version used because protocol
behavior can change.

Client system prompts and tool history increase prefill time; the short-prompt
generation benchmark does not predict full coding-session latency. Typed SSE
is still buffered, as described above.
For measured MTP versus baseline results and output-parity limits see the
[benchmark recipe](../qwen38/README.md).
