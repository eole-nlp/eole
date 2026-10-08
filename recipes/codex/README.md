# Use Codex with local Qwen served by Eole

This experimental recipe targets the Codex CLI and Linux desktop app through
Eole's stateless `/v1/responses` endpoint. The answering model is local Qwen,
and Codex executes tools under its normal permissions. A working HTTP response
alone does not establish that Qwen can complete a coding task.

## Start Eole and check the protocol

From the repository root, complete the [Qwen3.8 conversion setup](../qwen38/README.md).
The recommended checkpoint is converted `RedHatAI/Qwen3.8-27B-INT4`.

```bash
export EOLE_MODEL_DIR=/mnt/InternalCrucial4/LLM_work
export QWEN38_MODEL="$EOLE_MODEL_DIR/Qwen3.8-27B-INT4"
export EOLE_TORCH_COMPILE=1
export EOLE_COMPILE_MODE=0
eole serve -c recipes/codex/serve.yaml --host 127.0.0.1 --port 5000
```

This YAML fixes single-sequence greedy generation, a 32K context, a 2048-token
output budget, and disables MTP for the initial integration test. Establish
correctness before comparing MTP throughput.

In another terminal:

```bash
python recipes/codex/smoke_test.py --base-url http://127.0.0.1:5000/v1 \
  --model qwen3.8-27B
```

The smoke test checks JSON text, typed SSE text, typed function arguments, and
an add-tool/result round trip. It executes only a synthetic integer addition;
it does not run generated shell commands.

## Create an isolated Codex configuration

```bash
python recipes/codex/configure.py --output-dir /tmp/eole-codex-home
mkdir -p /tmp/eole-codex-demo
printf 'EOLE_SENTINEL_42\n' > /tmp/eole-codex-demo/sentinel.txt
```

The helper creates `config.toml` and a matching Qwen model catalog. It refuses
to overwrite existing configuration. Context metadata must match the server;
pass `--context-window` if you change the YAML. The provider uses HTTP Responses
with `supports_websockets=false`, no OpenAI authentication, and disabled hosted
web search. The model name must match the server YAML ID exactly.

For the CLI, install Codex using its [official instructions](https://learn.chatgpt.com/docs/codex/cli).
Run it with the isolated home:

```bash
CODEX_HOME=/tmp/eole-codex-home codex -C /tmp/eole-codex-demo
```

If your Linux desktop package bundles the CLI without installing a `codex`
command, use that package's bundled binary instead. For example, the tested
package places it at `/usr/lib/chatgpt/resources/codex`.

## Linux desktop app

Quit the running desktop app before launching it with a different `CODEX_HOME`;
a second invocation can otherwise reuse the existing process and configuration.
Launch your installed app executable from a terminal:

```bash
CODEX_HOME=/tmp/eole-codex-home <your-codex-app-executable>
```

For the Linux package whose desktop launcher is `chatgpt`, that is:

```bash
CODEX_HOME=/tmp/eole-codex-home chatgpt
```

Select the local **qwen3.8-27B (Eole)** model in a new local Codex chat and open
`/tmp/eole-codex-demo` as the project. The custom provider/catalog configuration
is described in the [official gateway guide](https://learn.chatgpt.com/docs/enterprise/connect-to-a-gateway).
Do not select a cloud work environment for this localhost provider. If the model
is absent, check the app's active host and configuration, then restart it.
An existing chat can retain its previous model selection.

First ask it to read `sentinel.txt` with a tool and report its exact contents.
Then ask it to create a palindrome function and a unittest file, run the tests,
and report their output. Review the tool calls through the normal permission
flow and independently inspect the files and test results. To return to your
normal configuration, quit the app and launch it without the `CODEX_HOME` override.

## Supported behavior and limits

- Text SSE deltas are emitted during generation. Each tool block is buffered
  until its complete arguments can be parsed, then emitted as a typed call.
- Replayed user, developer, assistant, function-call and function-result items
  retain their order and call IDs. Use `store=false`; server-side history and
  `previous_response_id` are unsupported.
- Function tools, namespaced tools, and custom raw-input tools are translated
  for Qwen's chat template. Custom tools such as `apply_patch` are represented
  to Qwen as a function with one `input` string, then translated back to a
  `custom_tool_call`. The backend does not enforce a custom grammar. Codex
  still validates/executes its tool input; malformed patches can fail.
- Hosted web/file search, images, JSON-schema output, background responses,
  encrypted reasoning replay, and server-side compaction are unsupported.
  Start a fresh chat before reaching the context limit. Requests for these
  operations return an error instead of silently enabling them.
- Server generation is capped by the YAML. A budget stop emits
  `response.incomplete`; engine/parsing errors emit `response.failed`.
  Generated-token usage counts include EOS; input token counts use the server
  tokenizer or its existing estimate fallback. Cache and reasoning breakdowns
  are unavailable and reported as zero.
- The server disables Qwen thinking through its existing chat-template setting;
  Codex reasoning-effort settings do not change that policy.
- This endpoint adds no authentication. Keep the server bound to loopback.
  Advanced desktop features and third-party plugins have not been validated
  against this model. Use short, isolated coding tasks first.

## Validation status

CPU protocol regression tests cover streaming, function/custom/namespaced
calls, tool-result replay, tag boundaries, EOS handling, input limits, and
error/incomplete responses. Codex CLI **0.162.0-alpha.2**, bundled with the Linux
app, accepted the generated model catalog and completed a Responses tool-call
round trip against a deterministic fake inference engine. Its shell tool hit
a sandbox-directory error in the development environment, so that check does
not establish successful shell execution or Qwen coding quality.

The installed Linux app code reads `CODEX_HOME`, and its app-server
`initialize`/`model/list` checks exposed the generated local Qwen catalog entry.
The desktop UI and live GPU
Qwen workflow still require validation on the host. Record your app/CLI version
and smoke-test results before treating this as a validated integration.
