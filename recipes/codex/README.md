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

This YAML fixes single-sequence greedy generation, a 28,672-token context, a 4096-token
output budget, splits prefill into 256-token chunks, and disables MTP for the
initial integration test. Chunking limits temporary prefill workspace for the
large instruction/tool prompts sent by the desktop app; it does not reduce
the configured context window. Establish
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
pass `--context-window` if you change the YAML. The helper sets both catalog
context fields and the top-level Codex context limit to the same value, with
`effective_context_window_percent=90`. Its defaults also include:

```toml
model_context_window = 28672
model_auto_compact_token_limit = 24576
```

The compaction threshold reserves 4096 tokens below the context limit; it is
adjusted by the same amount when `--context-window` changes. Keep these values
aligned with the server, and restart Codex in a new chat after changing them.
For an existing isolated home, edit `config.toml` at the top level (before any
`[section]`) and update both `context_window` and `max_context_window` in
`models.json`; the helper deliberately refuses to overwrite existing files.
These settings are described in the
[official configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference).
The provider uses HTTP Responses
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
flow and independently inspect the files and test results.

## Return to the default OpenAI configuration

The setup above keeps Qwen configuration in `/tmp/eole-codex-home` and leaves
`~/.codex/config.toml` and your normal sign-in untouched. Switching back means
starting Codex with your normal home directory and selecting an OpenAI model.
You do not need to delete the Qwen files or log out of your OpenAI account.

**CLI:** exit the Qwen session, then launch without the home override:

```bash
unset CODEX_HOME
env -u CODEX_HOME codex
```

Use the bundled CLI path instead of `codex` if that is how you normally launch
it. Do not pass `--oss`, `--local-provider`, a Qwen `--model`, or an Eole
`--config` override. In the new session, use `/status` to verify that the active
provider/model is OpenAI rather than Eole/Qwen. If needed, use `/model` to
select your usual OpenAI model. Your normal ChatGPT/API authentication is read
from the normal Codex home; sign in normally only if Codex asks.

**Linux desktop app:** fully quit the app, including any process left running
in the tray. Then launch it normally from the desktop menu, or run:

```bash
unset CODEX_HOME
env -u CODEX_HOME chatgpt
```

Replace `chatgpt` with your installed app executable if different. A second
launch while the Qwen app process is still running can reuse that process and
its configuration. Start a **new local chat** and select your usual **OpenAI
model** in the model picker; an existing Qwen chat can retain its model choice.
You should see the normal OpenAI model catalog instead of the isolated Qwen
catalog. If you previously used a different custom `CODEX_HOME` for your normal
OpenAI setup, restore that original value instead of unsetting it.

If you exported `CODEX_HOME=/tmp/eole-codex-home` in a shell startup file or
added it to a desktop launcher, remove that override there too, then restart
the terminal/app. The one-command environment assignments in this recipe do
not persist after the command exits.

**If you copied the Qwen settings into your normal config manually:** back up
`~/.codex/config.toml`, then restore its previous settings. Remove the Qwen
`model` value and the custom `model_catalog_json` path, and remove
`model_provider = "eole"` or replace it with `model_provider = "openai"`.
The unused `[model_providers.eole]` table can also be removed. Restore any
`web_search` setting you changed for the recipe. Keep your other settings and
authentication files. Quit/restart Codex and select an OpenAI model in a new
chat. See the [official configuration guide](https://learn.chatgpt.com/docs/config-file/config-basic)
for configuration locations and override precedence.

## Supported behavior and limits

- Text SSE deltas are emitted during generation. Each tool block is buffered
  until its complete arguments can be parsed, then emitted as a typed call.
- System/developer instructions are combined into one leading system message
  for Qwen. Replayed user, assistant, function-call and function-result items
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

## First-prompt troubleshooting

For `CUDA out of memory`, stop the desktop request/retries and restart the
server with **`recipes/codex/serve.yaml`**, not `recipes/qwen38/serve.yaml`.
The latter enables MTP and unchunked prefill. A short smoke-test prompt does
not exercise the desktop app's much larger instructions and tool definitions.
The server logs the rendered prompt token count and output budget for each
accepted Responses request. Chunked prefill reduces temporary workspace, but
KV caches and other persistent allocations still grow with the context;
long sessions may require a smaller context or fewer client tools.

The settings above were reported to fit the user's local Qwen INT4 desktop
setup. Memory requirements still depend on the hardware and runtime. The
client must reserve room for generation as history and tool results grow;
server-side compaction remains unsupported, so check whether client history
compaction succeeds and start a fresh chat if it fails. In the same test,
disabling plugins did not reduce the reported 53 tools. Do not assume plugin
disabling alone reduces the prompt: check the server's actual `tools` and
`input_tokens` counts.

A `Unsupported output format 'json_schema'` rejection is a separate protocol
limit. The endpoint supports text output only and does not silently discard a
schema constraint. If a later request gets `200 OK`, inspect its completion or
failure separately. Streaming HTTP `200` means the stream opened; a terminal
`response.failed` event still reports a failed inference. After any allocation
failure, failed-request decoder/MTP state is cleared before the next request.

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
