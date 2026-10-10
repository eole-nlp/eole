import json
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from eole.server.app import create_app


class _FakeModel:
    def __init__(self, response):
        self.loaded = True
        self.chunks = response if isinstance(response, list) else [response]
        self.response = "".join(self.chunks)
        self.render_calls = []
        self.infer_calls = []
        self.engine = SimpleNamespace(infer_list_stream=lambda inputs, settings: iter(self.chunks))

    def apply_chat_template(self, messages, tools=None, tool_choice=None, enable_thinking=False, reasoning_effort=None):
        self.render_calls.append((messages, tools, tool_choice, enable_thinking, reasoning_effort))
        return "rendered prompt<think>\n" if enable_thinking else "rendered prompt"

    async def infer_async(self, inputs, settings, is_chat):
        self.infer_calls.append((inputs, settings, is_chat))
        return [[0.0]], [[self.response]]


class _FakeServer:
    def __init__(self, model):
        self.models = {"qwen3.8-27B": model}

    def start(self, config_file):
        return None

    async def maybe_load_model(self, model_id):
        return None


def _request_payload(stream=False):
    return {
        "model": "qwen3.8-27B",
        "messages": [{"role": "user", "content": "List files."}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Run a command.",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
        "stream": stream,
    }


def _client_for(response):
    model = _FakeModel(response)
    server = _FakeServer(model)
    with patch("eole.server.app.Server", return_value=server):
        app = create_app("unused.yaml")
    return TestClient(app), model


def test_openai_endpoint_returns_structured_tool_call_and_forwards_tools():
    raw = "<tool_call><function=bash><parameter=command>ls</parameter></function></tool_call>"
    client, model = _client_for(raw)

    response = client.post("/v1/chat/completions", json=_request_payload())

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    call = choice["message"]["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"] == {"name": "bash", "arguments": '{"command":"ls"}'}
    assert model.render_calls[0][1] == _request_payload()["tools"]
    assert model.infer_calls[0][0] == "rendered prompt"
    assert model.infer_calls[0][2] is False


def test_openai_endpoint_preserves_tool_result_history():
    client, model = _client_for("Finished.")
    payload = _request_payload()
    payload["messages"] = [
        {"role": "user", "content": "List files."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command":"ls"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_123", "name": "bash", "content": "README.md"},
    ]

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Finished."
    rendered_messages = model.render_calls[0][0]
    assert rendered_messages[1]["tool_calls"][0]["function"]["arguments"] == {"command": "ls"}
    assert rendered_messages[2]["tool_call_id"] == "call_123"


def test_openai_endpoint_accepts_pi_developer_role():
    client, model = _client_for("Finished.")
    payload = _request_payload()
    payload["messages"] = [
        {"role": "developer", "content": "You are a coding assistant."},
        {"role": "user", "content": "Write a Flappy Bird game."},
    ]
    payload["reasoning_effort"] = "medium"

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert model.render_calls[0][0][0] == {"role": "developer", "content": "You are a coding assistant."}
    assert model.render_calls[0][3] is True
    assert model.render_calls[0][4] == "medium"


def test_openai_stream_returns_tool_call_delta():
    raw = "<tool_call><function=bash><parameter=command>ls</parameter></function></tool_call>"
    client, _ = _client_for(raw)

    response = client.post("/v1/chat/completions", json=_request_payload(stream=True))

    assert response.status_code == 200
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")]
    tool_deltas = [
        choice["delta"]["tool_calls"]
        for event in events
        for choice in event["choices"]
        if choice["delta"].get("tool_calls")
    ]
    assert len(tool_deltas) == 1
    assert tool_deltas[0][0]["function"] == {"name": "bash", "arguments": '{"command":"ls"}'}
    assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_openai_stream_returns_incremental_reasoning_and_text():
    client, model = _client_for(["Inspecting", " files</thi", "nk>", "Found it."])
    payload = _request_payload(stream=True)
    payload["enable_thinking"] = True

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")]
    deltas = [event["choices"][0]["delta"] for event in events]
    assert "".join(delta.get("reasoning_content") or "" for delta in deltas) == "Inspecting files"
    assert "".join(delta.get("content") or "" for delta in deltas) == "Found it."
    assert events[-1]["choices"][0]["finish_reason"] == "stop"
    assert model.render_calls[0][3] is True


def test_openai_stream_normalizes_newlines_in_tool_arguments_after_reasoning():
    raw = [
        "Inspecting files.</think>",
        "<tool_call>｟newline｠<function=bash>｟newline｠",
        "<parameter=command>｟newline｠pwd; ls -la｟newline｠</parameter>｟newline｠",
        "</function>｟newline｠</tool_call>",
    ]
    client, _ = _client_for(raw)
    payload = _request_payload(stream=True)
    payload["enable_thinking"] = True

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")]
    deltas = [event["choices"][0]["delta"] for event in events]
    assert "".join(delta.get("reasoning_content") or "" for delta in deltas) == "Inspecting files."
    tool_calls = [call for delta in deltas for call in (delta.get("tool_calls") or [])]
    assert tool_calls[0]["function"] == {"name": "bash", "arguments": '{"command":"pwd; ls -la"}'}


def test_openai_non_streaming_response_separates_reasoning():
    client, _ = _client_for("<think>Inspecting files</think>Found it.")

    response = client.post("/v1/chat/completions", json=_request_payload())

    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message["reasoning_content"] == "Inspecting files"
    assert message["content"] == "Found it."


def test_openai_non_streaming_response_uses_prompt_reasoning_state():
    client, _ = _client_for("Inspecting files</think>Found it.")
    payload = _request_payload()
    payload["enable_thinking"] = True

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message["reasoning_content"] == "Inspecting files"
    assert message["content"] == "Found it."


def test_openai_chat_template_kwargs_enable_thinking():
    client, model = _client_for("Done.")
    payload = _request_payload()
    payload["chat_template_kwargs"] = {"enable_thinking": True, "preserve_thinking": True}

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert model.render_calls[0][3] is True


def test_openai_tool_choice_none_does_not_execute_model_tool_markup():
    raw = "<tool_call><function=bash><parameter=command>ls</parameter></function></tool_call>"
    client, model = _client_for(raw)
    payload = _request_payload()
    payload["tool_choice"] = "none"

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["tool_calls"] is None
    assert choice["message"]["content"] == raw
    assert model.render_calls[0][1] is None


def test_openai_invalid_tool_choice_returns_400():
    client, _ = _client_for("unused")
    payload = _request_payload()
    payload["tool_choice"] = {"type": "function", "function": {"name": "missing"}}

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_tool_choice"


def test_openai_required_tool_choice_without_tools_returns_400():
    client, _ = _client_for("unused")
    payload = _request_payload()
    payload.pop("tools")
    payload["tool_choice"] = "required"

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_tool_choice"
