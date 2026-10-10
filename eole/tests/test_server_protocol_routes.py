"""Protocol registration preserves public aliases and response contracts."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from eole.bin.run.serve import create_app


@pytest.fixture
def client():
    class FakeModel:
        loaded = True
        engine = SimpleNamespace(infer_list_stream=lambda prompt, settings: iter(["Hello"]))

        def get_model_limits(self):
            return 128, 1024

        def count_tokens(self, prompt):
            return 17

        def apply_chat_template(self, messages, **kwargs):
            return "rendered prompt"

        async def infer_async(self, inputs, settings, is_chat):
            return [[0.0]], [["Hello"]]

    async def load(model_id):
        pass

    server = SimpleNamespace(models={"qwen": FakeModel()}, start=lambda config: None, maybe_load_model=load)
    with patch("eole.bin.run.serve.Server", return_value=server):
        app = create_app("unused.yaml")
    with TestClient(app) as result:
        yield result


@pytest.mark.parametrize("path", ["/v1/messages", "/anthropic/v1/messages"])
@pytest.mark.parametrize("stream", [False, True])
def test_anthropic_message_routes(client, path, stream):
    response = client.post(
        path,
        json={"model": "qwen", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 32, "stream": stream},
    )
    assert response.status_code == 200
    if stream:
        assert "event: message_start" in response.text
        assert '"text": "Hello"' in response.text
        assert "event: message_stop" in response.text
    else:
        assert response.json()["content"] == [{"type": "text", "text": "Hello"}]
        assert response.json()["stop_reason"] == "end_turn"


@pytest.mark.parametrize("path", ["/v1/messages/count_tokens", "/anthropic/v1/messages/count_tokens"])
def test_anthropic_token_count_routes(client, path):
    response = client.post(
        path, json={"model": "qwen", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 32}
    )
    assert response.status_code == 200
    assert response.json() == {"input_tokens": 17}


@pytest.mark.parametrize("path", ["/v1/models", "/anthropic/v1/models", "/v1/models/qwen", "/anthropic/v1/models/qwen"])
def test_model_discovery_routes(client, path):
    response = client.get(path)
    assert response.status_code == 200
    body = response.json()
    model = body["data"][0] if "data" in body else body
    assert model["id"] == "qwen"
    assert (model["max_tokens"], model["max_input_tokens"]) == (128, 1024)


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/openai/chat/completions"])
@pytest.mark.parametrize("stream", [False, True])
def test_openai_chat_routes(client, path, stream):
    response = client.post(
        path, json={"model": "qwen", "messages": [{"role": "user", "content": "Hi"}], "stream": stream}
    )
    assert response.status_code == 200
    if stream:
        assert '"content":"Hello"' in response.text
        assert "data: [DONE]" in response.text
    else:
        assert response.json()["choices"][0]["message"]["content"] == "Hello"
        assert response.json()["choices"][0]["finish_reason"] == "stop"
