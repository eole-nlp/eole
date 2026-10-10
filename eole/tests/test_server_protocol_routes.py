"""Protocol registration preserves public aliases and response contracts."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from eole.bin.run.serve import create_app


def make_client():
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
    return TestClient(app)


class TestProtocolRoutes(unittest.TestCase):
    def setUp(self):
        self.client = make_client()
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_anthropic_message_routes(self):
        for path in ["/v1/messages", "/anthropic/v1/messages"]:
            for stream in [False, True]:
                with self.subTest(path=path, stream=stream):
                    response = self.client.post(
                        path,
                        json={
                            "model": "qwen",
                            "messages": [{"role": "user", "content": "Hi"}],
                            "max_tokens": 32,
                            "stream": stream,
                        },
                    )
                    self.assertTrue(response.status_code == 200)
                    if stream:
                        self.assertTrue("event: message_start" in response.text)
                        self.assertTrue('"text": "Hello"' in response.text)
                        self.assertTrue("event: message_stop" in response.text)
                    else:
                        self.assertTrue(response.json()["content"] == [{"type": "text", "text": "Hello"}])
                        self.assertTrue(response.json()["stop_reason"] == "end_turn")

    def test_anthropic_token_count_routes(self):
        for path in ["/v1/messages/count_tokens", "/anthropic/v1/messages/count_tokens"]:
            with self.subTest(path=path):
                response = self.client.post(
                    path, json={"model": "qwen", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 32}
                )
                self.assertTrue(response.status_code == 200)
                self.assertTrue(response.json() == {"input_tokens": 17})

    def test_model_discovery_routes(self):
        for path in ["/v1/models", "/anthropic/v1/models", "/v1/models/qwen", "/anthropic/v1/models/qwen"]:
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertTrue(response.status_code == 200)
                body = response.json()
                model = body["data"][0] if "data" in body else body
                self.assertTrue(model["id"] == "qwen")
                self.assertTrue((model["max_tokens"], model["max_input_tokens"]) == (128, 1024))

    def test_openai_chat_routes(self):
        for path in ["/v1/chat/completions", "/openai/chat/completions"]:
            for stream in [False, True]:
                with self.subTest(path=path, stream=stream):
                    response = self.client.post(
                        path, json={"model": "qwen", "messages": [{"role": "user", "content": "Hi"}], "stream": stream}
                    )
                    self.assertTrue(response.status_code == 200)
                    if stream:
                        self.assertTrue('"content":"Hello"' in response.text)
                        self.assertTrue("data: [DONE]" in response.text)
                    else:
                        self.assertTrue(response.json()["choices"][0]["message"]["content"] == "Hello")
                        self.assertTrue(response.json()["choices"][0]["finish_reason"] == "stop")
