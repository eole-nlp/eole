"""The persistent scheduler must exit before the application process does."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import unittest

from fastapi.testclient import TestClient
from eole.server.app import create_app
from eole.server.model import Server


class ContinuousLifespanTests(unittest.TestCase):
    def test_shutdown_unloads_continuous_model(self):
        server = Server()
        server.models = {"test": SimpleNamespace(engine=SimpleNamespace(continuous=object()))}
        server.unload_model = AsyncMock()
        with patch("eole.server.app.Server", return_value=server), patch.object(server, "start"):
            app = create_app("unused")
            with TestClient(app) as client:
                self.assertEqual(client.get("/health").status_code, 200)
        server.unload_model.assert_awaited_once_with("test")

    def test_other_generation_routes_are_rejected_before_headers_or_loading(self):
        server = Server()
        server.models = {"test": SimpleNamespace(continuous_config={}, engine=None)}
        server.maybe_load_model = AsyncMock(side_effect=AssertionError("must reject before loading"))
        with patch("eole.server.app.Server", return_value=server), patch.object(server, "start"):
            app = create_app("unused")
            with TestClient(app) as client:
                for path in (
                    "/infer",
                    "/v1/chat/completions",
                    "/openai/chat/completions",
                    "/v1/messages",
                    "/anthropic/v1/messages",
                    "/ui/chat",
                    "/ui/translate",
                ):
                    for stream in (False, True):
                        response = client.post(path, json={"model": "test", "stream": stream})
                        self.assertEqual(response.status_code, 409, path)
                        self.assertEqual(response.json()["error"]["code"], "continuous_responses_only")
                self.assertEqual(client.get("/health").status_code, 200)
        server.maybe_load_model.assert_not_awaited()
