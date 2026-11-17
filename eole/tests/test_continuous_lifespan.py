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
