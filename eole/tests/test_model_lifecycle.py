"""Model residency policy and streaming leases without GPU dependencies."""

import asyncio
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path
import yaml

from eole.server.model import Server, Model
from eole.server.lifecycle import ModelLifecycleMiddleware
from eole.server.ui_runtime import Workspace


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    def server(self, policy="single"):
        server = Server()
        server.model_policy = policy
        events = []

        def model(name, loaded):
            m = SimpleNamespace(loaded=loaded, processing_task=None)

            def load():
                events.append("load:" + name)
                m.loaded = True

            def unload():
                events.append("unload:" + name)
                m.loaded = False

            m.load, m.unload = load, unload
            return m

        server.models = {"a": model("a", True), "b": model("b", False)}
        return server, events

    async def test_switch_and_reuse(self):
        server, events = self.server()
        workspace = Workspace(server)
        self.assertIs(workspace.lock, server.model_lock)
        async with workspace.lock:
            await workspace.prepare("b", server.models["b"], {})
            await workspace.prepare("b", server.models["b"], {})
        self.assertEqual(events, ["unload:a", "load:b"])
        self.assertFalse(server.models["a"].loaded)
        self.assertTrue(server.models["b"].loaded)
        self.assertIsNone(server.model_activity)

    async def test_omitted_policy_preserves_existing_api_and_workspace_behavior(self):
        server, events = self.server(None)
        await server.prepare_model("b", {})
        self.assertEqual(events, ["load:b"])
        await server.maybe_load_model("a")
        self.assertEqual(events, ["load:b", "unload:b"])

    async def test_keep_loaded(self):
        server, events = self.server("keep_loaded")
        await server.prepare_model("b", {})
        self.assertEqual(events, ["load:b"])
        self.assertTrue(server.models["a"].loaded)

    async def test_streaming_lease_blocks_switch_but_not_state(self):
        server, events = self.server()
        entered, release = asyncio.Event(), asyncio.Event()

        async def app(scope, receive, send):
            if scope["path"] == "/v1/responses":
                entered.set()
                await release.wait()
            events.append(scope["path"])

        middleware = ModelLifecycleMiddleware(app, server)
        first = asyncio.create_task(middleware(dict(type="http", path="/v1/responses"), None, None))
        await entered.wait()

        async def switch():
            async with Workspace(server).lock:
                await server.prepare_model("b", {})

        second = asyncio.create_task(switch())
        await middleware(dict(type="http", path="/ui/state"), None, None)
        await asyncio.sleep(0)
        self.assertEqual(events, ["/ui/state"])
        release.set()
        await asyncio.gather(first, second)
        self.assertEqual(events, ["/ui/state", "/v1/responses", "unload:a", "load:b"])

    async def test_api_aliases_share_the_model_lease(self):
        server, _ = self.server()
        observed = []

        async def app(scope, receive, send):
            observed.append(server.model_lock.locked())

        middleware = ModelLifecycleMiddleware(app, server)
        for path in (
            "/infer",
            "/unload_model",
            "/v1/chat/completions",
            "/openai/chat/completions",
            "/v1/messages",
            "/anthropic/v1/messages",
            "/anthropic/v1/messages/count_tokens",
        ):
            await middleware(dict(type="http", path=path), None, None)
        self.assertEqual(observed, [True] * 7)
        await middleware(dict(type="http", path="/ui/cancel/job"), None, None)
        self.assertFalse(observed[-1])

    async def test_disconnect_drains_model_load(self):
        server, _ = self.server("keep_loaded")
        started, release = threading.Event(), threading.Event()

        def load():
            started.set()
            release.wait(5)
            server.models["b"].loaded = True

        server.models["b"].load = load
        task = asyncio.create_task(server.prepare_model("b", {}))
        await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(server.models["b"].loaded)
        self.assertIsNone(server.model_activity)

    async def test_failure_can_be_retried(self):
        server, _ = self.server()
        original = server.models["b"].load
        server.models["b"].load = Mock(side_effect=RuntimeError("load failed"))
        with self.assertRaisesRegex(RuntimeError, "load failed"):
            await server.prepare_model("b", {})
        self.assertIsNone(server.model_activity)
        server.models["b"].load = original
        await server.prepare_model("b", {})
        self.assertTrue(server.models["b"].loaded)


class ConfigTests(unittest.TestCase):
    def test_invalid_config_fails_before_preloading(self):
        for extra in [
            dict(model_policy="bad"),
            dict(model_policy="single"),
            dict(workspace=dict(models=dict(ocr="missing"))),
        ]:
            config = dict(
                models_root="/models",
                models=[dict(id="a", path="a", preload=True), dict(id="b", path="b", preload=True)],
            )
            config.update(extra)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "server.yaml"
                path.write_text(yaml.safe_dump(config))
                with patch("eole.server.model.Model") as model:
                    with self.assertRaises(ValueError):
                        Server().start(path)
                    model.assert_not_called()

    def test_unload_terminates_engine_and_releases_reference(self):
        model = Model()
        engine = Mock()
        model.engine, model.loaded = engine, True
        with patch("eole.server.model.torch.cuda.empty_cache"):
            model.unload()
        engine.terminate.assert_called_once()
        self.assertIsNone(model.engine)
        self.assertFalse(model.loaded)


class WorkspaceSelectionTests(unittest.TestCase):
    def test_browser_model_defaults(self):
        import shutil
        import subprocess

        node = shutil.which("node")
        if not node:
            self.skipTest("Node is only required for browser helper checks")
        subprocess.run(
            [node, str(Path(__file__).with_name("web") / "test_workspace_models.cjs")], check=True, timeout=20
        )


class ModelDiscoveryTests(unittest.TestCase):
    def test_lazy_model_discovery_does_not_change_residency(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from unittest.mock import AsyncMock
        from eole.server.anthropic import register_anthropic

        model = SimpleNamespace(
            loaded=False, pre_config=dict(max_length=4096, context_length=16384), get_model_limits=lambda: (0, 0)
        )
        server = SimpleNamespace(models={"lazy": model}, model_policy="single", maybe_load_model=AsyncMock())
        app = FastAPI()
        register_anthropic(app, server)
        client = TestClient(app)
        self.assertEqual(client.get("/v1/models").json()["data"][0]["id"], "lazy")
        result = client.get("/anthropic/v1/models/lazy").json()
        self.assertEqual(result["max_input_tokens"], 12288)
        self.assertEqual(result["max_tokens"], 4096)
        server.maybe_load_model.assert_not_called()
