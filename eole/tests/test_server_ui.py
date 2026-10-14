"""Bundled assets and request-scoped UI protocol, without a GPU."""

import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from eole.server.ui import register_ui
from eole.predict.generator import GeneratorLM


class Engine:
    predictor = Mock(spec=GeneratorLM)
    config = SimpleNamespace(
        world_size=1, self_speculative_decoding=True, compute_dtype="bf16", context_length=128, max_length=32
    )

    def infer_list_stream(self, prompt, settings, generation_stats, cancel_event):
        self.settings = settings
        yield "Hello "
        yield "world"
        generation_stats.update(
            output_tokens=3,
            stopped_on_eos=True,
            decode_tokens_per_second=100.0,
            first_token_seconds=0.1,
            mtp_active=settings["self_speculative_decoding"],
            cancelled=cancel_event.is_set(),
        )


class Model:
    loaded = True
    pre_config = {}

    def __init__(self):
        self.engine = Engine()

    def apply_chat_template(self, messages, enable_thinking=False):
        return messages[-1]["content"]

    def count_tokens(self, prompt):
        return 10

    def get_model_limits(self):
        return 32, 96


class TestServerUI(unittest.TestCase):
    def setUp(self):
        self.model = Model()
        app = FastAPI()
        register_ui(app, SimpleNamespace(models={"test": self.model}))
        self.client = TestClient(app)
        self.payload = {"model": "test", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 16}

    def events(self, payload=None):
        response = self.client.post("/ui/chat", json=payload or self.payload)
        self.assertEqual(response.status_code, 200)
        return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]

    def test_assets_and_state(self):
        self.assertIn("A little space", self.client.get("/").text)
        self.assertEqual(self.client.get("/ui/assets/app.js").status_code, 200)
        self.assertEqual(self.client.get("/ui/assets/../../ui.py").status_code, 404)
        with patch("eole.server.ui.Hardware.read", return_value={"gpus": []}):
            self.assertEqual(self.client.get("/ui/state").json()["models"][0]["id"], "test")

    def test_exact_worker_stats_and_effective_presets(self):
        result = self.events()
        self.assertEqual(result[0]["type"], "start")
        self.assertEqual("".join(e["text"] for e in result if e["type"] == "delta"), "Hello world")
        self.assertEqual(result[-1]["stats"]["output_tokens"], 3)
        self.assertEqual(result[-1]["stats"]["decode_tokens_per_second"], 100.0)
        self.assertEqual(self.model.engine.settings["top_k"], 1)
        self.assertTrue(self.model.engine.settings["self_speculative_decoding"])
        self.events({**self.payload, "preset": "creative"})
        self.assertEqual(self.model.engine.settings["top_k"], 40)
        self.assertFalse(self.model.engine.settings["self_speculative_decoding"])

    def test_unknown_model_invalid_limits_and_context(self):
        self.assertEqual(self.client.post("/ui/chat", json={**self.payload, "model": "missing"}).status_code, 404)
        self.assertEqual(self.client.post("/ui/chat", json={**self.payload, "max_tokens": 0}).status_code, 422)
        self.assertEqual(self.events({**self.payload, "max_tokens": 33})[-1]["type"], "error")
        self.model.count_tokens = lambda prompt: 120
        self.assertEqual(self.events()[-1]["type"], "error")
        self.assertEqual(self.client.post("/ui/cancel/missing").status_code, 404)


class TestUICancellation(unittest.IsolatedAsyncioTestCase):
    async def test_stop_releases_worker_and_records_cancelled_request(self):
        import asyncio
        import threading
        import httpx

        started = threading.Event()
        finished = threading.Event()
        model = Model()

        def generate(prompt, settings, generation_stats, cancel_event):
            started.set()
            yield "first"
            if not cancel_event.wait(3):
                raise RuntimeError("Cancellation was not delivered")
            generation_stats.update(output_tokens=1, cancelled=True)
            finished.set()

        model.engine.infer_list_stream = generate
        app = FastAPI()
        register_ui(app, SimpleNamespace(models={"test": model}))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            with patch("eole.server.ui.Hardware.read", return_value={"gpus": []}):
                task = asyncio.create_task(
                    client.post(
                        "/ui/chat",
                        json={"model": "test", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 16},
                    )
                )
                for _ in range(200):
                    if started.is_set():
                        break
                    await asyncio.sleep(0.01)
                self.assertTrue(started.is_set())
                snapshot = (await client.get("/ui/state")).json()
                request_id = snapshot["active"][0]["id"]
                self.assertEqual((await client.post("/ui/cancel/" + request_id)).status_code, 200)
                result = await asyncio.wait_for(task, 3)
                self.assertIn('"cancelled": true', result.text)
                self.assertTrue(finished.is_set())
                snapshot = (await client.get("/ui/state")).json()
                self.assertEqual(snapshot["active"], [])
                self.assertEqual(snapshot["recent"][0]["status"], "cancelled")


class TestWorkerMetrics(unittest.TestCase):
    def test_worker_captures_metrics_before_next_request_and_restores_ui_settings(self):
        from concurrent.futures import ThreadPoolExecutor
        import threading
        from eole.inference_engine import InferenceEnginePY

        engine = InferenceEnginePY.__new__(InferenceEnginePY)
        engine.config = SimpleNamespace(world_size=1)
        engine.predictor = SimpleNamespace(top_k=1, _tgt_eos_idx=[99])
        engine.vocabs = {"tgt": SimpleNamespace(ids_to_tokens={1: "hello", 2: " world"})}
        engine.transform_pipe = None
        engine._build_inference_iterator = lambda src: None
        engine.logger = Mock()

        def predict(iterator, settings, streamer):
            engine.predictor.top_k = settings["top_k"]
            streamer.put([1])
            streamer.put([2])
            engine.predictor.decode_timings = [(1, 0.01)]
            engine.predictor.warmup_time = [0.03]
            engine.predictor._speculative_drafted_tokens = 5
            engine.predictor._speculative_accepted_tokens = 4
            streamer.end()

        engine._predict = predict
        stats = {}
        with ThreadPoolExecutor(max_workers=1) as pool:
            engine._thread_pool = pool
            result = list(
                engine.infer_list_stream(
                    "prompt", settings={"top_k": 40}, generation_stats=stats, cancel_event=threading.Event()
                )
            )
        self.assertEqual(result, ["hello", " world"])
        self.assertEqual(engine.predictor.top_k, 1)
        self.assertEqual(stats["output_tokens"], 2)
        self.assertEqual(stats["decode_tokens_per_second"], 100)
        self.assertEqual(stats["mtp_acceptance"], 0.8)
        self.assertFalse(stats["cancelled"])
        self.assertGreaterEqual(stats["first_token_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
