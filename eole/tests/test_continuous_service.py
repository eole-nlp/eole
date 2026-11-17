"""Exercise worker ownership, shared progress, cancellation and shutdown on CPU."""

from concurrent.futures import ThreadPoolExecutor
from queue import SimpleQueue
from threading import Event
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import torch

from eole.server.continuous import ContinuousService


class Streamer:
    created = Event()
    count = 0

    def __init__(self, *args, cancel_event=None, **kwargs):
        self.cancel_event = cancel_event if cancel_event is not None else Event()
        self.token_count = 0
        self.first_token_time = self.last_token_time = None
        self.queue = SimpleQueue()

    @property
    def cancelled(self):
        return self.cancel_event.is_set()

    def put(self, tokens):
        if not self.cancelled:
            self.token_count += 1
            self.last_token_time = perf_counter()
            self.first_token_time = self.first_token_time or self.last_token_time
            self.queue.put(tokens[0])

    def end(self):
        self.queue.put(None)

    def __iter__(self):
        type(self).count += 1
        if type(self).count == 2:
            type(self).created.set()
        while True:
            value = self.queue.get(timeout=3)
            if value is None:
                return
            yield value


class Runner:
    entered = Event()
    release = Event()
    maximum = 0
    fail = False

    def __init__(self, *args, **kwargs):
        self.jobs = {}

    @property
    def has_work(self):
        return bool(self.jobs)

    def submit(self, identifier, ids, budget):
        self.jobs[identifier] = 0

    def cancel(self, identifier):
        self.jobs.pop(identifier, None)

    def forget(self, identifier):
        pass

    def tick(self):
        type(self).entered.set()
        if not type(self).release.wait(3):
            raise RuntimeError("test did not release worker")
        if type(self).fail:
            raise RuntimeError("forward failed")
        type(self).maximum = max(type(self).maximum, len(self.jobs))
        events = []
        for identifier, count in list(self.jobs.items()):
            count += 1
            self.jobs[identifier] = count
            status = "running" if count == 1 else "length"
            events.append((identifier, count, status))
            if status != "running":
                self.jobs.pop(identifier)
        return events

    def close(self):
        self.jobs.clear()


class ContinuousServiceTests(unittest.TestCase):
    def setUp(self):
        Runner.entered, Runner.release, Runner.maximum, Runner.fail = Event(), Event(), 0, False
        Streamer.created, Streamer.count = Event(), 0
        self.pool = ThreadPoolExecutor(1)
        engine = SimpleNamespace(
            _thread_pool=self.pool,
            predictor=SimpleNamespace(_tgt_eos_idx=[9], max_length=2),
            vocabs=None,
            transform_pipe=None,
            _build_inference_iterator=lambda src: iter([({"src": torch.tensor([[1, 2]])}, 0)]),
        )
        self.service = ContinuousService(engine, runner_factory=Runner)
        self.patch = patch("eole.server.continuous.GenerationStreamer", Streamer)
        self.patch.start()
        self.service.start()

    def tearDown(self):
        Runner.release.set()
        self.service.close()
        self.pool.shutdown(wait=True)
        self.patch.stop()

    def test_second_request_joins_before_first_finishes(self):
        with ThreadPoolExecutor(2) as clients:
            first = clients.submit(lambda: list(self.service.stream("first", {})))
            try:
                self.assertTrue(Runner.entered.wait(2))
                second = clients.submit(lambda: list(self.service.stream("second", {})))
                self.assertTrue(Streamer.created.wait(2))
            finally:
                Runner.release.set()
            self.assertEqual(first.result(timeout=3), [1, 2])
            self.assertEqual(second.result(timeout=3), [1, 2])
        self.assertEqual(Runner.maximum, 2)

    def test_pre_cancelled_request_does_not_reach_runner(self):
        signal, stats = Event(), {}
        signal.set()
        self.assertEqual(list(self.service.stream("cancel", {}, stats, signal)), [])
        self.assertFalse(Runner.entered.is_set())
        self.assertTrue(stats["cancelled"])
        self.assertEqual(stats["output_tokens"], 0)

    def test_worker_failure_drains_stream_and_rejects_new_requests(self):
        Runner.fail = True
        Runner.release.set()
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            list(self.service.stream("failure", {}))
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            list(self.service.stream("next", {}))

    def test_shutdown_rejects_new_work_and_releases_admission(self):
        Runner.release.set()
        self.assertEqual(list(self.service.stream("complete", {})), [1, 2])
        # Every completed request returns its one admission ticket.
        permits = self.service.options["slots"] + self.service.options["max_pending"]
        for _ in range(permits):
            self.assertTrue(self.service.admission.acquire(blocking=False))
        self.assertFalse(self.service.admission.acquire(blocking=False))
        for _ in range(permits):
            self.service.admission.release()
        self.service.close()
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            list(self.service.stream("closed", {}))

    def test_idle_chunk_validation_fails_before_worker_start(self):
        for value in (0, 64, "512"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Idle prefill"):
                ContinuousService(None, prefill_chunk_size=128, prefill_idle_chunk_size=value)

    def test_non_greedy_or_mutable_settings_are_rejected(self):
        for settings in ({"temperature": 0.7}, {"top_k": 20}, {"presence_penalty": 1}, {"max_length": 0}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                list(self.service.stream("unsupported", settings))


class ContinuousResponsesTests(unittest.IsolatedAsyncioTestCase):
    async def test_enabled_responses_can_overlap_without_endpoint_lock(self):
        from fastapi import FastAPI
        import httpx
        from threading import Lock
        from eole.server.responses import register_responses
        from eole.tests.test_responses_api import FakeModel

        entered, lock, count = Event(), Lock(), [0]
        model = FakeModel()
        model.continuous = SimpleNamespace(validate=ContinuousService.validate)

        def generate(prompt, settings, generation_stats, cancel_event):
            with lock:
                count[0] += 1
                if count[0] == 2:
                    entered.set()
            if not entered.wait(2):
                raise RuntimeError("Responses requests were serialized")
            yield "done"
            generation_stats.update(output_tokens=1, stopped_on_eos=True)

        model.infer_list_stream = generate

        async def load(_):
            pass

        app = FastAPI()
        register_responses(app, SimpleNamespace(models={"test": model}, maybe_load_model=load))
        import asyncio

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            responses = await asyncio.gather(
                *[
                    client.post("/v1/responses", json={"model": "test", "input": "hello", "max_output_tokens": 16})
                    for _ in range(2)
                ]
            )
        self.assertEqual([r.status_code for r in responses], [200, 200])
        self.assertEqual([r.json()["status"] for r in responses], ["completed", "completed"])
