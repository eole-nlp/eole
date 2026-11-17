"""Cancellation must not consume a batch slot or fail a neighbouring request."""

import asyncio
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock

from eole.server.model import Model, QueuedRequest


class BatchCancellationTests(unittest.IsolatedAsyncioTestCase):
    def request(self, text):
        return QueuedRequest(text, {}, False, asyncio.get_running_loop().create_future(), 0)

    async def test_cancelled_queued_request_is_not_dispatched(self):
        model = Model(model_id="test")
        cancelled, live = self.request("cancelled"), self.request("live")
        cancelled.future.cancel()
        calls = []
        with ThreadPoolExecutor(1) as pool:

            def infer(inputs, settings):
                calls.append(inputs)
                return [[0]] * len(inputs), None, [[x] for x in inputs]

            model.engine = SimpleNamespace(_thread_pool=pool, infer_list=infer)
            await model._process_batch([cancelled, live])
        self.assertEqual(calls, [["live"]])
        self.assertEqual(await live.future, ([[0]], [["live"]]))

    async def test_cancel_during_forward_does_not_fail_neighbour(self):
        model = Model(model_id="test")
        cancelled, live = self.request("cancelled"), self.request("live")
        entered, release = threading.Event(), threading.Event()
        with ThreadPoolExecutor(1) as pool:

            def infer(inputs, settings):
                entered.set()
                if not release.wait(3):
                    raise RuntimeError("test forward blocked")
                return [[0]] * len(inputs), None, [[x] for x in inputs]

            model.engine = SimpleNamespace(_thread_pool=pool, infer_list=infer)
            task = asyncio.create_task(model._process_batch([cancelled, live]))
            try:
                self.assertTrue(await asyncio.wait_for(asyncio.to_thread(entered.wait, 2), 3))
                cancelled.future.cancel()
            finally:
                release.set()
                await task
        self.assertEqual(await live.future, ([[0]], [["live"]]))
        self.assertTrue(cancelled.future.cancelled())

    async def test_queued_settings_are_copied_and_none_is_independent(self):
        model = Model(model_id="test")
        model.loaded = True
        model.ensure_batch_processor = AsyncMock()
        settings = {"max_length": 7}
        task = asyncio.create_task(model.infer_async("first", settings=settings))
        first = await asyncio.wait_for(model.request_queue.get(), 2)
        settings["max_length"] = 99
        self.assertEqual(first.settings, {"max_length": 7})
        first.future.set_result("done")
        self.assertEqual(await task, "done")
        task = asyncio.create_task(model.infer_async("second"))
        second = await asyncio.wait_for(model.request_queue.get(), 2)
        self.assertEqual(second.settings, {})
        self.assertIsNot(first.settings, second.settings)
        second.future.set_result("done")
        await task
