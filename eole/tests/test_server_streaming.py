"""Shared streaming lifecycle checks without a model or GPU."""

import asyncio
import threading
import unittest
from types import SimpleNamespace

from eole.server.streaming import inference_stream


class TestInferenceStream(unittest.IsolatedAsyncioTestCase):
    async def test_chunks_and_generation_stats(self):
        stats = {}

        def generate(prompt, settings, generation_stats):
            self.assertEqual((prompt, settings), ("prompt", {"max_length": 3}))
            yield "a"
            yield "b"
            generation_stats.update(output_tokens=2, stopped_on_eos=True)

        async with inference_stream(
            SimpleNamespace(infer_list_stream=generate), "prompt", {"max_length": 3}, stats
        ) as chunks:
            self.assertEqual([chunk async for chunk in chunks], ["a", "b"])
        self.assertEqual(stats, {"output_tokens": 2, "stopped_on_eos": True})

    async def test_worker_error_is_propagated(self):
        def generate(prompt, settings):
            yield "first"
            raise RuntimeError("inference failed")

        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            async with inference_stream(SimpleNamespace(infer_list_stream=generate), "", {}) as chunks:
                async for _ in chunks:
                    pass

    async def check_early_exit(self, cancel):
        release, finished = threading.Event(), threading.Event()
        received = asyncio.Event()

        def generate(prompt, settings):
            try:
                yield "first"
                if not release.wait(5):
                    raise RuntimeError("test worker was not released")
                yield "remaining"
            finally:
                finished.set()

        async def consume():
            async with inference_stream(SimpleNamespace(infer_list_stream=generate), "", {}) as chunks:
                async for _ in chunks:
                    received.set()
                    if cancel:
                        await asyncio.Future()
                    break

        task = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(received.wait(), 2)
            if cancel:
                task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            self.assertFalse(finished.is_set())
        finally:
            release.set()
        if cancel:
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, 2)
        else:
            await asyncio.wait_for(task, 2)
        self.assertTrue(finished.is_set())

    async def test_early_exit_drains_worker(self):
        await self.check_early_exit(cancel=False)

    async def test_cancellation_drains_worker(self):
        await self.check_early_exit(cancel=True)
