"""Shared streaming lifecycle checks without a model or GPU."""

import asyncio
import threading
import unittest
from types import SimpleNamespace

from eole.server.streaming import inference_stream


class TestInferenceStream(unittest.IsolatedAsyncioTestCase):
    async def test_completed_segment_does_not_cancel_a_multi_segment_job(self):
        signal = threading.Event()

        def generate(prompt, settings, cancel_event):
            self.assertFalse(cancel_event.is_set())
            yield prompt

        for prompt in ("first", "second"):
            async with inference_stream(
                SimpleNamespace(infer_list_stream=generate), prompt, {}, cancel_event=signal
            ) as chunks:
                self.assertEqual([chunk async for chunk in chunks], [prompt])
            self.assertFalse(signal.is_set())

    async def test_explicit_cancel_signal_stops_generation_on_early_exit(self):
        signal, finished = threading.Event(), threading.Event()

        def generate(prompt, settings, cancel_event):
            self.assertIs(cancel_event, signal)
            yield "first"
            self.assertTrue(cancel_event.wait(2))
            finished.set()

        async with inference_stream(SimpleNamespace(infer_list_stream=generate), "", {}, cancel_event=signal) as chunks:
            async for _ in chunks:
                break
        self.assertTrue(signal.is_set())
        self.assertTrue(finished.is_set())

    async def test_implicit_cancel_stops_disconnected_request(self):
        finished = threading.Event()

        def generate(prompt, settings, cancel_event):
            yield "first"
            if not cancel_event.wait(2):
                raise RuntimeError("disconnected request kept generating")
            finished.set()

        async with inference_stream(SimpleNamespace(infer_list_stream=generate), "", {}) as chunks:
            async for _ in chunks:
                break
        self.assertTrue(finished.is_set())

    async def test_chunks_and_generation_stats(self):
        stats = {}

        def generate(prompt, settings, generation_stats, cancel_event):
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
        def generate(prompt, settings, cancel_event):
            yield "first"
            raise RuntimeError("inference failed")

        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            async with inference_stream(SimpleNamespace(infer_list_stream=generate), "", {}) as chunks:
                async for _ in chunks:
                    pass

    async def check_early_exit(self, cancel):
        release, finished = threading.Event(), threading.Event()
        received = asyncio.Event()

        def generate(prompt, settings, cancel_event):
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


class TestStreamingSettingsRestoration(unittest.IsolatedAsyncioTestCase):
    async def test_bridge_restores_engine_overrides_after_completion_and_disconnect(self):
        from concurrent.futures import ThreadPoolExecutor
        from unittest.mock import Mock
        from eole.inference_engine import InferenceEnginePY
        from eole.predict.streamer import GenerationCancelled

        for disconnect in (False, True):
            with self.subTest(disconnect=disconnect), ThreadPoolExecutor(1) as pool:
                engine = InferenceEnginePY.__new__(InferenceEnginePY)
                engine.config = SimpleNamespace(world_size=1)
                engine.vocabs = {"tgt": SimpleNamespace(ids_to_tokens=["hello", "<eos>"])}
                engine.transform_pipe = None
                engine.logger = Mock()
                engine._thread_pool = pool
                engine.predictor = SimpleNamespace(model=Mock(), _tgt_eos_idx=[1], max_length=50)
                engine._build_inference_iterator = lambda **kwargs: iter([])

                def predict(_iterator, settings, streamer):
                    engine.predictor.max_length = settings["max_length"]
                    streamer.put([0])
                    if disconnect:
                        if not streamer.cancel_event.wait(2):
                            raise RuntimeError("disconnect did not reach inference worker")
                        raise GenerationCancelled()
                    streamer.end()

                engine._predict = predict
                async with inference_stream(engine, "hello", {"max_length": 7}) as chunks:
                    if disconnect:
                        async for _ in chunks:
                            break
                    else:
                        self.assertEqual([part async for part in chunks], ["hello"])
                self.assertEqual(engine.predictor.max_length, 50)
