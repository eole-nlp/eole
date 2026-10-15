"""Bridge blocking inference into async API handlers with worker cleanup."""

import asyncio
import threading
from contextlib import asynccontextmanager


@asynccontextmanager
async def inference_stream(
    engine, prompt, settings, generation_stats=None, cancel_event=None, preserve_special_tokens=False
):
    """Yield an async chunk iterator and drain its worker when the consumer exits.

    Without an explicit cancel_event, disconnects stop delivery only. With
    one, signal cooperative cancellation and still drain the worker before
    releasing request-local state.
    The engine's persistent single-worker pool still owns inference scheduling.
    """
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()
    cancelled = threading.Event()
    completed = False

    def produce():
        try:
            kwargs = {"settings": settings}
            if generation_stats is not None:
                kwargs["generation_stats"] = generation_stats
            if cancel_event is not None:
                kwargs["cancel_event"] = cancel_event
            if preserve_special_tokens:
                kwargs["preserve_special_tokens"] = True
            for chunk in engine.infer_list_stream(prompt, **kwargs):
                if not cancelled.is_set():
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    async def chunks():
        nonlocal completed
        while True:
            item = await queue.get()
            if item is None:
                completed = True
                return
            if isinstance(item, Exception):
                raise item
            yield item

    worker = loop.run_in_executor(None, produce)
    try:
        yield chunks()
    finally:
        cancelled.set()
        if cancel_event is not None and not completed:
            cancel_event.set()
        await asyncio.shield(worker)
