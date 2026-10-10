"""Bridge blocking inference into async API handlers with worker cleanup."""

import asyncio
import threading
from contextlib import asynccontextmanager


@asynccontextmanager
async def inference_stream(engine, prompt, settings, generation_stats=None):
    """Yield an async chunk iterator and drain its worker when the consumer exits.

    Disconnects stop delivery, not inference. Keep the worker alive until it
    finishes so request-local state is not released while generation continues.
    The engine's persistent single-worker pool still owns inference scheduling.
    """
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()
    cancelled = threading.Event()

    def produce():
        try:
            kwargs = {"settings": settings}
            if generation_stats is not None:
                kwargs["generation_stats"] = generation_stats
            for chunk in engine.infer_list_stream(prompt, **kwargs):
                if not cancelled.is_set():
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    async def chunks():
        while True:
            item = await queue.get()
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    worker = loop.run_in_executor(None, produce)
    try:
        yield chunks()
    finally:
        cancelled.set()
        await asyncio.shield(worker)
