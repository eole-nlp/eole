"""Shared request scheduling, cancellation and generation for workspace views."""

import asyncio
from collections import deque
from contextlib import asynccontextmanager
import json
import threading
from time import perf_counter, time
import uuid

from eole.predict.generator import GeneratorLM
from eole.server.openai_chat import _OpenAIStreamParser, _chat_prompt_opens_thinking
from eole.server.streaming import inference_stream


def event(event_type, **data):
    return "data: " + json.dumps({"type": event_type, **data}, ensure_ascii=False) + "\n\n"


class Workspace:
    def __init__(self, server=None):
        self.recent = deque(maxlen=50)
        self.active = {}
        self.server = server
        self.lock = server.model_lock if getattr(server, "model_policy", None) == "single" else asyncio.Lock()

    async def prepare(self, model_id, model, row):
        if self.server is not None and hasattr(self.server, "prepare_model"):
            await self.server.prepare_model(model_id, row)
        elif not model.loaded:
            await asyncio.to_thread(model.load)

    @asynccontextmanager
    async def track(self, model_id, task):
        request_id = uuid.uuid4().hex
        row = dict(
            id=request_id, model=model_id, task=task, status="queued", timestamp=time(), cancel=threading.Event()
        )
        self.active[request_id] = row
        try:
            yield row
        finally:
            row.pop("cancel").set()
            self.active.pop(request_id, None)
            if row["status"] not in ("complete", "error", "cancelled", "incomplete"):
                row["status"] = "cancelled"
            self.recent.appendleft(row)

    async def generate(self, model, prompt, settings, cancel, row, budget_text=None):
        """Yield text deltas and one worker stats record; never mutate API defaults."""
        limit, input_limit = model.get_model_limits()
        if settings["max_length"] > limit:
            raise ValueError(f"This model allows at most {limit} output tokens")
        prompt_tokens = model.count_tokens(budget_text if budget_text is not None else prompt)
        decoder_only = isinstance(model.engine.predictor, GeneratorLM)
        if input_limit and prompt_tokens + (settings["max_length"] if decoder_only else 0) > input_limit + limit:
            raise ValueError("Input exceeds the context budget. Shorten the text or reduce output tokens.")
        if model.engine.config.world_size > 1:
            raise ValueError("The workspace currently requires single-process inference")
        stats = {}
        submitted = perf_counter()
        if decoder_only:
            parser = _OpenAIStreamParser(
                parse_tools=False,
                thinking=_chat_prompt_opens_thinking(prompt["text"] if isinstance(prompt, dict) else prompt),
            )
            async with inference_stream(
                model.engine, prompt, settings, generation_stats=stats, cancel_event=cancel
            ) as chunks:
                async for chunk in chunks:
                    row["status"] = "stopping" if cancel.is_set() else "generating"
                    for kind, value in parser.feed(chunk):
                        yield dict(type="delta", kind=kind, text=value.replace("｟newline｠", "\n"))
                for kind, value in parser.feed("", final=True):
                    yield dict(type="delta", kind=kind, text=value.replace("｟newline｠", "\n"))
            if stats.get("first_token_seconds") is not None:
                stats["first_token_seconds"] += stats.get("queue_seconds", 0)
        else:
            # Encoder-decoder models use native source inputs, not chat templates.
            # Drain an in-flight segment on disconnect before releasing the lock.
            def run():
                predictor = model.engine.predictor
                previous = {key: getattr(predictor, key) for key in settings if hasattr(predictor, key)}
                try:
                    if cancel.is_set():
                        return ""
                    _, _, predictions = model.engine.infer_list([prompt], settings=settings)
                    return predictions[0][0].replace("｟newline｠", "\n")
                finally:
                    for key, value in previous.items():
                        setattr(predictor, key, value)

            row["status"] = "generating"
            future = asyncio.get_running_loop().run_in_executor(model.engine._thread_pool, run)
            try:
                text = await asyncio.shield(future)
            finally:
                if not future.done():
                    cancel.set()
                    await asyncio.shield(future)
            if not cancel.is_set():
                yield dict(type="delta", kind="text", text=text)
            # infer_list doesn't expose exact generated IDs or synchronized decode
            # timings. Do not label token estimates as measured decode throughput.
            stats.update(output_tokens=None, decode_tokens_per_second=None, first_token_seconds=None, mtp_active=False)
        stats.update(prompt_tokens=prompt_tokens, cancelled=cancel.is_set(), request_seconds=perf_counter() - submitted)
        stats["finish_reason"] = (
            "cancelled"
            if cancel.is_set()
            else (
                "stop"
                if stats.get("stopped_on_eos")
                else (
                    "length"
                    if stats.get("output_tokens", 0) is not None
                    and stats.get("output_tokens", 0) >= settings["max_length"]
                    else "unknown"
                )
            )
        )
        yield dict(type="stats", stats=stats)
