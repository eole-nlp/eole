"""Bundled local workspace and request-scoped inference telemetry."""

import asyncio
from collections import deque
import json
from pathlib import Path
import threading
from time import perf_counter, time
import uuid

from fastapi import HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import eole
from eole.server.openai_chat import _OpenAIStreamParser, _chat_prompt_opens_thinking
from eole.server.streaming import inference_stream
from eole.predict.generator import GeneratorLM


class UIMessage(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str = Field(max_length=500000)
    reasoning_content: str | None = Field(default=None, max_length=500000)


class UIChat(BaseModel):
    model: str
    messages: list[UIMessage] = Field(min_length=1, max_length=200)
    max_tokens: int = Field(default=4096, ge=1, le=32768)
    preset: str = Field(default="fast", pattern="^(fast|creative)$")
    thinking: bool = False
    temperature: float = Field(default=0.7, ge=0, le=2)


class Hardware:
    """Optional host telemetry, cached away from the token loop."""

    def __init__(self):
        self.cached = {}
        self.updated = 0
        self.lock = threading.Lock()

    def read(self):
        with self.lock:
            if perf_counter() - self.updated < 2:
                return self.cached
            out = {"gpus": [], "cpu_percent": None, "ram_used_gib": None, "ram_total_gib": None}
            try:
                import psutil

                memory = psutil.virtual_memory()
                out.update(
                    cpu_percent=psutil.cpu_percent(),
                    ram_used_gib=memory.used / 2**30,
                    ram_total_gib=memory.total / 2**30,
                )
            except (ImportError, OSError):
                pass
            try:
                import pynvml

                pynvml.nvmlInit()
                try:
                    for index in range(pynvml.nvmlDeviceGetCount()):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        name = pynvml.nvmlDeviceGetName(handle)
                        gpu = {
                            "index": index,
                            "name": name.decode() if isinstance(name, bytes) else name,
                            "used_gib": memory.used / 2**30,
                            "total_gib": memory.total / 2**30,
                        }
                        for field, getter in (
                            ("utilization", lambda: pynvml.nvmlDeviceGetUtilizationRates(handle).gpu),
                            ("temperature", lambda: pynvml.nvmlDeviceGetTemperature(handle, 0)),
                            ("power_w", lambda: pynvml.nvmlDeviceGetPowerUsage(handle) / 1000),
                        ):
                            try:
                                gpu[field] = getter()
                            except pynvml.NVMLError:
                                gpu[field] = None
                        out["gpus"].append(gpu)
                finally:
                    pynvml.nvmlShutdown()
            except (ImportError, OSError):
                pass
            except Exception:
                # Driver telemetry is optional; inference must remain available.
                pass
            self.cached, self.updated = out, perf_counter()
            return out


def register_ui(app, server):
    assets = Path(__file__).with_name("web")
    app.mount("/ui/assets", StaticFiles(directory=assets), name="ui-assets")
    hardware = Hardware()
    recent = deque(maxlen=50)
    active = {}
    lock = asyncio.Lock()

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(assets / "index.html")

    @app.get("/ui/state")
    async def state():
        models = []
        for model_id, model in server.models.items():
            config = getattr(model.engine, "config", None) if model.loaded else None
            settings = model.pre_config
            models.append(
                {
                    "id": model_id,
                    "loaded": model.loaded,
                    "dtype": str(getattr(config, "compute_dtype", settings.get("compute_dtype", "auto"))),
                    "quant_type": getattr(config, "quant_type", ""),
                    "weight_bits": (
                        getattr(config, "w_bit", None)
                        if getattr(config, "quant_type", "") not in ("", "gguf")
                        else None
                    ),
                    "context_length": getattr(config, "context_length", settings.get("context_length")),
                    "max_tokens": getattr(config, "max_length", settings.get("max_length", 4096)),
                    "mtp_configured": bool(
                        getattr(config, "self_speculative_decoding", settings.get("self_speculative_decoding", False))
                    ),
                    "streaming_supported": getattr(config, "world_size", settings.get("world_size", 1)) == 1
                    and (not model.loaded or isinstance(model.engine.predictor, GeneratorLM)),
                }
            )
        return {
            "version": eole.__version__,
            "compile_enabled": eole.EOLE_TORCH_COMPILE,
            "compile_mode": eole.EOLE_COMPILE_MODE,
            "models": models,
            "hardware": await asyncio.to_thread(hardware.read),
            "active": [{k: v for k, v in row.items() if k != "cancel"} for row in active.values()],
            "recent": list(recent),
        }

    @app.post("/ui/cancel/{request_id}")
    async def cancel(request_id: str):
        row = active.get(request_id)
        if row is None:
            raise HTTPException(404, "Request is no longer running")
        row["cancel"].set()
        row["status"] = "stopping"
        return {"status": "stopping"}

    @app.post("/ui/chat")
    async def chat(request: UIChat):
        if request.model not in server.models:
            raise HTTPException(404, "Unknown model")
        model = server.models[request.model]

        async def events():
            request_id = uuid.uuid4().hex
            cancel_event = threading.Event()
            started = perf_counter()
            row = {
                "id": request_id,
                "model": request.model,
                "status": "queued",
                "timestamp": time(),
                "cancel": cancel_event,
            }
            active[request_id] = row

            def event(event_type, **data):
                return "data: " + json.dumps({"type": event_type, **data}, ensure_ascii=False) + "\n\n"

            yield event("start", request_id=request_id)
            try:
                async with lock:
                    if cancel_event.is_set():
                        row["status"] = "cancelled"
                        yield event("done", stats={"cancelled": True, "output_tokens": 0})
                        return
                    row["status"] = "preparing"
                    if not model.loaded:
                        await asyncio.to_thread(model.load)
                    if model.engine.config.world_size > 1 or not isinstance(model.engine.predictor, GeneratorLM):
                        raise ValueError(
                            "Interactive streaming requires a single-process decoder-only model; "
                            "use /infer for other models"
                        )
                    limit, input_limit = model.get_model_limits()
                    if request.max_tokens > limit:
                        raise ValueError(f"This model allows at most {limit} output tokens")
                    prompt = model.apply_chat_template(
                        [message.model_dump(exclude_none=True) for message in request.messages],
                        enable_thinking=request.thinking,
                    )
                    prompt_tokens = model.count_tokens(prompt)
                    if input_limit and prompt_tokens + request.max_tokens > input_limit + limit:
                        raise ValueError("Chat exceeds the context budget. Start a new chat or reduce output tokens.")
                    settings = {
                        "max_length": request.max_tokens,
                        "beam_size": 1,
                        "n_best": 1,
                        "batch_size": 1,
                        "min_length": 0,
                        "report_time": True,
                        "top_k": 1 if request.preset == "fast" else 40,
                        "temperature": 1.0 if request.preset == "fast" else request.temperature,
                        "top_p": 0.0 if request.preset == "fast" else 0.95,
                        "self_speculative_decoding": request.preset == "fast"
                        and model.engine.config.self_speculative_decoding,
                    }
                    stats = {}
                    submission = perf_counter()
                    parser = _OpenAIStreamParser(parse_tools=False, thinking=_chat_prompt_opens_thinking(prompt))
                    async with inference_stream(
                        model.engine, prompt, settings, generation_stats=stats, cancel_event=cancel_event
                    ) as chunks:
                        async for chunk in chunks:
                            row["status"] = "stopping" if cancel_event.is_set() else "generating"
                            for kind, value in parser.feed(chunk):
                                yield event("delta", kind=kind, text=value.replace("｟newline｠", "\n"))
                        for kind, value in parser.feed("", final=True):
                            yield event("delta", kind=kind, text=value.replace("｟newline｠", "\n"))
                    if stats.get("first_token_seconds") is not None:
                        stats["first_token_seconds"] += stats.get("queue_seconds", 0) + submission - started
                    stats.update(prompt_tokens=prompt_tokens, request_seconds=perf_counter() - started)
                    stats["finish_reason"] = (
                        "cancelled"
                        if stats.get("cancelled")
                        else (
                            "stop"
                            if stats.get("stopped_on_eos")
                            else "length" if stats.get("output_tokens", 0) >= request.max_tokens else "unknown"
                        )
                    )
                    row.update(stats)
                    row["status"] = "cancelled" if stats.get("cancelled") else "complete"
                    yield event("done", stats=stats)
            except Exception as exc:
                row["status"] = "error"
                yield event("error", message=str(exc))
            finally:
                cancel_event.set()
                row.pop("cancel", None)
                active.pop(request_id, None)
                recent.appendleft(row)

        return StreamingResponse(
            events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )
