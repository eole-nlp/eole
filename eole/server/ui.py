"""Bundled local workspace and request-scoped inference telemetry."""

import asyncio
from pathlib import Path
import threading
from typing import Literal
from time import perf_counter

from fastapi import HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

import eole
from eole.server.ui_images import ChatImage, chat_prompt, image_capabilities
from eole.server.ui_runtime import Workspace, event
from eole.server.ui_speculation import capabilities, selection
from eole.server.ui_knowledge import register_knowledge
from eole.server.ui_ocr import register_ocr
from eole.server.ui_ocr_models import ocr_adapter
from eole.server.ui_documents import register_document_import
from eole.server.ui_translation import register_translation, translation_capabilities
from eole.predict.generator import GeneratorLM


class UIMessage(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str = Field(max_length=500000)
    images: list[ChatImage] = Field(default_factory=list, max_length=4)
    reasoning_content: str | None = Field(default=None, max_length=500000)


class UIChat(BaseModel):
    model: str
    messages: list[UIMessage] = Field(min_length=1, max_length=200)
    max_tokens: int = Field(default=4096, ge=1, le=32768)
    preset: str = Field(default="fast", pattern="^(fast|creative)$")
    vocabulary_quantization: Literal["none", "int8", "int4"] | None = None
    speculation: Literal["off", "mtp", "dflash2"] | None = None
    dflash2_quantization: Literal["none", "int8"] | None = None
    thinking: bool = False
    temperature: float = Field(default=0.7, ge=0, le=2)

    @model_validator(mode="after")
    def validate_images(self):
        if sum(len(message.images) for message in self.messages) > 4:
            raise ValueError("At most four images are supported per conversation request")
        if any(message.images and message.role != "user" for message in self.messages):
            raise ValueError("Images can only be attached to user messages")
        return self


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
    workspace = Workspace(server)
    recent, active, lock = workspace.recent, workspace.active, workspace.lock
    register_translation(app, server, workspace)
    register_document_import(app)
    register_ocr(app, server, workspace)
    register_knowledge(app, server, workspace)

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(assets / "index.html")

    @app.get("/ui/state")
    async def state():
        models = []
        for model_id, model in server.models.items():
            config = getattr(model.engine, "config", None) if model.loaded else None
            settings = model.pre_config
            ocr = ocr_adapter(model)
            translation = translation_capabilities(model)
            models.append(
                {
                    "id": model_id,
                    "loaded": model.loaded,
                    "translation": {**translation, "available": translation["available"] and ocr is None},
                    "chat_available": ocr is None
                    and not translation["preferred"]
                    and translation["prompt_style"] != "raw",
                    "images_supported": image_capabilities(model),
                    "ocr": ocr is not None,
                    "ocr_layout": bool(ocr and ocr.layout),
                    "dtype": str(getattr(config, "compute_dtype", settings.get("compute_dtype", "auto"))),
                    "quant_type": getattr(config, "quant_type", ""),
                    "weight_bits": (
                        getattr(config, "w_bit", None)
                        if getattr(config, "quant_type", "") not in ("", "gguf")
                        else None
                    ),
                    "vocabulary_quantization": getattr(
                        config, "vocabulary_quantization", settings.get("vocabulary_quantization", "none")
                    ),
                    "context_length": getattr(config, "context_length", settings.get("context_length")),
                    "max_tokens": getattr(config, "max_length", settings.get("max_length", 4096)),
                    "mtp_configured": capabilities(model)["mtp_available"],
                    "speculation": capabilities(model),
                    "streaming_supported": getattr(config, "world_size", settings.get("world_size", 1)) == 1
                    and (not model.loaded or isinstance(model.engine.predictor, GeneratorLM)),
                }
            )
        return {
            "workspace": getattr(server, "workspace", {}),
            "model_policy": getattr(server, "model_policy", "keep_loaded"),
            "model_activity": getattr(server, "model_activity", None),
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
            started = perf_counter()
            async with workspace.track(request.model, "chat") as row:
                cancel_event = row["cancel"]
                yield event("start", request_id=row["id"])
                try:
                    async with lock:
                        if cancel_event.is_set():
                            row["status"] = "cancelled"
                            yield event("done", stats={"cancelled": True, "output_tokens": 0})
                            return
                        row["status"] = "preparing"
                        mode = selection(
                            model,
                            request.speculation,
                            eligible=request.preset == "fast" and not any(m.images for m in request.messages),
                        )
                        await workspace.prepare(
                            request.model,
                            model,
                            row,
                            vocabulary_quantization=request.vocabulary_quantization,
                            speculation=mode,
                            dflash2_quantization=request.dflash2_quantization,
                        )
                        if not isinstance(model.engine.predictor, GeneratorLM):
                            raise ValueError("Chat requires a decoder-only model; use Translate for translation models")
                        prompt, budget_text = await asyncio.to_thread(
                            chat_prompt, model, request.messages, request.thinking
                        )
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
                            "self_speculative_decoding": mode != "off",
                        }
                        submission = perf_counter()
                        async for item in workspace.generate(
                            model, prompt, settings, cancel_event, row, budget_text=budget_text
                        ):
                            if item["type"] == "stats":
                                stats = item["stats"]
                            else:
                                yield event("delta", kind=item["kind"], text=item["text"])
                        if stats.get("first_token_seconds") is not None:
                            stats["first_token_seconds"] += submission - started
                        stats["request_seconds"] = perf_counter() - started
                        row.update(stats)
                        row["status"] = "cancelled" if stats.get("cancelled") else "complete"
                        yield event("done", stats=stats)
                except Exception as exc:
                    row["status"] = "error"
                    yield event("error", message=str(exc))

        return StreamingResponse(
            events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )
