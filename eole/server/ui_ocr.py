"""Local PDF rasterization and request-scoped, page-at-a-time OCR."""

import asyncio
import base64
import binascii
from io import BytesIO
from pathlib import Path
import re
from time import perf_counter

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from eole.server.streaming import inference_stream
from eole.server.ui_runtime import event


class OCRImport(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    data: str = Field(max_length=28000000)


class OCRPage(BaseModel):
    model: str
    data: str = Field(max_length=4000000)
    max_tokens: int = Field(default=4096, ge=1, le=16384)


def decode_data(data, limit):
    try:
        raw = base64.b64decode(data, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Invalid file encoding") from exc
    if len(raw) > limit:
        raise ValueError(f"File exceeds {limit // 1000000} MB")
    return raw


def encode_page(image):
    image = image.convert("RGB")
    image.thumbnail((1024, 1024))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return dict(data=base64.b64encode(buffer.getvalue()).decode(), width=image.width, height=image.height)


def open_image(raw):
    with Image.open(BytesIO(raw)) as image:
        if image.width * image.height > 25000000:
            raise ValueError("Image limit: 25 megapixels")
        image.load()
        return ImageOps.exif_transpose(image).convert("RGB")


def import_pages(request):
    raw = decode_data(request.data, 20000000)
    if Path(request.name).suffix.lower() == ".pdf":
        try:
            import pymupdf
        except ImportError as exc:
            raise ValueError("PDF OCR requires pip install 'eole[ocr]'") from exc
        with pymupdf.open(stream=raw, filetype="pdf") as document:
            if document.needs_pass:
                raise ValueError("Unlock the PDF before importing")
            if not 1 <= len(document) <= 50:
                raise ValueError("PDF limit: 1–50 pages")
            pages = []
            for page in document:
                scale = 1024 / max(page.rect.width, page.rect.height)
                pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False, colorspace=pymupdf.csRGB)
                pages.append(encode_page(Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)))
            return pages
    return [encode_page(open_image(raw))]


def ocr_capabilities(model):
    """This workflow uses the DeepSeek-OCR prompt and grounding format."""
    name = str(getattr(model, "model_path", "")) + " " + str(getattr(model, "model_id", ""))
    return bool(re.search(r"deepseek[-_]?ocr", name, re.I))


def parse_ocr(text):
    boxes = []
    pattern = r"<\|ref\|>(.*?)<\|/ref\|>\s*<\|det\|>(.*?)<\|/det\|>"
    for label, coordinates in re.findall(pattern, text, re.S):
        for values in re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", coordinates):
            x1, y1, x2, y2 = map(int, values)
            if 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000:
                boxes.append(dict(label=label, x=x1 / 10, y=y1 / 10, width=(x2 - x1) / 10, height=(y2 - y1) / 10))
    markdown = re.sub(pattern, "", text, flags=re.S)
    # Cancellation can end inside a grounding marker. Keep it only in raw output.
    markdown = re.sub(r"<\|ref\|>.*\Z", "", markdown, flags=re.S)
    markdown = re.sub(r"<\|(?:endoftext|eos)\|>", "", markdown)
    return dict(markdown=markdown.strip(), boxes=boxes)


def register_ocr(app, server, workspace):
    @app.post("/ui/ocr/import")
    def upload(request: OCRImport):
        try:
            return dict(name=Path(request.name).name, pages=import_pages(request))
        except Exception as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/ui/ocr/page")
    async def recognize(request: OCRPage):
        started = perf_counter()
        model = server.models.get(request.model)
        if model is None:
            raise HTTPException(404, "Unknown model")
        if not ocr_capabilities(model):
            raise HTTPException(422, "Select a DeepSeek-OCR model for this workflow")
        try:
            image = await asyncio.to_thread(open_image, decode_data(request.data, 3000000))
            image.thumbnail((1024, 1024))
        except Exception as exc:
            raise HTTPException(422, str(exc)) from exc

        async def events():
            async with workspace.track(request.model, "ocr") as row:
                cancel = row["cancel"]
                yield event("start", request_id=row["id"])
                try:
                    async with workspace.lock:
                        if cancel.is_set():
                            row["status"] = "cancelled"
                            yield event("done", cancelled=True)
                            return
                        row["status"] = "preparing"
                        if not model.loaded:
                            await asyncio.to_thread(model.load)
                        config = model.engine.config
                        if config.world_size != 1 or config.data_type != "image":
                            raise ValueError("OCR requires a single-process model configured with data_type: image")
                        limit, _ = model.get_model_limits()
                        if request.max_tokens > limit:
                            raise ValueError(f"Model output limit is {limit} tokens")
                        prompt = {
                            "text": "{image}\n<|grounding|>Convert the document to markdown.",
                            "images": {"image": image},
                        }
                        settings = dict(
                            max_length=request.max_tokens,
                            batch_size=1,
                            beam_size=1,
                            n_best=1,
                            top_k=1,
                            top_p=0.0,
                            temperature=1.0,
                            min_length=0,
                            self_speculative_decoding=False,
                            report_time=True,
                        )
                        stats, parts = {}, []
                        async with inference_stream(
                            model.engine,
                            prompt,
                            settings,
                            generation_stats=stats,
                            cancel_event=cancel,
                            preserve_special_tokens=True,
                        ) as chunks:
                            async for chunk in chunks:
                                row["status"] = "stopping" if cancel.is_set() else "generating"
                                text = chunk.replace("｟newline｠", "\n")
                                parts.append(text)
                                yield event("delta", text=text)
                        raw = "".join(parts)
                        stats["request_seconds"] = perf_counter() - started
                        incomplete = (
                            not stats.get("stopped_on_eos", False)
                            and stats.get("output_tokens", 0) >= request.max_tokens
                        )
                        row.update(stats)
                        row["status"] = "cancelled" if cancel.is_set() else "incomplete" if incomplete else "complete"
                        yield event(
                            "done",
                            raw=raw,
                            **parse_ocr(raw),
                            stats=stats,
                            cancelled=cancel.is_set(),
                            incomplete=incomplete,
                        )
                except Exception as exc:
                    row["status"] = "error"
                    yield event("error", message=str(exc))

        return StreamingResponse(
            events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )
