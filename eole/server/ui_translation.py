"""Translation-specific prompts and structure-preserving text jobs for the UI."""

from collections import Counter
from dataclasses import dataclass
import json
import re
from time import perf_counter

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from eole.predict.generator import GeneratorLM
from eole.server.ui_runtime import event

# Hy-MT2's documented language names; other models can configure their own list.
LANGUAGES = [
    "Arabic",
    "Bengali",
    "Burmese",
    "Cantonese",
    "Chinese",
    "Czech",
    "Dutch",
    "English",
    "Filipino",
    "French",
    "German",
    "Gujarati",
    "Hebrew",
    "Hindi",
    "Indonesian",
    "Italian",
    "Japanese",
    "Kazakh",
    "Khmer",
    "Korean",
    "Malay",
    "Marathi",
    "Mongolian",
    "Persian",
    "Polish",
    "Portuguese",
    "Russian",
    "Spanish",
    "Tamil",
    "Telugu",
    "Thai",
    "Tibetan",
    "Traditional Chinese",
    "Turkish",
    "Ukrainian",
    "Urdu",
    "Uyghur",
    "Vietnamese",
]
PLACEHOLDER = re.compile(r"\{\{[^{}\n]+\}\}|\$\{[^{}\n]+\}|%(?:\d+\$)?[sdf]|</?[A-Za-z][^<>]*>")


class Term(BaseModel):
    source: str = Field(min_length=1, max_length=200)
    target: str = Field(min_length=1, max_length=200)


class TranslationRequest(BaseModel):
    model: str
    text: str = Field(min_length=1, max_length=80000)
    source: str = Field(default="auto", min_length=1, max_length=80)
    target: str = Field(default="French", min_length=1, max_length=80)
    format: str = Field(default="text", pattern="^(text|markdown|srt|vtt|json)$")
    tone: str = Field(default="default", pattern="^(default|formal|informal)$")
    context: str = Field(default="", max_length=4000)
    glossary: list[Term] = Field(default_factory=list, max_length=100)
    max_tokens: int = Field(default=4096, ge=1, le=32768)
    alternative: bool = False


def translation_capabilities(model):
    options = getattr(model, "translation_config", {}) or {}
    raw = model.loaded and not isinstance(model.engine.predictor, GeneratorLM)
    return {
        "prompt_style": options.get("prompt_style", "raw" if raw else "instruction"),
        "languages": options.get("languages", LANGUAGES),
        "source": options.get("source_language", "auto"),
        "target": options.get("target_language", "French"),
        "preferred": bool(options),
        "available": not raw or bool(options.get("source_language") and options.get("target_language")),
    }


@dataclass
class Segment:
    start: int
    end: int
    text: str
    json_string: bool = False


class Document:
    """Keep separators, JSON keys and subtitle metadata outside model generation."""

    def __init__(self, text, format):
        self.text = text
        self.segments = []
        self.format = format
        if format == "json":

            def invalid(value):
                raise ValueError(f"Invalid JSON number: {value}")

            json.loads(text, parse_constant=invalid)
            for match in re.finditer(r'"(?:[^"\\]|\\.)*"', text):
                if text[match.end() :].lstrip().startswith(":"):
                    continue
                value = json.loads(match.group())
                if value.strip():
                    self.segments.append(Segment(match.start(), match.end(), value, True))
        elif format in ("srt", "vtt"):
            if format == "vtt" and not text.lstrip("\ufeff").startswith("WEBVTT"):
                raise ValueError("A WebVTT file must start with WEBVTT")
            for block in re.finditer(r"[^\r\n]+(?:\r?\n(?!\r?\n)[^\r\n]+)*", text):
                lines = list(re.finditer(r"[^\r\n]+", block.group()))
                if format == "vtt" and re.match(r"^(?:WEBVTT|NOTE|STYLE|REGION)(?:\s|$)", lines[0].group()):
                    continue
                times = [i for i, line in enumerate(lines) if "-->" in line.group()]
                if not times:
                    raise ValueError("A subtitle cue is missing its timestamp")
                i = times[0]
                timing = lines[i].group()
                if not re.match(r"(?:\d{2}:)?\d{2}:\d{2}[.,]\d{3}\s+-->\s+", timing):
                    raise ValueError("Invalid subtitle timestamp")
                if i + 1 < len(lines):
                    start = block.start() + lines[i + 1].start()
                    end = block.start() + lines[-1].end()
                    self.segments.append(Segment(start, end, text[start:end]))
            if not self.segments:
                raise ValueError("No subtitle cues found")
        else:
            for match in re.finditer(r"\S[\s\S]*?(?=\r?\n[ \t]*\r?\n|\Z)", text):
                start, end = match.start(), match.end()
                end = start + len(match.group().rstrip())
                # Bound prompt sizes without cutting through words. Preserve the
                # original inter-segment whitespace in the document skeleton.
                while end - start > 2400:
                    split = text.rfind(" ", start + 1000, start + 2400)
                    next_start = split + 1
                    if split < 0:
                        punctuation = list(re.finditer(r"[。！？；]", text[start + 1000 : start + 2400]))
                        if punctuation:
                            split = start + 1000 + punctuation[-1].end()
                        elif re.search(r"[\u3040-\u30ff\u3400-\u9fff]", text[start : start + 2400]):
                            split = start + 2400  # CJK text need not contain word spaces.
                        else:
                            raise ValueError("A text span is too long to split safely")
                        next_start = split
                    self.segments.append(Segment(start, split, text[start:split]))
                    start = next_start
                if start < end:
                    self.segments.append(Segment(start, end, text[start:end]))
        if not self.segments:
            raise ValueError("There is no translatable text")
        if len(self.segments) > 200:
            raise ValueError("This document exceeds 200 segments; split it into smaller files")
        self.outputs = {}

    def render(self):
        result, cursor = [], 0
        for i, segment in enumerate(self.segments):
            result.append(self.text[cursor : segment.start])
            value = self.outputs.get(i)
            if value is None:
                result.append(self.text[segment.start : segment.end])
            else:
                if self.format in ("srt", "vtt") and "\r\n" in self.text:
                    value = value.replace("\r\n", "\n").replace("\n", "\r\n")
                result.append(json.dumps(value, ensure_ascii=False) if segment.json_string else value)
            cursor = segment.end
        return "".join(result) + self.text[cursor:]


def translation_prompt(request, text):
    parts = []
    if request.glossary:
        parts.append(
            "Reference the following translations:\n"
            + "\n".join(
                f"{json.dumps(term.source, ensure_ascii=False)} translates to "
                f"{json.dumps(term.target, ensure_ascii=False)}"
                for term in request.glossary
            )
        )
    if request.context.strip():
        parts.append("[Background Information]\n" + request.context)
    if request.source != "auto":
        parts.append(f"The source language is {request.source}.")
    if request.tone != "default":
        parts.append(f"The translation style must conform to a {request.tone} tone.")
    if request.format == "markdown":
        parts.append("Preserve Markdown formatting, code, numbers, and URLs.")
    if PLACEHOLDER.search(text):
        parts.append("Preserve placeholders and inline tags exactly, without translating them.")
    if request.format in ("srt", "vtt"):
        parts.append("Preserve the number of subtitle text lines and all inline tags.")
    # Keep the documented Hy-MT2 instruction immediately adjacent to the source.
    # Appending extra instructions here can cause the model to translate them.
    parts.append(
        f"Translate the following text into {request.target}. "
        "Note that you should only output the translated result without any additional explanation:\n" + text
    )
    return "\n\n".join(parts)


def register_translation(app, server, workspace):
    @app.post("/ui/translate")
    async def translate(request: TranslationRequest):
        if request.model not in server.models:
            raise HTTPException(404, "Unknown model")
        if request.source == request.target:
            raise HTTPException(422, "Choose different source and target languages")
        try:
            document = Document(request.text, request.format)
        except (ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        model = server.models[request.model]

        async def events():
            started = perf_counter()
            async with workspace.track(request.model, "translate") as row:
                cancel = row["cancel"]
                yield event("start", request_id=row["id"], segments=len(document.segments))
                try:
                    async with workspace.lock:
                        if not cancel.is_set():
                            row["status"] = "preparing"
                            await workspace.prepare(request.model, model, row, speculation="off")
                        caps = translation_capabilities(model)
                        if request.target not in caps["languages"] or (
                            request.source != "auto" and request.source not in caps["languages"]
                        ):
                            raise ValueError("The selected language is not configured for this model")
                        raw = caps["prompt_style"] == "raw"
                        if raw:
                            if not caps["available"]:
                                raise ValueError("Configure source_language and target_language for this native model")
                            if request.source != caps["source"] or request.target != caps["target"]:
                                raise ValueError(
                                    "This native model has a fixed language direction; use its configured pair"
                                )
                            if request.glossary or request.context.strip() or request.tone != "default":
                                raise ValueError(
                                    "Glossary, context and tone require an instruction-based translation model"
                                )
                        outputs, token_count, prompt_count, decode_seconds, decode_tokens = [], 0, 0, 0, 0
                        first_token = None
                        incomplete = False
                        for i, segment in enumerate(document.segments):
                            if cancel.is_set():
                                break
                            row["status"] = "preparing"
                            yield event("progress", segment=i, total=len(document.segments))
                            prompt = (
                                segment.text
                                if raw
                                else model.apply_chat_template(
                                    [{"role": "user", "content": translation_prompt(request, segment.text)}]
                                )
                            )
                            settings = dict(
                                max_length=request.max_tokens,
                                min_length=0,
                                max_length_ratio=0,
                                n_best=1,
                                batch_size=1,
                                beam_size=getattr(model.engine.config, "beam_size", 1) if raw else 1,
                                report_time=True,
                                self_speculative_decoding=False,
                                top_k=20 if request.alternative else 1,
                                top_p=0.6 if request.alternative else 0.0,
                                temperature=0.7 if request.alternative else 1.0,
                            )
                            value = ""
                            async for item in workspace.generate(model, prompt, settings, cancel, row):
                                if item["type"] == "stats":
                                    stats = item["stats"]
                                elif item["kind"] == "text":
                                    if first_token is None:
                                        first_token = perf_counter() - started
                                    value += item["text"]
                                    yield event("delta", segment=i, text=value)
                            prompt_count += stats.get("prompt_tokens", 0)
                            if stats.get("output_tokens") is None:
                                token_count = None
                            elif token_count is not None:
                                token_count += stats["output_tokens"]
                            decode_seconds += stats.get("decode_seconds") or 0
                            decode_tokens += stats.get("decode_tokens") or 0
                            if cancel.is_set():
                                break
                            if stats["finish_reason"] == "length":
                                incomplete = True
                                yield event(
                                    "warning",
                                    message="Output limit reached. This segment is incomplete; increase the limit.",
                                )
                                break
                            value = value.strip()
                            if Counter(PLACEHOLDER.findall(segment.text)) != Counter(PLACEHOLDER.findall(value)):
                                raise ValueError(
                                    f"Segment {i + 1} changed a placeholder or tag. Review it before exporting."
                                )
                            if request.format in ("srt", "vtt") and len(value.splitlines()) != len(
                                segment.text.splitlines()
                            ):
                                raise ValueError(
                                    f"Segment {i + 1} changed subtitle line count. Review or shorten the cue."
                                )
                            document.outputs[i] = value
                            outputs.append(dict(source=segment.text, target=value))
                            yield event("segment", segment=i, text=value, document=document.render())
                        complete = len(outputs) == len(document.segments)
                        status = "cancelled" if cancel.is_set() else "complete" if complete else "incomplete"
                        stats = dict(
                            output_tokens=token_count,
                            prompt_tokens=prompt_count,
                            decode_tokens_per_second=decode_tokens / decode_seconds if decode_seconds else None,
                            first_token_seconds=first_token,
                            request_seconds=perf_counter() - started,
                            mtp_active=False,
                            cancelled=cancel.is_set(),
                            completed_segments=len(outputs),
                            total_segments=len(document.segments),
                            incomplete=incomplete or not complete,
                        )
                        row.update(stats, status=status)
                        yield event("done", stats=stats, text=document.render(), segments=outputs, complete=complete)
                except Exception as exc:
                    row["status"] = "error"
                    yield event("error", message=str(exc), text=document.render(), complete=False)

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
