"""Stateless, text/function-tool Responses API for local coding clients.

No server-side conversation storage, hosted tools, or encrypted reasoning state.
Qwen tool blocks are buffered individually; ordinary text streams immediately.
"""

import asyncio
import json
import threading
import time
import uuid

from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from eole.constants import DefaultTokens


class ResponsesRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    input: str | list[dict]
    instructions: str | None = None
    tools: list[dict] = Field(default_factory=list)
    tool_choice: str | dict = "auto"
    stream: bool = False
    store: bool = False
    previous_response_id: str | None = None
    max_output_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, ge=0, le=1)


def _text(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError("Expected text or a list of text content parts")
    parts = []
    for part in content:
        if not isinstance(part, dict):
            raise ValueError("Content parts must be objects")
        if part.get("type") not in ("input_text", "output_text", "text"):
            raise ValueError(f"Unsupported content type: {part.get('type')}")
        parts.append(part["text"])
    return "\n".join(parts)


def responses_messages(request):
    """Translate replayed Responses items, preserving tool call/result IDs."""
    if request.previous_response_id or request.store:
        raise ValueError("Use store=false and replay input history; previous_response_id/storage are unsupported")
    extra = request.model_extra or {}
    for name in ("background", "conversation", "context_management"):
        if extra.get(name):
            raise ValueError(f"{name} is unsupported")
    if extra.get("truncation", "disabled") != "disabled":
        raise ValueError("Automatic truncation is unsupported")
    fmt = (extra.get("text") or {}).get("format", {}).get("type", "text")
    if fmt != "text":
        raise ValueError("Only text output format is supported")
    messages = []
    if request.instructions:
        messages.append({"role": "system", "content": request.instructions})
    items = [{"role": "user", "content": request.input}] if isinstance(request.input, str) else request.input
    for item in items:
        kind = item.get("type", "message")
        if kind == "message":
            role = item.get("role", "user")
            if role == "developer":
                role = "system"
            if role not in ("system", "user", "assistant"):
                raise ValueError(f"Unsupported message role: {role}")
            messages.append({"role": role, "content": _text(item.get("content", ""))})
        elif kind in ("function_call", "custom_tool_call"):
            args = json.loads(item["arguments"]) if kind == "function_call" else {"input": item["input"]}
            if not isinstance(args, dict):
                raise ValueError("Tool arguments must be a JSON object")
            call = {
                "id": item["call_id"],
                "type": "function",
                "function": {
                    "name": ((item.get("namespace") + ".") if item.get("namespace") else "") + item["name"],
                    "arguments": args,
                },
            }
            if messages and messages[-1]["role"] == "assistant":
                messages[-1].setdefault("tool_calls", []).append(call)
            else:
                messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
        elif kind in ("function_call_output", "custom_tool_call_output"):
            messages.append({"role": "tool", "tool_call_id": item["call_id"], "content": _text(item["output"])})
        elif kind == "reasoning":
            if item.get("encrypted_content"):
                raise ValueError("Encrypted reasoning items are unsupported")
            # Visible reasoning summaries are not needed to replay the tool loop.
        else:
            raise ValueError(f"Unsupported input item: {kind}")
    return messages


def responses_tools(request):
    tools, schemas, custom = [], {}, set()
    expanded = []
    for tool in request.tools:
        if tool.get("type") == "namespace":
            for child in tool.get("tools", []):
                expanded.append({**child, "name": tool["name"] + "." + child["name"]})
        else:
            expanded.append(tool)
    for tool in expanded:
        kind = tool.get("type")
        if kind not in ("function", "custom"):
            raise ValueError(f"Unsupported tool type: {kind}; use client-executed function/custom tools")
        name = tool["name"]
        if name in schemas:
            raise ValueError(f"Duplicate tool name: {name}")
        schema = tool.get("parameters", {"type": "object", "properties": {}})
        if not isinstance(schema, dict):
            raise ValueError("Tool parameters must be a JSON schema object")
        description = tool.get("description", "")
        if kind == "custom":
            custom.add(name)
            schema = {"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]}
            description += "\nReturn the complete raw tool input in the input string argument."
        schemas[name] = schema
        tools.append({"type": "function", "function": {"name": name, "description": description, "parameters": schema}})
    choice = request.tool_choice
    if isinstance(choice, dict):
        if choice.get("type") not in ("function", "custom") or choice.get("name") not in schemas:
            raise ValueError("tool_choice must name a declared function/custom tool")
        choice = {"type": "function", "function": {"name": choice["name"]}}
    elif choice not in ("auto", "none", "required"):
        raise ValueError("Unsupported tool_choice")
    if choice == "required" and not tools:
        raise ValueError("tool_choice=required needs tools")
    return tools, schemas, custom, choice


class OutputParser:
    """Incremental tag scanner; never leak partial reasoning/tool markup."""

    def __init__(self):
        self.pending = ""
        self.thinking = False
        self.tool_end = None
        self.tool = ""

    def feed(self, chunk, final=False):
        self.pending += chunk
        result = []
        tags = ("<think>", "</think>", "<tool_call>", "<tool_use", DefaultTokens.SEP)
        while self.pending:
            if self.tool_end:
                end = self.pending.find(self.tool_end)
                if end < 0:
                    self.tool += self.pending
                    self.pending = ""
                    # Retain a suffix so a closing tag split across chunks is found.
                    keep = len(self.tool_end) - 1
                    self.pending, self.tool = self.tool[-keep:], self.tool[:-keep]
                    break
                end += len(self.tool_end)
                # Normalize after assembling the whole block so sentinels
                # split across chunks are replaced before XML/schema parsing.
                tool = (self.tool + self.pending[:end]).replace(DefaultTokens.SEP, "\n")
                result.append(("tool", tool))
                self.pending, self.tool, self.tool_end = self.pending[end:], "", None
                continue
            found = [(self.pending.find(tag), tag) for tag in tags if tag in self.pending]
            if found:
                pos, tag = min(found)
                if pos and not self.thinking:
                    result.append(("text", self.pending[:pos]))
                self.pending = self.pending[pos + len(tag) :]
                if tag == "<think>":
                    self.thinking = True
                elif tag == "</think>":
                    self.thinking = False
                elif tag in ("<tool_call>", "<tool_use"):
                    self.tool = tag
                    self.tool_end = "</tool_call>" if tag == "<tool_call>" else "</tool_use>"
                elif not self.thinking:
                    result.append(("text", "\n"))
                continue
            keep = 0
            if not final:
                for size in range(1, min(len(self.pending), max(map(len, tags))) + 1):
                    if any(tag.startswith(self.pending[-size:]) for tag in tags):
                        keep = size
            text = self.pending[:-keep] if keep else self.pending
            self.pending = self.pending[-keep:] if keep else ""
            if text and not self.thinking:
                result.append(("text", text))
            break
        if final and self.thinking:
            raise ValueError("Model output ended inside a reasoning block")
        if final and self.tool_end:
            raise ValueError("Model output ended inside a tool call; refusing to execute incomplete arguments")
        return result


class ResponseEvents:
    def __init__(self, request, schemas, custom):
        self.request, self.schemas, self.custom = request, schemas, custom
        self.sequence = 0
        self.output = []
        self.active = None
        self.response = {
            "id": "resp_" + uuid.uuid4().hex,
            "object": "response",
            "created_at": int(time.time()),
            "status": "in_progress",
            "model": request.model,
            "output": [],
            "error": None,
            "incomplete_details": None,
            "usage": None,
            "store": False,
        }

    def event(self, kind, **fields):
        event = {"type": kind, "sequence_number": self.sequence, **fields}
        self.sequence += 1
        return event

    def start(self):
        return [
            self.event("response.created", response=dict(self.response)),
            self.event("response.in_progress", response=dict(self.response)),
        ]

    def close_text(self):
        if self.active is None:
            return []
        item = self.active
        index = len(self.output) - 1
        item["status"] = "completed"
        part = item["content"][0]
        fields = {"item_id": item["id"], "output_index": index, "content_index": 0}
        self.active = None
        return [
            self.event("response.output_text.done", **fields, text=part["text"]),
            self.event("response.content_part.done", **fields, part=dict(part)),
            self.event("response.output_item.done", output_index=index, item=item),
        ]

    def accept(self, kind, value):
        from eole.bin.run.serve import AnthropicTool, _coerce_tool_inputs_from_schema, _parse_anthropic_response_content

        events = []
        if kind == "text":
            if self.active is None:
                self.active = {
                    "type": "message",
                    "id": "msg_" + uuid.uuid4().hex,
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "", "annotations": [], "logprobs": []}],
                }
                self.output.append(self.active)
                events.append(
                    self.event(
                        "response.output_item.added",
                        output_index=len(self.output) - 1,
                        item={**self.active, "content": []},
                    )
                )
                events.append(
                    self.event(
                        "response.content_part.added",
                        item_id=self.active["id"],
                        output_index=len(self.output) - 1,
                        content_index=0,
                        part=dict(self.active["content"][0]),
                    )
                )
            self.active["content"][0]["text"] += value
            events.append(
                self.event(
                    "response.output_text.delta",
                    item_id=self.active["id"],
                    output_index=len(self.output) - 1,
                    content_index=0,
                    delta=value,
                    logprobs=[],
                )
            )
            return events
        events.extend(self.close_text())
        blocks, _ = _parse_anthropic_response_content(value)
        _coerce_tool_inputs_from_schema(
            blocks, [AnthropicTool(name=name, input_schema=schema) for name, schema in self.schemas.items()]
        )
        if len(blocks) != 1 or blocks[0].get("type") != "tool_use":
            raise ValueError("Invalid model tool call")
        block = blocks[0]
        name, args = block["name"], block["input"]
        if name not in self.schemas or self.request.tool_choice == "none" or not isinstance(args, dict):
            raise ValueError(f"Model called undeclared or disabled tool: {name}")
        if isinstance(self.request.tool_choice, dict) and name != self.request.tool_choice["name"]:
            raise ValueError("Model did not honor the selected tool")
        if (self.request.model_extra or {}).get("parallel_tool_calls") is False and any(
            item["type"] != "message" for item in self.output
        ):
            raise ValueError("Model emitted multiple calls with parallel_tool_calls=false")
        is_custom = name in self.custom
        value = args.get("input") if is_custom else json.dumps(args, ensure_ascii=False)
        if not isinstance(value, str):
            raise ValueError("Custom tool input must be a string")
        field = "input" if is_custom else "arguments"
        item = {
            "type": "custom_tool_call" if is_custom else "function_call",
            "id": "fc_" + uuid.uuid4().hex,
            "call_id": "call_" + uuid.uuid4().hex,
            "name": name,
            "status": "completed",
            field: value,
        }
        if "." in name:
            item["namespace"], item["name"] = name.rsplit(".", 1)
        index = len(self.output)
        self.output.append(item)
        events.append(
            self.event(
                "response.output_item.added", output_index=index, item={**item, "status": "in_progress", field: ""}
            )
        )
        prefix = "response.custom_tool_call_input" if is_custom else "response.function_call_arguments"
        events.append(self.event(prefix + ".delta", item_id=item["id"], output_index=index, delta=value))
        events.append(self.event(prefix + ".done", item_id=item["id"], output_index=index, **{field: value}))
        events.append(self.event("response.output_item.done", output_index=index, item=item))
        return events

    def finish(self, input_tokens, output_tokens, incomplete=False):
        events = self.close_text()
        if (
            not incomplete
            and (self.request.tool_choice == "required" or isinstance(self.request.tool_choice, dict))
            and not any(item["type"] != "message" for item in self.output)
        ):
            raise ValueError("Model did not honor tool_choice=required")
        self.response.update(
            status="incomplete" if incomplete else "completed",
            incomplete_details={"reason": "max_output_tokens"} if incomplete else None,
            output=self.output,
            usage={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        )
        events.append(self.event("response.incomplete" if incomplete else "response.completed", response=self.response))
        return events

    def fail(self, exc):
        self.response.update(status="failed", output=self.output, error={"code": "server_error", "message": str(exc)})
        return self.event("response.failed", response=self.response)


def register_responses(app, server):
    locks = {}

    @app.post("/v1/responses")
    async def responses(request: ResponsesRequest):
        try:
            messages = responses_messages(request)
            tools, schemas, custom, choice = responses_tools(request)
        except (ValueError, KeyError, TypeError) as exc:
            return JSONResponse(
                status_code=400, content={"error": {"type": "invalid_request_error", "message": str(exc)}}
            )
        if request.model not in server.models:
            raise HTTPException(status_code=404, detail=f"Model '{request.model}' not found")
        await server.maybe_load_model(request.model)
        model = server.models[request.model]
        prompt = model.apply_chat_template(messages, tools=tools or None, tool_choice=choice)
        max_output, max_input = model.get_model_limits()
        budget = request.max_output_tokens or max_output
        if budget > max_output:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "type": "invalid_request_error",
                        "message": f"max_output_tokens exceeds server maximum ({max_output})",
                    }
                },
            )
        input_tokens = model.count_tokens(prompt)
        if max_input > 0 and input_tokens > max_input:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "type": "invalid_request_error",
                        "message": f"Input exceeds context window ({max_input} input tokens)",
                    }
                },
            )
        settings = {"max_length": budget}
        if request.temperature is not None:
            settings["temperature"] = request.temperature
            if request.temperature == 0:
                settings["top_k"] = 1
                settings["temperature"] = 1.0
        if request.top_p is not None:
            settings["top_p"] = request.top_p
        lock = locks.setdefault(request.model, asyncio.Lock())

        async def events():
            state, parser = ResponseEvents(request, schemas, custom), OutputParser()
            for event in state.start():
                yield event
            # Serialize this endpoint's requests; a disconnected client must not
            # release the engine while its worker is still generating.
            async with lock:
                loop = asyncio.get_running_loop()
                queue = asyncio.Queue()
                cancelled = threading.Event()
                generation_stats = {}

                def produce():
                    try:
                        for chunk in model.engine.infer_list_stream(
                            prompt, settings=settings, generation_stats=generation_stats
                        ):
                            if not cancelled.is_set():
                                loop.call_soon_threadsafe(queue.put_nowait, chunk)
                    except Exception as exc:
                        loop.call_soon_threadsafe(queue.put_nowait, exc)
                    finally:
                        loop.call_soon_threadsafe(queue.put_nowait, None)

                worker = loop.run_in_executor(None, produce)
                raw = []
                try:
                    while True:
                        chunk = await queue.get()
                        if chunk is None:
                            break
                        if isinstance(chunk, Exception):
                            raise chunk
                        raw.append(chunk)
                        for kind, value in parser.feed(chunk):
                            for event in state.accept(kind, value):
                                yield event
                    for kind, value in parser.feed("", final=True):
                        for event in state.accept(kind, value):
                            yield event
                    output_tokens = generation_stats.get("output_tokens", model.count_tokens("".join(raw)))
                    incomplete = not generation_stats.get("stopped_on_eos", False) and output_tokens >= budget
                    for event in state.finish(input_tokens, output_tokens, incomplete=incomplete):
                        yield event
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
                except Exception as exc:
                    yield state.fail(exc)
                finally:
                    cancelled.set()
                    await asyncio.shield(worker)

        if request.stream:

            async def sse():
                async for event in events():
                    yield f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"

            return StreamingResponse(
                sse(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
            )
        final = None
        async for event in events():
            if event["type"] in ("response.completed", "response.incomplete", "response.failed"):
                final = event["response"]
        return JSONResponse(content=final, status_code=500 if final["status"] == "failed" else 200)
