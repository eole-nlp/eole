"""OpenAI Chat Completions request types, conversion, and routes."""

import json
import time
import uuid
from typing import Any, List, Union, Optional, Literal

from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from eole.utils.logging import logger
from eole.server.streaming import inference_stream
from eole.server.tool_parsing import _coerce_tool_inputs_from_schema, _parse_anthropic_response_content
from eole.server.utils import _log_json_payload, estimate_tokens
from eole.constants import DefaultTokens


class OpenAIFunctionCall(BaseModel):
    name: str
    arguments: Union[str, dict] = "{}"


class OpenAIToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: OpenAIFunctionCall


class OpenAIMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "developer", "user", "assistant", "tool"]
    # Per the OpenAI API spec, content can be a string, an array of content
    # parts (multimodal), or null (when the message only carries tool_calls).
    content: Optional[Union[str, List[Any]]] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[OpenAIToolCall]] = None
    reasoning_content: Optional[str] = None


class OpenAIChatRequest(BaseModel):
    # Silently drop any OpenAI-compatible fields not explicitly declared here
    # (e.g. top_k, response_format, seed) so that
    # clients that send them don't receive a 422.
    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "example": {
                "model": "llama3-8b-instruct",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"},
                ],
                "temperature": 0.7,
                "max_tokens": 100,
            }
        },
    )

    model: str
    messages: List[OpenAIMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 0.95
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[dict] = None
    user: Optional[str] = None
    tools: Optional[List[dict]] = None
    tool_choice: Optional[Union[str, dict]] = None
    enable_thinking: Optional[bool] = None
    reasoning_effort: Optional[str] = None
    chat_template_kwargs: Optional[dict] = None


class OpenAIUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAIChoice(BaseModel):
    index: int
    message: OpenAIMessage
    finish_reason: Literal["stop", "length", "content_filter", "null", "tool_calls"]


class OpenAIChatResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[OpenAIChoice]
    usage: OpenAIUsage

    class Config:
        json_schema_extra = {
            "example": {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "llama3-8b-instruct",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello! How can I assist you today?"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            }
        }


class OpenAIStreamDelta(BaseModel):
    """Delta object inside a streaming chunk choice."""

    role: Optional[Literal["assistant"]] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[dict]] = None


class OpenAIStreamChoice(BaseModel):
    """Choice object inside a streaming chunk."""

    index: int
    delta: OpenAIStreamDelta
    finish_reason: Optional[Literal["stop", "length", "tool_calls"]] = None


class OpenAIStreamChunk(BaseModel):
    """A single Server-Sent Events chunk in OpenAI streaming format."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: List[OpenAIStreamChoice]


def _parse_openai_response_content(text: str, tools=None):
    """Convert model-emitted tool XML into OpenAI message fields."""
    if not tools:
        return text or None, [], "stop"
    content_blocks, _ = _parse_anthropic_response_content(text)
    _coerce_tool_inputs_from_schema(content_blocks, tools)
    allowed_names = {tool.get("function", tool).get("name") for tool in tools}

    text_parts = []
    tool_calls = []
    for block in content_blocks:
        if block.get("type") == "text":
            if block.get("text", "").strip():
                text_parts.append(block["text"])
        elif block.get("type") == "tool_use" and block.get("name") in allowed_names:
            tool_id = block.get("id", f"toolu_{uuid.uuid4().hex[:8]}")
            if tool_id.startswith("toolu_"):
                tool_id = "call_" + tool_id.removeprefix("toolu_")
            tool_calls.append(
                OpenAIToolCall(
                    id=tool_id,
                    function=OpenAIFunctionCall(
                        name=block["name"],
                        arguments=json.dumps(block.get("input", {}), ensure_ascii=False, separators=(",", ":")),
                    ),
                )
            )
        elif block.get("type") == "tool_use":
            text_parts.append(f"Tool call to unavailable function '{block.get('name', '')}'.")

    content = "\n".join(text_parts) if text_parts else None
    return content, tool_calls, "tool_calls" if tool_calls else "stop"


class _OpenAIStreamParser:
    """Split model text, reasoning, and complete tool blocks incrementally."""

    def __init__(self, parse_tools=False, thinking=False):
        self.pending = ""
        self.thinking = thinking
        self.parse_tools = parse_tools
        self.tool = ""
        self.tool_end = None
        self.tool_was_thinking = False

    def feed(self, chunk: str, final=False):
        self.pending += chunk
        result = []

        while self.pending:
            if self.tool_end:
                end = self.pending.find(self.tool_end)
                if end < 0:
                    if final:
                        kind = "reasoning" if self.tool_was_thinking else "text"
                        result.append((kind, (self.tool + self.pending).replace(DefaultTokens.SEP, "\n")))
                        self.pending = ""
                        self.tool = ""
                        self.tool_end = None
                        break
                    keep = len(self.tool_end) - 1
                    self.tool += self.pending[:-keep] if keep else self.pending
                    self.pending = self.pending[-keep:] if keep else ""
                    break
                end += len(self.tool_end)
                result.append(("tool", (self.tool + self.pending[:end]).replace(DefaultTokens.SEP, "\n")))
                self.pending = self.pending[end:]
                self.tool = ""
                self.tool_end = None
                continue

            tags = ["</think>" if self.thinking else "<think>", DefaultTokens.SEP]
            if self.parse_tools:
                tags.extend(("<tool_call>", "<tool_use"))
            found = [(self.pending.find(tag), tag) for tag in tags if tag in self.pending]
            if found:
                position, tag = min(found)
                if position:
                    result.append(("reasoning" if self.thinking else "text", self.pending[:position]))
                self.pending = self.pending[position + len(tag) :]
                if tag == "<think>":
                    self.thinking = True
                elif tag == "</think>":
                    self.thinking = False
                elif tag in ("<tool_call>", "<tool_use"):
                    self.tool = tag
                    self.tool_end = "</tool_call>" if tag == "<tool_call>" else "</tool_use>"
                    self.tool_was_thinking = self.thinking
                else:
                    result.append(("reasoning" if self.thinking else "text", "\n"))
                continue

            keep = 0
            if not final:
                max_tag_length = max(map(len, tags))
                for size in range(1, min(len(self.pending), max_tag_length) + 1):
                    if any(tag.startswith(self.pending[-size:]) for tag in tags):
                        keep = size
            text = self.pending[:-keep] if keep else self.pending
            self.pending = self.pending[-keep:] if keep else ""
            if text:
                result.append(("reasoning" if self.thinking else "text", text))
            break
        return result


def _parse_openai_complete_response(text: str, tools=None, thinking=False):
    """Split a complete response while preserving reasoning separately."""
    parser = _OpenAIStreamParser(parse_tools=bool(tools), thinking=thinking)
    content_parts = []
    reasoning_parts = []
    tool_calls = []
    for kind, value in parser.feed(text, final=True):
        if kind == "text":
            content_parts.append(value)
        elif kind == "reasoning":
            reasoning_parts.append(value)
        else:
            tool_content, calls, _ = _parse_openai_response_content(value, tools)
            if tool_content:
                content_parts.append(tool_content)
            tool_calls.extend(calls)
    content = "".join(content_parts).strip() or None
    reasoning = "".join(reasoning_parts).strip() or None
    return content, reasoning, tool_calls, "tool_calls" if tool_calls else "stop"


def _chat_prompt_opens_thinking(chat_input: str) -> bool:
    """Return whether generation begins inside a template-provided think block."""
    return chat_input.rstrip().endswith("<think>")


def _openai_thinking_enabled(request: OpenAIChatRequest) -> bool:
    """Resolve Pi and other OpenAI-compatible thinking controls."""
    if request.enable_thinking is not None:
        return request.enable_thinking
    template_value = (request.chat_template_kwargs or {}).get("enable_thinking")
    if isinstance(template_value, bool):
        return template_value
    return request.reasoning_effort not in (None, "none", "off")


def _openai_reasoning_effort(request: OpenAIChatRequest) -> Optional[str]:
    """Resolve the reasoning effort forwarded to the model chat template."""
    if request.reasoning_effort is not None:
        return request.reasoning_effort
    template_value = (request.chat_template_kwargs or {}).get("reasoning_effort")
    return template_value if isinstance(template_value, str) else None


def _openai_messages_for_template(messages: List[OpenAIMessage]) -> list:
    """Preserve OpenAI tool history and decode function arguments for Jinja."""
    rendered = []
    for message in messages:
        item = message.model_dump(exclude_none=True)
        valid_tool_calls = []
        malformed_tool_calls = []
        for tool_call in item.get("tool_calls", []):
            arguments = tool_call["function"].get("arguments", {})
            if isinstance(arguments, str):
                try:
                    tool_call["function"]["arguments"] = json.loads(arguments)
                except json.JSONDecodeError:
                    malformed_tool_calls.append(
                        "<tool_call>"
                        + json.dumps(
                            {"name": tool_call["function"]["name"], "arguments": arguments},
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "</tool_call>"
                    )
                    continue
            valid_tool_calls.append(tool_call)
        if valid_tool_calls:
            item["tool_calls"] = valid_tool_calls
        else:
            item.pop("tool_calls", None)
        if malformed_tool_calls:
            existing_content = item.get("content")
            malformed_content = "\n".join(malformed_tool_calls)
            item["content"] = (
                f"{existing_content}\n{malformed_content}"
                if isinstance(existing_content, str) and existing_content
                else malformed_content
            )
        rendered.append(item)
    return rendered


def _prepare_openai_tool_request(messages: list, tools, tool_choice):
    """Apply OpenAI tool-choice semantics to tools and rendered messages."""
    if isinstance(tool_choice, str) and tool_choice not in ("auto", "none", "required"):
        raise ValueError(f"Invalid tool_choice: {tool_choice}")
    if tool_choice == "none":
        return messages, None
    if not tools:
        if tool_choice == "required" or isinstance(tool_choice, dict):
            raise ValueError("tool_choice requires a non-empty tools list")
        return messages, None

    selected_tools = list(tools)
    required_name = None
    require_call = tool_choice == "required"
    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        if choice_type == "function":
            required_name = tool_choice.get("function", {}).get("name")
        elif choice_type == "tool":
            required_name = tool_choice.get("name")
        else:
            raise ValueError(f"Invalid tool_choice type: {choice_type}")
        if not required_name:
            raise ValueError("Named tool_choice requires a tool name")
        if required_name:
            selected_tools = [
                tool for tool in selected_tools if tool.get("function", tool).get("name") == required_name
            ]
            if not selected_tools:
                raise ValueError(f"Unknown tool requested by tool_choice: {required_name}")
            require_call = True

    if not require_call:
        return messages, selected_tools

    instruction = (
        f"You must call the {required_name} tool to answer this request."
        if required_name
        else "You must call one or more of the provided tools to answer this request."
    )
    messages = [dict(message) for message in messages]
    if messages and messages[0].get("role") == "system":
        content = messages[0].get("content")
        if isinstance(content, list):
            messages[0]["content"] = [*content, {"type": "text", "text": instruction}]
        else:
            messages[0]["content"] = f"{content}\n\n{instruction}" if content else instruction
    else:
        messages.insert(0, {"role": "system", "content": instruction})
    return messages, selected_tools


def map_openai_to_eole_settings(openai_request: OpenAIChatRequest) -> dict:
    """
    Map OpenAI parameters to Eole settings.
    """
    settings = {}

    if openai_request.temperature is not None:
        settings["temperature"] = openai_request.temperature

    if openai_request.top_p is not None:
        settings["top_p"] = openai_request.top_p

    if openai_request.max_tokens is not None:
        settings["max_length"] = openai_request.max_tokens

    if openai_request.stop is not None:
        if isinstance(openai_request.stop, str):
            settings["stop"] = [openai_request.stop]
        else:
            settings["stop"] = openai_request.stop

    # Note: presence_penalty, frequency_penalty, logit_bias
    # may not have direct equivalents in your engine
    # You can add custom mappings if your engine supports similar features

    return settings


def register_openai_chat(app, server):
    @app.post("/v1/chat/completions", response_model=OpenAIChatResponse)
    @app.post("/openai/chat/completions", response_model=OpenAIChatResponse)  # Alternative path
    async def openai_chat(request: OpenAIChatRequest):
        """
        OpenAI-compatible chat completions endpoint.
        This allows the server to be used as a drop-in replacement
        for OpenAI or other LLM APIs.
        """
        try:
            # _log_json_payload("INCOMING REQUEST [openai]", request.model_dump())

            # Check if n > 1 (multiple completions not supported in simple implementation)
            if request.n > 1:
                from fastapi.responses import JSONResponse

                return JSONResponse(
                    status_code=400,
                    content={
                        "error": {
                            "message": "Multiple completions (n > 1) not yet supported",
                            "type": "invalid_request_error",
                            "code": "multiple_completions_not_supported",
                        }
                    },
                )

            # Convert OpenAI messages to the format expected by your engine
            messages = _openai_messages_for_template(request.messages)

            # Map OpenAI parameters to Eole settings
            settings = map_openai_to_eole_settings(request)

            # Ensure model is loaded
            model_id = request.model
            if model_id not in server.models:
                from fastapi.responses import JSONResponse

                return JSONResponse(
                    status_code=404,
                    content={
                        "error": {
                            "message": f"Model '{model_id}' not found",
                            "type": "invalid_request_error",
                            "code": "model_not_found",
                        }
                    },
                )

            await server.maybe_load_model(model_id)

            # Forward tools/tool_choice to the chat template if provided.
            template_tools = request.tools or None
            template_tool_choice = request.tool_choice or None
            enable_thinking = _openai_thinking_enabled(request)
            reasoning_effort = _openai_reasoning_effort(request)
            try:
                messages, template_tools = _prepare_openai_tool_request(messages, template_tools, template_tool_choice)
            except ValueError as exc:
                from fastapi.responses import JSONResponse

                return JSONResponse(
                    status_code=400,
                    content={
                        "error": {
                            "message": str(exc),
                            "type": "invalid_request_error",
                            "code": "invalid_tool_choice",
                        }
                    },
                )

            # ----------------------------------------------------------------
            # Streaming path
            # ----------------------------------------------------------------
            if request.stream:
                model_obj = server.models[model_id]
                if not model_obj.loaded:
                    model_obj.load()

                chat_input = model_obj.apply_chat_template(
                    messages,
                    tools=template_tools,
                    tool_choice=template_tool_choice,
                    enable_thinking=enable_thinking,
                    reasoning_effort=reasoning_effort,
                )
                prompt_opens_thinking = _chat_prompt_opens_thinking(chat_input)
                completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
                created_ts = int(time.time())

                async def _stream_sse():
                    """Async generator that yields SSE-formatted data lines."""
                    async with inference_stream(model_obj.engine, chat_input, settings) as chunks:
                        # First chunk: role announcement
                        first_chunk = OpenAIStreamChunk(
                            id=completion_id,
                            created=created_ts,
                            model=model_id,
                            choices=[
                                OpenAIStreamChoice(
                                    index=0,
                                    delta=OpenAIStreamDelta(role="assistant"),
                                )
                            ],
                        )
                        yield f"data: {first_chunk.model_dump_json()}\n\n"

                        parser = _OpenAIStreamParser(parse_tools=bool(template_tools), thinking=prompt_opens_thinking)
                        raw_chunks: list = []
                        tool_index = 0
                        emitted_tool_call = False

                        def _stream_parts(parts):
                            nonlocal tool_index, emitted_tool_call
                            for kind, value in parts:
                                if kind == "reasoning":
                                    delta = OpenAIStreamDelta(reasoning_content=value)
                                elif kind == "text":
                                    delta = OpenAIStreamDelta(content=value)
                                else:
                                    tool_content, tool_calls, _ = _parse_openai_response_content(value, template_tools)
                                    if tool_content:
                                        content_chunk = OpenAIStreamChunk(
                                            id=completion_id,
                                            created=created_ts,
                                            model=model_id,
                                            choices=[
                                                OpenAIStreamChoice(
                                                    index=0,
                                                    delta=OpenAIStreamDelta(content=tool_content),
                                                )
                                            ],
                                        )
                                        yield f"data: {content_chunk.model_dump_json()}\n\n"
                                    if not tool_calls:
                                        continue
                                    delta = OpenAIStreamDelta(
                                        tool_calls=[
                                            {"index": tool_index + index, **tool_call.model_dump()}
                                            for index, tool_call in enumerate(tool_calls)
                                        ]
                                    )
                                    tool_index += len(tool_calls)
                                    emitted_tool_call = True
                                chunk = OpenAIStreamChunk(
                                    id=completion_id,
                                    created=created_ts,
                                    model=model_id,
                                    choices=[OpenAIStreamChoice(index=0, delta=delta)],
                                )
                                yield f"data: {chunk.model_dump_json()}\n\n"

                        async for item in chunks:
                            raw_chunks.append(item)
                            for event in _stream_parts(parser.feed(item)):
                                yield event
                        for event in _stream_parts(parser.feed("", final=True)):
                            yield event

                        _log_json_payload("MODEL RESPONSE [openai stream]", "".join(raw_chunks))

                        # Final chunk with finish_reason
                        final_chunk = OpenAIStreamChunk(
                            id=completion_id,
                            created=created_ts,
                            model=model_id,
                            choices=[
                                OpenAIStreamChoice(
                                    index=0,
                                    delta=OpenAIStreamDelta(),
                                    finish_reason="tool_calls" if emitted_tool_call else "stop",
                                )
                            ],
                        )
                        final_data = f"data: {final_chunk.model_dump_json()}\n\n"
                        # _log_json_payload("SERVER RESPONSE [openai stream final chunk]", final_chunk.model_dump())
                        yield final_data
                        yield "data: [DONE]\n\n"

                return StreamingResponse(
                    _stream_sse(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                    },
                )

            # ----------------------------------------------------------------
            # Non-streaming path
            # ----------------------------------------------------------------
            model_obj = server.models[model_id]
            chat_input = model_obj.apply_chat_template(
                messages,
                tools=template_tools,
                tool_choice=template_tool_choice,
                enable_thinking=enable_thinking,
                reasoning_effort=reasoning_effort,
            )
            prompt_opens_thinking = _chat_prompt_opens_thinking(chat_input)
            scores, preds = await model_obj.infer_async(
                inputs=chat_input,
                settings=settings,
                is_chat=False,
            )

            # Calculate token usage (rough estimation)
            # content can be None (tool-call-only message) or a list (multipart);
            # normalise to plain text for token counting.
            def _content_as_str(c):
                if c is None:
                    return ""
                if isinstance(c, list):
                    return " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
                return str(c)

            prompt_text = " ".join([_content_as_str(msg.content) for msg in request.messages])
            prompt_tokens = estimate_tokens(prompt_text)
            completion_text = preds[0][0] if preds and preds[0] else ""
            completion_tokens = estimate_tokens(completion_text)
            response_content, reasoning_content, response_tool_calls, finish_reason = _parse_openai_complete_response(
                completion_text, template_tools, thinking=prompt_opens_thinking
            )

            _log_json_payload("MODEL RESPONSE [openai]", completion_text)

            # Build OpenAI-compatible response
            response = OpenAIChatResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
                object="chat.completion",
                created=int(time.time()),
                model=model_id,
                choices=[
                    OpenAIChoice(
                        index=0,
                        message=OpenAIMessage(
                            role="assistant",
                            content=response_content,
                            reasoning_content=reasoning_content,
                            tool_calls=response_tool_calls or None,
                        ),
                        finish_reason=finish_reason,
                    )
                ],
                usage=OpenAIUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )

            # _log_json_payload("SERVER RESPONSE [openai]", response.model_dump())
            return response

        except Exception as e:
            logger.error(f"Error in OpenAI chat endpoint: {e}")
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=500,
                content={"error": {"message": str(e), "type": "internal_error", "code": "internal_error"}},
            )

    # -----------------------------------------------------------------------
    # Anthropic Messages API endpoints
