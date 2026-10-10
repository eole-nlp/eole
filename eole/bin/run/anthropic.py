"""Anthropic Messages request types, conversion, and routes."""

import json
import uuid
from typing import Any, List, Union, Optional, Literal

from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from eole.utils.logging import logger
from eole.bin.run.streaming import inference_stream
from eole.bin.run.tool_parsing import _coerce_tool_inputs_from_schema, _parse_anthropic_response_content
from eole.bin.run.serving_utils import _log_json_payload, estimate_tokens
from eole.bin.run.serving_utils import _post_process_model_output


class AnthropicTextBlock(BaseModel):
    """Text content block used in Anthropic messages."""

    type: Literal["text"] = "text"
    text: str


class AnthropicToolUseBlock(BaseModel):
    """Tool use content block (assistant calling a tool)."""

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict = Field(default_factory=dict)


class AnthropicToolResultBlock(BaseModel):
    """Tool result content block (user providing tool output)."""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Union[str, List[Any]] = ""


class AnthropicTool(BaseModel):
    """Tool definition for the Anthropic API."""

    name: str
    description: Optional[str] = None
    input_schema: dict = Field(default_factory=dict)


class AnthropicToolChoice(BaseModel):
    """Tool choice specification for the Anthropic API."""

    type: Literal["auto", "any", "tool"] = "auto"
    name: Optional[str] = None  # required when type == "tool"


class AnthropicInputMessage(BaseModel):
    """A single message in an Anthropic Messages request."""

    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant"]
    content: Union[str, List[Any]]


class AnthropicMessagesRequest(BaseModel):
    """Anthropic Messages API request."""

    model_config = ConfigDict(extra="ignore")

    model: str
    messages: List[AnthropicInputMessage]
    system: Optional[Union[str, List[Any]]] = None
    max_tokens: int = 1024
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    tools: Optional[List[AnthropicTool]] = None
    tool_choice: Optional[AnthropicToolChoice] = None


class AnthropicUsage(BaseModel):
    """Token usage in Anthropic response format."""

    input_tokens: int
    output_tokens: int


class AnthropicMessagesResponse(BaseModel):
    """Anthropic Messages API response."""

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: List[Any]
    model: str
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage


def _anthropic_messages_to_openai(messages: list, system=None) -> list:
    """
    Convert a list of Anthropic-format messages to OpenAI-style messages
    suitable for ``apply_chat_template``.

    Anthropic content blocks are handled as follows:

    * ``text``        → plain text (concatenated for the same turn)
    * ``tool_use``    → collected into a ``tool_calls`` array on the assistant
                        message, matching the OpenAI / HuggingFace chat-template
                        standard so that tool-aware templates render correctly.
    * ``tool_result`` → a separate ``{"role": "tool", …}`` message so that
                        models with an OpenAI-style tool-result slot receive
                        the data correctly.

    Top-level and message-level system prompts are combined into one leading
    ``{"role": "system", …}`` message.  Claude Code can append system-role
    cache markers mid-conversation, while many Hugging Face chat templates
    require the system message to be first.
    """
    openai_messages: list = []
    system_parts: list[str] = []

    def content_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                item.get("text", "") if isinstance(item, dict) else str(getattr(item, "text", item)) for item in content
            )
        else:
            return str(content)

    if system is not None:
        system_text = content_text(system)
        if system_text:
            system_parts.append(system_text)

    for msg in messages:
        role = msg.role if hasattr(msg, "role") else msg.get("role", "user")
        content = msg.content if hasattr(msg, "content") else msg.get("content", "")

        if role == "system":
            system_text = content_text(content)
            if system_text:
                system_parts.append(system_text)
            continue

        if isinstance(content, str):
            openai_messages.append({"role": role, "content": content})
            continue

        # content is a list of blocks
        text_parts: list = []
        tool_calls: list = []
        pending_tool_results: list = []

        for block in content:
            if isinstance(block, dict):
                btype = block.get("type", "text")
            elif hasattr(block, "type"):
                btype = block.type
                block = block.model_dump() if hasattr(block, "model_dump") else vars(block)
            else:
                btype = "text"
                block = {"text": str(block)}

            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "tool_use":
                tool_id = block.get("id", f"toolu_{uuid.uuid4().hex[:8]}")
                tool_name = block.get("name", "")
                tool_input = block.get("input", {})
                # Use OpenAI tool_calls format so that HF chat templates that
                # understand tool_calls render the assistant turn correctly.
                tool_calls.append(
                    {
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": tool_input,
                        },
                    }
                )
            elif btype == "tool_result":
                tool_use_id = block.get("tool_use_id", "")
                tool_content = block.get("content", "")
                if isinstance(tool_content, list):
                    tool_content = "\n".join(
                        (b.get("text", "") if isinstance(b, dict) else str(b)) for b in tool_content
                    )
                pending_tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_use_id,
                        "content": str(tool_content),
                    }
                )

        # Build the message for this turn.  When the assistant made tool calls
        # the content should be null (or the text prefix if any) and the
        # tool_calls array should be set.
        if tool_calls:
            msg_out: dict = {"role": role}
            # Include any preceding text as content (or null if none)
            msg_out["content"] = "\n".join(text_parts) if text_parts else ""
            msg_out["tool_calls"] = tool_calls
            openai_messages.append(msg_out)
        elif text_parts:
            openai_messages.append({"role": role, "content": "\n".join(text_parts)})
        # If only tool_result blocks were present (user turn), there is no
        # assistant message to add — just the pending tool results below.

        openai_messages.extend(pending_tool_results)

    if system_parts:
        openai_messages.insert(0, {"role": "system", "content": "\n".join(system_parts)})

    return openai_messages


def _anthropic_tools_to_openai(tools: List[AnthropicTool]) -> list:
    """
    Convert a list of Anthropic tool definitions to OpenAI function-tool
    format for use with chat templates that understand OpenAI-style tools.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.input_schema,
            },
        }
        for t in tools
    ]


def _map_anthropic_to_eole_settings(request: AnthropicMessagesRequest) -> dict:
    """Map Anthropic request parameters to Eole inference settings."""
    settings: dict = {}
    if request.temperature is not None:
        settings["temperature"] = request.temperature
    if request.top_p is not None:
        settings["top_p"] = request.top_p
    if request.max_tokens is not None:
        settings["max_length"] = request.max_tokens
    if request.stop_sequences:
        settings["stop"] = request.stop_sequences
    return settings


def register_anthropic(app, server):
    def _anthropic_model_info(model_id: str, model_obj) -> dict:
        """Return a single Anthropic-format model descriptor."""
        max_tokens, max_input_tokens = model_obj.get_model_limits()
        return {
            "id": model_id,
            "type": "model",
            "display_name": model_id,
            "created_at": "1970-01-01T00:00:00Z",
            "max_input_tokens": max_input_tokens,
            "max_tokens": max_tokens,
        }

    @app.get("/v1/models")
    @app.get("/anthropic/v1/models")
    async def anthropic_list_models():
        """
        Anthropic-compatible model listing endpoint.

        Returns the list of models currently configured on this server with
        their context-window limits (``max_input_tokens``) and maximum
        generation budget (``max_tokens``).
        """
        data = []
        for model_id, model_obj in server.models.items():
            if model_obj.loaded:
                data.append(_anthropic_model_info(model_id, model_obj))
        return {"data": data}

    @app.get("/v1/models/{model_id:path}")
    @app.get("/anthropic/v1/models/{model_id:path}")
    async def anthropic_get_model(model_id: str):
        """
        Anthropic-compatible single-model info endpoint.

        Accepts any ``model_id`` string.  If the id is not found in the server
        configuration the first available model is used (mirrors the alias
        resolution in the ``/v1/messages`` endpoint).

        Returns an Anthropic-format model descriptor with ``max_input_tokens``
        and ``max_tokens`` derived from the model's inference configuration.
        """
        from fastapi.responses import JSONResponse as _JSONResponse

        if model_id in server.models:
            resolved_id = model_id
        elif server.models:
            resolved_id = next(iter(server.models))
        else:
            return _JSONResponse(
                status_code=404,
                content={
                    "type": "error",
                    "error": {
                        "type": "not_found_error",
                        "message": f"Model '{model_id}' not found",
                    },
                },
            )
        model_obj = server.models[resolved_id]
        if not model_obj.loaded:
            await server.maybe_load_model(resolved_id)
        return _anthropic_model_info(resolved_id, model_obj)

    @app.post("/v1/messages/count_tokens")
    @app.post("/anthropic/v1/messages/count_tokens")
    async def anthropic_count_tokens(request: AnthropicMessagesRequest):
        """
        Anthropic-compatible token-counting endpoint.

        Applies the chat template and counts the resulting tokens using the
        model's own tokenizer (falling back to a 4-chars-per-token estimate
        when a transform-based tokenizer is not configured).

        Returns::

            {"input_tokens": <int>}
        """
        from fastapi.responses import JSONResponse as _JSONResponse

        model_id = request.model
        if model_id not in server.models:
            if server.models:
                resolved_id = next(iter(server.models))
            else:
                return _JSONResponse(
                    status_code=404,
                    content={
                        "type": "error",
                        "error": {
                            "type": "not_found_error",
                            "message": f"Model '{model_id}' not found",
                        },
                    },
                )
        else:
            resolved_id = model_id

        await server.maybe_load_model(resolved_id)
        model_obj = server.models[resolved_id]
        if not model_obj.loaded:
            model_obj.load()

        openai_messages = _anthropic_messages_to_openai(request.messages, request.system)
        template_tools = None
        template_tool_choice = None
        if request.tools:
            template_tools = _anthropic_tools_to_openai(request.tools)
            template_tool_choice = "auto"
        chat_input = model_obj.apply_chat_template(
            openai_messages,
            tools=template_tools,
            tool_choice=template_tool_choice,
        )
        input_tokens = model_obj.count_tokens(chat_input)
        return {"input_tokens": input_tokens}

    @app.post("/v1/messages", response_model=AnthropicMessagesResponse)
    @app.post("/anthropic/v1/messages", response_model=AnthropicMessagesResponse)
    async def anthropic_messages(request: AnthropicMessagesRequest):
        """
        Anthropic Messages API compatible endpoint.

        Accepts requests in the `Anthropic Messages API
        <https://docs.anthropic.com/en/api/messages>`_ format and returns
        responses in the same format.  Both non-streaming (JSON body) and
        streaming (SSE) modes are supported.

        Tool calling is handled by converting Anthropic tool definitions to
        the OpenAI function-tool format understood by most chat templates,
        and by parsing tool-call blocks from the model output back into
        structured ``tool_use`` content blocks.
        """
        from fastapi.responses import JSONResponse as _JSONResponse

        try:
            # _log_json_payload("INCOMING REQUEST [anthropic]", request.model_dump())

            # ----------------------------------------------------------------
            # Resolve model
            # ----------------------------------------------------------------
            model_id = request.model
            if model_id not in server.models:
                if server.models:
                    # Accept any Claude / Anthropic alias and resolve to the
                    # first configured server model, echoing the alias back in
                    # the response (mirrors llama.cpp behaviour).
                    resolved_id = next(iter(server.models))
                else:
                    return _JSONResponse(
                        status_code=404,
                        content={
                            "type": "error",
                            "error": {
                                "type": "not_found_error",
                                "message": f"Model '{model_id}' not found",
                            },
                        },
                    )
            else:
                resolved_id = model_id

            # ----------------------------------------------------------------
            # Convert Anthropic messages to OpenAI-style for chat template
            # ----------------------------------------------------------------
            openai_messages = _anthropic_messages_to_openai(request.messages, request.system)
            # _log_json_payload("CONVERTED MESSAGES [anthropic→openai]", openai_messages)

            # ----------------------------------------------------------------
            # Convert tools to OpenAI function-tool format (if provided)
            # ----------------------------------------------------------------
            template_tools = None
            template_tool_choice = None
            if request.tools:
                template_tools = _anthropic_tools_to_openai(request.tools)
                # Default to "auto" when tools are present
                template_tool_choice = "auto"
                if request.tool_choice:
                    tc_type = request.tool_choice.type
                    if tc_type == "any":
                        template_tool_choice = "required"
                    elif tc_type == "tool" and request.tool_choice.name:
                        template_tool_choice = {
                            "type": "function",
                            "function": {"name": request.tool_choice.name},
                        }
                    else:
                        template_tool_choice = tc_type

            # ----------------------------------------------------------------
            # Map Anthropic parameters to Eole settings
            # ----------------------------------------------------------------
            settings = _map_anthropic_to_eole_settings(request)

            await server.maybe_load_model(resolved_id)
            model_obj = server.models[resolved_id]
            if not model_obj.loaded:
                model_obj.load()

            completion_id = f"msg_{uuid.uuid4().hex[:24]}"

            # Apply chat template once (shared by both streaming and non-streaming)
            chat_input = model_obj.apply_chat_template(
                openai_messages,
                tools=template_tools,
                tool_choice=template_tool_choice,
            )

            # ----------------------------------------------------------------
            # Guard: reject requests that exceed the model's context window
            # ----------------------------------------------------------------
            _max_tokens, _max_input_tokens = model_obj.get_model_limits()
            if _max_input_tokens > 0:
                input_token_count = model_obj.count_tokens(chat_input)
                if input_token_count > _max_input_tokens:
                    return _JSONResponse(
                        status_code=400,
                        content={
                            "type": "error",
                            "error": {
                                "type": "invalid_request_error",
                                "message": (
                                    f"Input length ({input_token_count} tokens) exceeds the model's "
                                    f"maximum context window ({_max_input_tokens} input tokens). "
                                    "Please reduce the length of the messages."
                                ),
                            },
                        },
                    )

            # ----------------------------------------------------------------
            # Streaming path
            # ----------------------------------------------------------------
            if request.stream:

                async def _stream_anthropic_sse():
                    """
                    Yield Anthropic SSE events.

                    Event sequence (mirrors the real Claude API):
                      message_start → content_block_start → ping →
                      content_block_delta* → content_block_stop →
                      message_delta → message_stop

                    The model output is buffered in full before emitting any
                    SSE events so that we can determine the correct content
                    block type (``text`` vs ``tool_use``) and emit properly
                    typed events.  This is required for Claude Code and other
                    Anthropic-API clients that rely on block-type semantics.
                    """
                    async with inference_stream(model_obj.engine, chat_input, settings) as chunks:
                        # Buffer all model output so we can classify the response
                        # type before emitting SSE events.
                        raw_chunks: list = []
                        async for item in chunks:
                            raw_chunks.append(item)

                        raw_text = "".join(raw_chunks)
                        _log_json_payload("MODEL RESPONSE [anthropic stream raw]", raw_text)
                        full_text = _post_process_model_output(raw_text)
                        # _log_json_payload("MODEL RESPONSE [anthropic stream]", full_text)

                        content_blocks, stop_reason = _parse_anthropic_response_content(full_text)
                        _coerce_tool_inputs_from_schema(content_blocks, request.tools)
                        output_tokens = estimate_tokens(full_text)

                        # Log the parsed content blocks so the user can see exactly
                        # what will be streamed to the client (tool_use id/name/input
                        # and whether the tool_call → tool_use conversion worked).
                        _log_json_payload(
                            "SERVER RESPONSE [anthropic stream content]",
                            {"stop_reason": stop_reason, "content_blocks": content_blocks},
                        )

                        # message_start
                        msg_start = {
                            "type": "message_start",
                            "message": {
                                "id": completion_id,
                                "type": "message",
                                "role": "assistant",
                                "content": [],
                                "model": model_id,
                                "stop_reason": None,
                                "stop_sequence": None,
                                "usage": {"input_tokens": 0, "output_tokens": 0},
                            },
                        }
                        # _log_json_payload("SERVER RESPONSE [anthropic stream start]", msg_start)
                        yield f"event: message_start\ndata: {json.dumps(msg_start)}\n\n"

                        # keepalive ping
                        yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"

                        # Emit one content block per parsed block, with correct types.
                        # For tool calls the Anthropic spec requires:
                        #   content_block_start: {type: "tool_use", id, name, input:{}}
                        #   content_block_delta: {type: "input_json_delta", partial_json: "…"}
                        # For plain text:
                        #   content_block_start: {type: "text", text: ""}
                        #   content_block_delta: {type: "text_delta", text: "…"}
                        for idx, block in enumerate(content_blocks):
                            if block["type"] == "tool_use":
                                cb_start = {
                                    "type": "content_block_start",
                                    "index": idx,
                                    "content_block": {
                                        "type": "tool_use",
                                        "id": block["id"],
                                        "name": block["name"],
                                        "input": {},
                                    },
                                }
                                yield f"event: content_block_start\ndata: {json.dumps(cb_start)}\n\n"
                                input_json = json.dumps(block["input"], ensure_ascii=False)
                                delta = {
                                    "type": "content_block_delta",
                                    "index": idx,
                                    "delta": {"type": "input_json_delta", "partial_json": input_json},
                                }
                                yield f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n"
                            else:
                                cb_start = {
                                    "type": "content_block_start",
                                    "index": idx,
                                    "content_block": {"type": "text", "text": ""},
                                }
                                yield f"event: content_block_start\ndata: {json.dumps(cb_start)}\n\n"
                                delta = {
                                    "type": "content_block_delta",
                                    "index": idx,
                                    "delta": {"type": "text_delta", "text": block.get("text", "")},
                                }
                                yield f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n"

                            yield (
                                f"event: content_block_stop\n"
                                f"data: {json.dumps({'type': 'content_block_stop', 'index': idx})}\n\n"
                            )

                        # message_delta
                        msg_delta = {
                            "type": "message_delta",
                            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                            "usage": {"output_tokens": output_tokens},
                        }
                        # _log_json_payload("SERVER RESPONSE [anthropic stream final]", msg_delta)
                        yield f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n"

                        # message_stop
                        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

                return StreamingResponse(
                    _stream_anthropic_sse(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )

            # ----------------------------------------------------------------
            # Non-streaming path
            # ----------------------------------------------------------------
            scores, preds = await model_obj.infer_async(
                inputs=chat_input,
                settings=settings,
                is_chat=False,
            )
            raw_text = preds[0][0] if preds and preds[0] else ""
            _log_json_payload("MODEL RESPONSE [anthropic raw]", raw_text)
            raw_text = _post_process_model_output(raw_text)

            # _log_json_payload("MODEL RESPONSE [anthropic]", raw_text)

            content_blocks, stop_reason = _parse_anthropic_response_content(raw_text)
            _coerce_tool_inputs_from_schema(content_blocks, request.tools)

            # Rough token estimation
            prompt_text = " ".join(
                (m.get("content", "") if isinstance(m, dict) else str(m.get("content", ""))) for m in openai_messages
            )
            input_tokens = estimate_tokens(prompt_text)
            output_tokens = estimate_tokens(raw_text)

            response = AnthropicMessagesResponse(
                id=completion_id,
                type="message",
                role="assistant",
                content=content_blocks,
                model=model_id,
                stop_reason=stop_reason,
                stop_sequence=None,
                usage=AnthropicUsage(input_tokens=input_tokens, output_tokens=output_tokens),
            )
            # _log_json_payload("SERVER RESPONSE [anthropic]", response.model_dump())
            return response

        except Exception as e:
            logger.error(f"Error in Anthropic messages endpoint: {e}")
            from fastapi.responses import JSONResponse as _JSONResponse2

            return _JSONResponse2(
                status_code=500,
                content={
                    "type": "error",
                    "error": {"type": "api_error", "message": str(e)},
                },
            )
