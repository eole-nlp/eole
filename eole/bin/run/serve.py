#!/usr/bin/env python

import datetime
import os
import re
import time
import gc
import yaml
import json
import uuid
from typing import Any, List, Union, Optional, Literal

import torch
import uvicorn

from fastapi import FastAPI, Request, Body
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

import asyncio
from dataclasses import dataclass

import eole
from eole.inference_engine import InferenceEnginePY
from eole.config.run import PredictConfig
from eole.config.inference import DecodingConfig
from eole.bin import register_bin, BaseBin
from eole.utils.logging import logger
from eole.constants import DefaultTokens

STATUS_OK = "ok"
STATUS_ERROR = "error"


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

_LOG_SEP = "=" * 72


def _log_json_payload(label: str, payload) -> None:
    """Log a labelled JSON payload to the server console.

    *payload* may be a dict/list (serialised to pretty-printed JSON) or a
    plain string (logged as-is).  The output is clearly delimited so it is
    easy to find in the server window.
    """
    if isinstance(payload, (dict, list)):
        body = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    else:
        body = str(payload)
    logger.info(f"\n{_LOG_SEP}\n{label}\n{_LOG_SEP}\n{body}\n{_LOG_SEP}")


# ---------------------------------------------------------------------------
# Model-output post-processing
# ---------------------------------------------------------------------------

# Compiled regex to strip <think>…</think> blocks that Qwen3 and similar
# "thinking" models produce when chain-of-thought reasoning is enabled.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _post_process_model_output(text: str) -> str:
    """Normalise raw model output before returning it to clients.

    1. Replace the eole internal newline sentinel (``｟newline｠``,
       i.e. ``DefaultTokens.SEP``) with actual ``\\n`` characters so that
       the text is readable in log output and in the API response.
    2. Strip ``<think>…</think>`` blocks emitted by Qwen3 and other models
       that run explicit chain-of-thought reasoning.  The reasoning content
       must not be forwarded to the API client, **but** any ``<tool_call>``
       or ``<tool_use>`` blocks nested inside a ``<think>`` block are
       *rescued* first so that tool calls are never silently discarded.
       Some thinking-mode models (e.g. Qwen3) place the tool call inside
       the ``<think>`` block rather than after it.
    """
    # Replace internal newline sentinel with real newline
    text = text.replace(DefaultTokens.SEP, "\n")

    # Strip <think>…</think> blocks, but rescue any embedded tool calls so
    # they are appended to the output after the reasoning is removed.
    rescued: list = []

    def _strip_think(m: re.Match) -> str:
        inner = m.group(0)
        for tc in _TOOL_CALL_SPLIT_RE.findall(inner):
            rescued.append(tc)
        for tu in _TOOL_USE_SPLIT_RE.findall(inner):
            rescued.append(tu)
        return ""

    text = _THINK_BLOCK_RE.sub(_strip_think, text)
    text = text.strip()
    if rescued:
        extra = "\n".join(rescued)
        text = f"{text}\n{extra}".strip() if text else extra
    return text


class TextRequest(DecodingConfig):
    """
    Standard text "completion" request
    (as well as encoder/decoder models e.g. translation).
    """

    model: int | str = Field(description="Model identifier from server configuration.")
    inputs: Union[str, List[str]] = Field(
        description="List of inputs to run inference on. "
        "A single string will be automatically cast to a single item list."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "model": "llama3-8b-instruct",
                "inputs": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a funny guy.<|eot_id|><|start_header_id|>user<|end_header_id|>Tell me a joke :)<|eot_id|><|start_header_id|>assistant<|end_header_id|>",  # noqa: E501
            }
        }


class TextResponse(BaseModel):
    """
    Response of TextRequest.
    """

    predictions: List[List[str]] = Field(description="List of prediction(s) for each input(s).")
    scores: List[List[float]] = Field(description="Pred scores from the model for each prediction.")

    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    [
                        "\n\nHere's one:\n\nWhy couldn't the bicycle stand up by itself?\n\n(wait for it...)\n\nBecause it was two-tired!\n\nHope that made you laugh!"  # noqa: E501
                    ]
                ],
                "scores": [[-0.040771484375]],
            }
        }

    @model_validator(mode="after")
    def _validate_response(self):
        """
        Automatically apply some formatting to the provided text response.
        This logic might be moved elsewhere at some point.
        """
        self.predictions = [[pred.replace(DefaultTokens.SEP, "\n") for pred in preds] for preds in self.predictions]
        return self


class ChatRequest(DecodingConfig):
    """
    Request format for chat-based interactions.
    """

    model: int | str = Field(description="Model identifier from server configuration.")
    messages: List[dict] = Field(description="List of message dictionaries with 'role' and 'content' keys.")

    class Config:
        json_schema_extra = {
            "example": {
                "model": "llama3-8b-instruct",
                "messages": [
                    {"role": "system", "content": "You are a funny guy."},
                    {"role": "user", "content": "Tell me a joke :)"},
                ],
            }
        }


# TODO: specific response model for chat mode?
# class ChatResponse(BaseModel):
#     choices: List[dict]


class OpenAIMessage(BaseModel):
    # Allow any extra fields (e.g. name, tool_call_id, tool_calls) sent by
    # OpenAI-compatible clients without triggering a 422.
    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "user", "assistant", "tool"]
    # Per the OpenAI API spec, content can be a string, an array of content
    # parts (multimodal), or null (when the message only carries tool_calls).
    content: Optional[Union[str, List[Any]]] = None


class OpenAIChatRequest(BaseModel):
    # Silently drop any OpenAI-compatible fields not explicitly declared here
    # (e.g. top_k, tools, tool_choice, response_format, seed, …) so that
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


class OpenAIUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAIChoice(BaseModel):
    index: int
    message: OpenAIMessage
    finish_reason: Literal["stop", "length", "content_filter", "null"]


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


# ---------------------------------------------------------------------------
# SSE streaming models (OpenAI chat.completion.chunk format)
# ---------------------------------------------------------------------------


class OpenAIStreamDelta(BaseModel):
    """Delta object inside a streaming chunk choice."""

    role: Optional[Literal["assistant"]] = None
    content: Optional[str] = None


class OpenAIStreamChoice(BaseModel):
    """Choice object inside a streaming chunk."""

    index: int
    delta: OpenAIStreamDelta
    finish_reason: Optional[Literal["stop", "length"]] = None


class OpenAIStreamChunk(BaseModel):
    """A single Server-Sent Events chunk in OpenAI streaming format."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: List[OpenAIStreamChoice]


# ---------------------------------------------------------------------------
# Anthropic Messages API models
# ---------------------------------------------------------------------------


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

    role: Literal["user", "assistant"]
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


# ---------------------------------------------------------------------------
# Anthropic helper functions
# ---------------------------------------------------------------------------

_TOOL_USE_SPLIT_RE = re.compile(r"(<tool_use[^>]*>.*?</tool_use>)", re.DOTALL)
_TOOL_USE_TAG_RE = re.compile(r"<tool_use([^>]*)>(.*?)</tool_use>", re.DOTALL)
_ATTR_ID_RE = re.compile(r'\bid="([^"]*)"')
_ATTR_NAME_RE = re.compile(r'\bname="([^"]*)"')

# <tool_call> format — used by Hermes/NousResearch and many open-source models
_TOOL_CALL_SPLIT_RE = re.compile(r"(<tool_call>.*?</tool_call>)", re.DOTALL)
_TOOL_CALL_BODY_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

# <function=NAME>…</function=NAME> / <parameter=K>V</parameter=K> format
# used by Claude Code system prompts and some open-source models.
# Both named closing tags (</function=bash>) and unnamed (</function>) are
# supported — the alternation (?:…=\1>|…>) handles both.
_FUNC_CALL_RE = re.compile(r"<function=([^>]+)>(.*?)(?:</function=\1>|</function>)", re.DOTALL)
_PARAM_BLOCK_RE = re.compile(r"<parameter=([^>]+)>(.*?)(?:</parameter=\1>|</parameter>)", re.DOTALL)


def _parse_function_xml_format(body: str):
    """Parse the ``<function=NAME><parameter=K>V</parameter=K></function=NAME>``
    XML format used by Claude Code system prompts and some open-source models.

    Returns ``(tool_id, tool_name, tool_input)`` or ``None`` if the body
    does not match this format.

    The parser handles the malformed-nesting case where the model wraps
    sibling ``<parameter>`` blocks inside each other; inner parameters are
    recursively extracted and the outer value is truncated at the first nested
    ``<parameter=`` marker so only the actual value is used.
    """
    m = _FUNC_CALL_RE.search(body)
    if not m:
        return None

    func_name = m.group(1).strip()
    func_body = m.group(2)

    params: dict = {}

    def _collect(text: str) -> None:
        for pm in _PARAM_BLOCK_RE.finditer(text):
            pname = pm.group(1).strip()
            pval_raw = pm.group(2)
            # Recurse first so nested params are captured under their own names
            _collect(pval_raw)
            # Value is the text before any nested <parameter= tag so we don't
            # include sibling parameters that the model mis-nested inside us.
            pval = pval_raw.split("<parameter=")[0].strip()
            if pname and pname not in params:
                params[pname] = pval

    _collect(func_body)

    # If no parameters with proper closing tags were found, fall back to an
    # open-ended split that stops at the next parameter/function tag boundary.
    # This handles models that emit parameters without closing tags at all.
    if not params:
        for pm in re.finditer(
            r"<parameter=([^>]+)>(.*?)(?=</?(?:parameter|function)(?:=|>)|$)",
            func_body,
            re.DOTALL,
        ):
            pname = pm.group(1).strip()
            pval = pm.group(2).strip()
            if pname and pname not in params:
                params[pname] = pval

    tool_id = f"toolu_{uuid.uuid4().hex[:8]}"
    return tool_id, func_name, params


def _parse_tool_call_block(body: str) -> tuple:
    """Parse the body of a ``<tool_call>`` block.

    Returns ``(tool_id, tool_name, tool_input)`` where *tool_id* is a
    generated ID, *tool_name* and *tool_input* are parsed from the content.

    Supports two body formats in order of preference:

    1. JSON ``{"name": "fn_name", "arguments": {...}}`` — standard format
       used by Hermes/NousResearch, Qwen, Command-R, etc.
    2. ``<function=NAME><parameter=K>V</parameter=K></function=NAME>`` XML
       format used by Claude Code system prompts and some open-source models.

    Falls back to ``{"raw": body}`` with an empty name when neither format
    is recognised.
    """
    body = body.strip()
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        # Try the <function=NAME>…</function=NAME> XML format before giving up.
        xml_result = _parse_function_xml_format(body)
        if xml_result is not None:
            return xml_result
        return f"toolu_{uuid.uuid4().hex[:8]}", "", {"raw": body}

    tool_name = data.get("name", data.get("function", ""))
    # Accept both "arguments" (OpenAI) and "parameters" (some models)
    tool_input = data.get("arguments", data.get("parameters", data))
    if isinstance(tool_input, str):
        try:
            tool_input = json.loads(tool_input)
        except json.JSONDecodeError:
            pass
    tool_id = f"toolu_{uuid.uuid4().hex[:8]}"
    return tool_id, tool_name, tool_input


def _parse_anthropic_response_content(text: str):
    """
    Parse model output text into a list of Anthropic content blocks.

    Handles three common tool-call formats emitted by open-source models:

    1. ``<tool_use id="…" name="…">{json}</tool_use>`` — our round-trip
       format when converting Anthropic → OpenAI messages for the template.
    2. ``<tool_call>{"name": "…", "arguments": {…}}</tool_call>`` — the
       standard format used by Hermes/NousResearch, Qwen, Command-R and many
       other open-source fine-tuned models.
    3. ``<tool_call><function=NAME><parameter=K>V</parameter=K></function=NAME></tool_call>``
       — the XML parameter format used by Claude Code system prompts when the
       model is instructed to reply in that style.

    Plain text around the tags becomes ``text`` blocks.

    Returns ``(content_blocks, stop_reason)`` where *stop_reason* is
    ``"tool_use"`` when at least one tool call was found, otherwise
    ``"end_turn"``.
    """
    blocks = []
    stop_reason = "end_turn"

    # ---------------------------------------------------------------
    # Prefer <tool_call> if it appears in the text (most common for
    # open-source tool-capable models).
    # ---------------------------------------------------------------
    if "<tool_call>" in text:
        parts = _TOOL_CALL_SPLIT_RE.split(text)
        for part in parts:
            if not part:
                continue
            m = _TOOL_CALL_BODY_RE.match(part)
            if m:
                tool_id, tool_name, tool_input = _parse_tool_call_block(m.group(1))
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": tool_name,
                        "input": tool_input,
                    }
                )
                stop_reason = "tool_use"
            else:
                stripped = part.strip()
                if stripped:
                    blocks.append({"type": "text", "text": stripped})
        if blocks:
            return blocks, stop_reason

    # ---------------------------------------------------------------
    # Fall back to <tool_use …>…</tool_use> format.
    # ---------------------------------------------------------------
    parts = _TOOL_USE_SPLIT_RE.split(text)
    for part in parts:
        if not part:
            continue
        tag_match = _TOOL_USE_TAG_RE.match(part)
        if tag_match:
            attrs, body = tag_match.group(1), tag_match.group(2).strip()
            id_m = _ATTR_ID_RE.search(attrs)
            name_m = _ATTR_NAME_RE.search(attrs)
            tool_id = id_m.group(1) if id_m else f"toolu_{uuid.uuid4().hex[:8]}"
            tool_name = name_m.group(1) if name_m else ""
            try:
                tool_input = json.loads(body)
            except json.JSONDecodeError:
                tool_input = {"raw": body}
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": tool_name,
                    "input": tool_input,
                }
            )
            stop_reason = "tool_use"
        else:
            stripped = part.strip()
            if stripped:
                blocks.append({"type": "text", "text": stripped})

    if not blocks:
        blocks = [{"type": "text", "text": text}]
    return blocks, stop_reason


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

    A top-level *system* prompt (string or list of text blocks) is prepended
    as a ``{"role": "system", …}`` message.
    """
    openai_messages: list = []

    if system is not None:
        if isinstance(system, str):
            system_text = system
        elif isinstance(system, list):
            system_text = "\n".join((b.get("text", "") if isinstance(b, dict) else str(b)) for b in system)
        else:
            system_text = str(system)
        openai_messages.append({"role": "system", "content": system_text})

    for msg in messages:
        role = msg.role if hasattr(msg, "role") else msg.get("role", "user")
        content = msg.content if hasattr(msg, "content") else msg.get("content", "")

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


def estimate_tokens(text: str) -> int:
    """
    Rough token estimation (about 4 chars per token).
    For production, use a proper tokenizer.
    """
    return len(text) // 4


class Server(object):
    """
    Main server class to manage configuration, models and corresponding constraints.
    """

    def __init__(self):
        self.start_time = time.time()
        self.models = {}
        self.models_root = None

    def start(self, server_config_path):
        """
        Initialize the server with the given configuration.
        """
        with open(server_config_path) as f:
            server_config = yaml.safe_load(f)
        self.models_root = server_config["models_root"]
        for model in server_config["models"]:
            # instantiate models
            model_id = model["id"]
            model_path = model["path"]
            self.models[model_id] = Model(
                model_id=model_id,
                model_path=model_path,
                models_root=self.models_root,
                model_type=model.get("model_type", "default"),
                pre_config=model.get("config", {}),
            )
            if model.get("preload", False):
                self.models[model_id].load()

    def available_models(self):
        """
        Return a list of available models.
        """
        models = []
        for model_id, model in self.models.items():
            models.append({"id": model_id})
        return models

    async def maybe_load_model(self, model_id_to_load):
        """
        Very naive method to ensure a single model is loaded for now.
        """
        for model_id, model in self.models.items():
            if model_id != model_id_to_load:
                model.unload()


@dataclass
class QueuedRequest:
    inputs: any
    settings: dict
    is_chat: bool
    future: asyncio.Future
    timestamp: float


class Model(object):
    """
    Represents a single model in the server.
    """

    def __init__(
        self,
        model_id=None,
        model_path=None,
        preload=False,
        models_root=None,
        model_type=False,
        pre_config={},
    ):
        self.loaded = False
        self.engine = None
        self.model_id = model_id
        self.preload = preload
        self.models_root = models_root
        self.model_path = model_path
        self.local_path = None
        self.model_type = model_type
        self.pre_config = pre_config
        self.request_queue = asyncio.Queue()
        self.batch_size = pre_config.get("batch_size", 8)
        self.batch_timeout = pre_config.get("batch_timeout", 0.1)
        self.processing_task = None

    def get_config(self):
        """
        Instanciate the configuration for the model.
        """
        # transforms and inference settings are retrieved from the model config for now
        self.config = PredictConfig(
            src="dummy",
            model_path=self.local_path,
            # TODO improve this
            gpu_ranks=[0],
            world_size=1,
            **self.pre_config,
        )

    async def _process_batch(self, batch):
        """
        Process a batch of requests together.
        """
        try:
            # Helper function to make settings hashable via JSON
            def settings_key(settings, is_chat):
                """Create a hashable key from settings."""
                return (json.dumps(settings, sort_keys=True), is_chat)

            # Group by settings and is_chat flag (only batch compatible requests)
            groups = {}
            for req in batch:
                key = settings_key(req.settings, req.is_chat)
                if key not in groups:
                    groups[key] = []
                groups[key].append(req)

            # Process each group
            for (settings_json, is_chat), reqs in groups.items():
                settings = json.loads(settings_json)

                # Collect all inputs
                all_inputs = []
                request_boundaries = []  # Track (start_idx, end_idx) for each request

                for req in reqs:
                    start_idx = len(all_inputs)

                    if is_chat:
                        # Chat mode: single input per request
                        all_inputs.append(self.apply_chat_template(req.inputs))
                        end_idx = len(all_inputs)
                    elif isinstance(req.inputs, str):
                        # Single string input
                        all_inputs.append(req.inputs)
                        end_idx = len(all_inputs)
                    elif isinstance(req.inputs, list):
                        # Multiple inputs in a single request
                        all_inputs.extend(req.inputs)
                        end_idx = len(all_inputs)
                    else:
                        # Fallback: treat as single input
                        all_inputs.append(req.inputs)
                        end_idx = len(all_inputs)

                    request_boundaries.append((start_idx, end_idx))

                # Run batched inference in thread pool to avoid blocking event loop
                scores, _, preds = await asyncio.get_event_loop().run_in_executor(
                    self.engine._thread_pool,  # ← warmed thread
                    lambda s=settings: self.engine.infer_list(all_inputs, settings=s),
                )

                # Distribute results back to individual requests using boundaries
                for req, (start_idx, end_idx) in zip(reqs, request_boundaries):
                    req_scores = scores[start_idx:end_idx]
                    req_preds = preds[start_idx:end_idx]
                    req.future.set_result((req_scores, req_preds))

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Set exception for all requests in batch
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)

    async def batch_processor(self):
        """
        Continuously collect and process requests in batches.
        """
        while True:
            batch = []
            deadline = None

            # Get first request (blocking)
            try:
                req = await self.request_queue.get()
                batch.append(req)
                deadline = time.time() + self.batch_timeout
            except Exception:
                continue

            # Collect more requests until timeout or batch full
            while len(batch) < self.batch_size and time.time() < deadline:
                try:
                    timeout = max(0, deadline - time.time())
                    req = await asyncio.wait_for(self.request_queue.get(), timeout=timeout)
                    batch.append(req)
                except asyncio.TimeoutError:
                    break

            # Process batch
            await self._process_batch(batch)

    def maybe_retrieve_model(self):
        """
        Download the model if it's not available locally.
        """
        from huggingface_hub import HfApi, snapshot_download

        hf_api = HfApi()
        try:
            hf_api.model_info(self.model_path)
        except Exception:
            self.local_path = os.path.expandvars(self.model_path)
        else:
            self.local_path = os.path.expandvars(os.path.join(self.models_root, self.model_path))
            logger.info(f"Downloading {self.model_path} from huggingface, " f"to local directory {self.local_path}")
            snapshot_download(repo_id=self.model_path, local_dir=self.local_path)

    async def ensure_batch_processor(self):
        """
        Start batch processor if not already running.
        Must be called from async context.
        """
        if self.processing_task is None or self.processing_task.done():
            self.processing_task = asyncio.create_task(self.batch_processor())
            logger.info(f"Started batch processor for model {self.model_id}")

    def load(self):
        """
        Create the inference engine.
        """
        self.maybe_retrieve_model()
        self.get_config()
        self.engine = InferenceEnginePY(self.config)
        self.loaded = True
        logger.info(f"Loaded model {self.model_id} from: {self.model_path}")

    def unload(self):
        """
        Not super clean, we might want to do better some day...
        """
        # Cancel batch processor if running
        if self.processing_task is not None and not self.processing_task.done():
            self.processing_task.cancel()
            self.processing_task = None

        # Clear any pending requests
        while not self.request_queue.empty():
            try:
                req = self.request_queue.get_nowait()
                if not req.future.done():
                    req.future.set_exception(Exception("Model unloaded"))
            except Exception:
                break
        del self.engine
        gc.collect()
        torch.cuda.empty_cache()
        self.engine = None
        self.loaded = False
        logger.info(f"Unloaded model {self.model_id}")

    def get_model_limits(self):
        """
        Return ``(max_tokens, max_input_tokens)`` for this model.

        *max_tokens* is the maximum number of generation tokens (``max_length``
        from the inference config).

        *max_input_tokens* is the maximum number of tokens that fit in the
        context window before generation.  It is derived from the effective
        context length:

        1. If ``context_length`` is set explicitly in the inference config, use
           that value.
        2. Otherwise fall back to ``original_max_position_embeddings`` from the
           model's RoPE configuration.
        3. If neither is available, default to 0 (unknown).

        Returns:
            tuple[int, int]: ``(max_tokens, max_input_tokens)``
        """
        if not self.loaded or self.engine is None:
            return 0, 0
        predictor = getattr(self.engine, "predictor", None)
        max_tokens = int(getattr(predictor, "max_length", 0) or 0)
        context_length = int(getattr(predictor, "context_length", 0) or 0)
        if context_length <= 0:
            # Fall back to original_max_position_embeddings from rope config
            rope = getattr(getattr(predictor, "model", None), "decoder", None)
            rope = getattr(rope, "rope", None)
            rope_cfg = getattr(getattr(rope, "model_config", None), "rope_config", None)
            context_length = int(getattr(rope_cfg, "original_max_position_embeddings", 0) or 0)
        max_input_tokens = max(0, context_length - max_tokens)
        return max_tokens, max_input_tokens

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in *text* using the model's tokenizer.

        The method walks the engine's transform pipeline to find the first
        ``TokenizerTransform`` and uses it to tokenize the string.  If no
        tokenizer is found (e.g. the model uses whitespace splitting), it falls
        back to ``estimate_tokens`` (≈4 chars per token).

        Args:
            text (str): The text to tokenize.

        Returns:
            int: Token count.
        """
        from eole.transforms.tokenize import TokenizerTransform

        if not self.loaded or self.engine is None:
            return estimate_tokens(text)
        transform_pipe = getattr(self.engine, "transform_pipe", None)
        if transform_pipe is not None:
            for transform in getattr(transform_pipe, "transforms", []):
                if isinstance(transform, TokenizerTransform):
                    try:
                        return len(transform._tokenize(text, side="src"))
                    except Exception:
                        break
        return estimate_tokens(text)

    def apply_chat_template(self, inputs, tools=None, tool_choice=None):
        """
        Render the model input based on the model chat template
        and the request inputs.

        Optional *tools* (list of OpenAI-style function tool dicts) and
        *tool_choice* are forwarded as Jinja2 template variables so that
        models whose templates understand them can emit the appropriate
        prompting tokens.

        HuggingFace template compatibility:
        - ``strftime_now`` global is available for templates that embed the
          current date/time (e.g. Llama 3.3).
        - ``{% generation %}`` blocks are stripped before rendering; they mark
          where generation starts in HF's ``apply_chat_template`` but are not
          standard Jinja2 and are not needed for inference.
        - ``tojson`` filter is provided for templates that serialise objects.
        """

        def raise_exception(message):
            raise TemplateError(message)

        chat_template = self.config.chat_template
        if chat_template is None:
            # Fall back to a standalone chat_template.jinja file in the model
            # directory (used by some modern HF models that don't embed the
            # template in tokenizer_config.json or config.json).
            jinja_file = os.path.join(self.local_path, "chat_template.jinja")
            if os.path.exists(jinja_file):
                with open(jinja_file, encoding="utf-8") as f:
                    chat_template = f.read()
            else:
                raise TemplateError(
                    f"Model '{self.model_id}' has no chat_template configured. "
                    "Set chat_template in the model's inference config or provide "
                    "a chat_template.jinja file in the model directory."
                )
        # Modern HuggingFace models store chat_template as a list of named
        # templates, e.g. [{"name": "default", "template": "..."}, ...].
        # Extract the "default" entry, or fall back to the first entry.
        if isinstance(chat_template, list):
            # Use .get() to safely handle list items that may lack a "template" key.
            template_str = next(
                (t.get("template") for t in chat_template if isinstance(t, dict) and t.get("name") == "default"),
                None,
            )
            if template_str is None and chat_template:
                first = chat_template[0]
                template_str = (
                    first.get("template") if isinstance(first, dict) else (first if isinstance(first, str) else None)
                )
            if not isinstance(template_str, str):
                raise TemplateError(f"Model '{self.model_id}': chat_template list contains no usable template string.")
            chat_template = template_str

        # Guard against any remaining non-string value (e.g. dict or bytes from
        # unexpected config formats) to give a clear error instead of a cryptic
        # "Can't compile non template nodes" TypeError from Jinja2.
        if not isinstance(chat_template, str):
            raise TemplateError(
                f"Model '{self.model_id}': chat_template has unexpected type "
                f"'{type(chat_template).__name__}' — expected a string. "
                "Check the model's config.json or chat_template.jinja file."
            )

        # Strip HuggingFace {% generation %} markers — they are used by the
        # HF tokenizer library to mark the generation boundary but are not
        # valid Jinja2 tags and are not needed for inference rendering.
        chat_template = chat_template.replace("{% generation %}", "")

        jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        jinja_env.globals["raise_exception"] = raise_exception
        # HF templates often call strftime_now() to embed the current date.
        jinja_env.globals["strftime_now"] = datetime.datetime.now().strftime
        # tojson is a common filter used in HF templates to serialise objects.
        jinja_env.filters["tojson"] = lambda v, **kw: json.dumps(v, ensure_ascii=False, **kw)

        template = jinja_env.from_string(chat_template)
        render_kwargs: dict = {
            "messages": inputs,
            "bos_token": "",  # handled in numericalize
            "eos_token": "",  # handled by stop conditions; use literal tokens in template
            "add_generation_prompt": True,
            # Many Qwen3 / "thinking" model templates gate on enable_thinking.
            # Default to False so the model uses direct tool-call output rather
            # than wrapping everything in <think>…</think> blocks (we strip
            # those blocks anyway, but avoiding them keeps the output cleaner).
            "enable_thinking": False,
        }
        if tools is not None:
            render_kwargs["tools"] = tools
        if tool_choice is not None:
            render_kwargs["tool_choice"] = tool_choice
        rendered_output = template.render(**render_kwargs)

        # _log_json_payload(f"RENDERED PROMPT [{self.model_id}]",rendered_output,)

        return rendered_output

    async def infer_async(self, inputs, settings={}, is_chat=False):
        """
        Queue inference request and wait for result.
        """
        # Ensure model is loaded (sync operation)
        if not self.loaded:
            self.load()

        # Ensure batch processor is running (async operation)
        await self.ensure_batch_processor()

        future = asyncio.Future()
        req = QueuedRequest(inputs=inputs, settings=settings, is_chat=is_chat, future=future, timestamp=time.time())
        await self.request_queue.put(req)
        return await future


def create_app(config_file):
    """
    Create and configure the FastAPI application.
    """
    app = FastAPI(
        title="Eole Inference Server",
        version=eole.__version__,
        summary="A simple inference server to expose various models.",
        description="",  # TODO
    )

    server = Server()
    server.start(config_file)

    @app.get("/")
    def root(request: Request):
        """
        Root endpoint returning HTML content to help users find the docs.
        """
        html_content = f"""
        <html>
            <head>
                <title>Eole Server</title>
            </head>
            <body>
                <h1>Eole Server</h1>
                <p>Probably not what you're looking for.</p>
                <p>API docs --> <a href="{request.url}docs">{request.url}docs</a>.</p>
            </body>
        </html>
        """
        return HTMLResponse(content=html_content, status_code=200)

    @app.get("/models")
    def models():
        """
        Return available models currently exposed.
        """
        models = server.available_models()
        out = {"models": models}
        return out

    @app.post("/unload_model")
    def unload_model(model_id):
        """
        Unload a specific model.
        """
        server.models[model_id].unload()

    @app.get("/health")
    def health():
        """
        Health check endpoint.
        """
        out = {}
        out["status"] = STATUS_OK
        return out

    @app.post("/infer", response_model=TextResponse)
    async def infer(
        request: Union[TextRequest, ChatRequest] = Body(
            openapi_examples={
                "text_request": {
                    "summary": "Text Request Example",
                    "description": "A sample text request",
                    "value": TextRequest.Config.json_schema_extra["example"],
                },
                "chat_request": {
                    "summary": "Chat Request Example",
                    "description": "A sample chat request",
                    "value": ChatRequest.Config.json_schema_extra["example"],
                },
            },
        ),
    ):
        """
        Run inference on the given input.
        """
        if isinstance(request, TextRequest):
            inputs = request.inputs if isinstance(request.inputs, list) else [request.inputs]
        else:  # ChatRequest
            inputs = request.messages
        model_id = request.model
        # automatically grab anything that is not model/inputs
        # (we could probably rely on pydantic model once properly implemented)
        non_settings_keys = ["inputs", "messages", "model"]
        settings = {k: v for k, v in request.model_dump().items() if k not in non_settings_keys}

        await server.maybe_load_model(model_id)
        scores, preds = await server.models[model_id].infer_async(
            inputs,
            settings=settings,
            is_chat=isinstance(request, ChatRequest),
        )
        response = {"predictions": preds, "scores": scores}
        return response

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
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

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
                )
                completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
                created_ts = int(time.time())

                async def _stream_sse():
                    """Async generator that yields SSE-formatted data lines."""
                    loop = asyncio.get_event_loop()
                    chunk_queue: asyncio.Queue = asyncio.Queue()

                    def _produce():
                        try:
                            for chunk in model_obj.engine.infer_list_stream(chat_input, settings=settings):
                                loop.call_soon_threadsafe(chunk_queue.put_nowait, chunk)
                        except Exception as exc:  # noqa: BLE001
                            loop.call_soon_threadsafe(chunk_queue.put_nowait, exc)
                        finally:
                            loop.call_soon_threadsafe(chunk_queue.put_nowait, None)

                    loop.run_in_executor(None, _produce)

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

                    raw_chunks: list = []
                    while True:
                        item = await chunk_queue.get()
                        if item is None:
                            break
                        if isinstance(item, Exception):
                            raise item
                        raw_chunks.append(item)

                    full_text = _post_process_model_output("".join(raw_chunks))
                    _log_json_payload("MODEL RESPONSE [openai stream]", full_text)

                    # Stream cleaned text as a single delta (model is done by now)
                    content_chunk = OpenAIStreamChunk(
                        id=completion_id,
                        created=created_ts,
                        model=model_id,
                        choices=[
                            OpenAIStreamChoice(
                                index=0,
                                delta=OpenAIStreamDelta(content=full_text),
                            )
                        ],
                    )
                    yield f"data: {content_chunk.model_dump_json()}\n\n"

                    # Final chunk with finish_reason
                    final_chunk = OpenAIStreamChunk(
                        id=completion_id,
                        created=created_ts,
                        model=model_id,
                        choices=[
                            OpenAIStreamChoice(
                                index=0,
                                delta=OpenAIStreamDelta(),
                                finish_reason="stop",
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
            # Run inference using chat mode
            scores, preds = await server.models[model_id].infer_async(
                inputs=messages,
                settings=settings,
                is_chat=True,
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
            completion_text = _post_process_model_output(completion_text)
            completion_tokens = estimate_tokens(completion_text)

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
                        message=OpenAIMessage(role="assistant", content=completion_text),
                        finish_reason="stop",  # You might want to detect "length" based on max_tokens
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
    # -----------------------------------------------------------------------

    def _anthropic_model_info(model_id: str, model_obj: "Model") -> dict:
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
                    loop = asyncio.get_event_loop()
                    chunk_queue: asyncio.Queue = asyncio.Queue()

                    def _produce():
                        try:
                            for chunk in model_obj.engine.infer_list_stream(chat_input, settings=settings):
                                loop.call_soon_threadsafe(chunk_queue.put_nowait, chunk)
                        except Exception as exc:  # noqa: BLE001
                            loop.call_soon_threadsafe(chunk_queue.put_nowait, exc)
                        finally:
                            loop.call_soon_threadsafe(chunk_queue.put_nowait, None)

                    loop.run_in_executor(None, _produce)

                    # Buffer all model output so we can classify the response
                    # type before emitting SSE events.
                    raw_chunks: list = []
                    while True:
                        item = await chunk_queue.get()
                        if item is None:
                            break
                        if isinstance(item, Exception):
                            raise item
                        raw_chunks.append(item)

                    raw_text = "".join(raw_chunks)
                    _log_json_payload("MODEL RESPONSE [anthropic stream raw]", raw_text)
                    full_text = _post_process_model_output(raw_text)
                    # _log_json_payload("MODEL RESPONSE [anthropic stream]", full_text)

                    content_blocks, stop_reason = _parse_anthropic_response_content(full_text)
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

    return app


@register_bin(name="serve")
class Serve(BaseBin):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--config",
            "-config",
            "-c",
            default="./server_conf.yaml",
            help="Path of server YAML config file.",
        )
        parser.add_argument("--host", type=str, default="0.0.0.0")
        parser.add_argument("--port", type=int, default="5000")

    @classmethod
    def run(cls, args):
        app = create_app(args.config)
        uvicorn.run(app=app, host=args.host, port=args.port, log_level="info")
