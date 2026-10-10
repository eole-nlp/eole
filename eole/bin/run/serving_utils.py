"""Protocol-independent helpers for logging and model chat templates."""

import json
import re

from eole.constants import DefaultTokens
from eole.utils.logging import logger
from eole.bin.run.tool_parsing import _TOOL_CALL_SPLIT_RE, _TOOL_USE_SPLIT_RE

_LOG_SEP = "=" * 72
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


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


def _normalize_developer_role(messages: list, chat_template: str) -> list:
    """Map developer instructions for templates without GPT-OSS role semantics."""
    if "<|channel|>" in chat_template:
        return messages
    normalized = []
    for message in messages:
        if message.get("role") == "developer":
            message = {**message, "role": "system"}
        normalized.append(message)
    return normalized


def estimate_tokens(text: str) -> int:
    """
    Rough token estimation (about 4 chars per token).
    For production, use a proper tokenizer.
    """
    return len(text) // 4
