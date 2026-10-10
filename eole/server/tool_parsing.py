"""Shared model-emitted tool parsing and schema coercion for serving APIs."""

import json
import re
import uuid

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


def _coerce_tool_inputs_from_schema(content_blocks, tools):
    """Recover JSON values from XML argument text using the declared tool schema.

    XML parameters are initially strings. Preserve fields declared as strings
    (for example a numeric-looking filename), and convert only when a parsed
    JSON value matches the requested type. Malformed values remain unchanged.
    """
    schemas = {}
    for tool in tools or []:
        if isinstance(tool, dict):
            function = tool.get("function", tool)
            name = function.get("name", "")
            schema = function.get("parameters", function.get("input_schema", {}))
        else:
            name = tool.name
            schema = tool.input_schema
        if name:
            schemas[name] = schema

    def coerce(value, schema):
        if not isinstance(schema, dict):
            return value
        kind = schema.get("type")
        if isinstance(value, str) and kind in ("integer", "number", "boolean", "array", "object", "null"):
            try:
                parsed = json.loads(value)
            except (ValueError, TypeError):
                return value
            matches = {
                "integer": isinstance(parsed, int) and not isinstance(parsed, bool),
                "number": isinstance(parsed, (int, float)) and not isinstance(parsed, bool),
                "boolean": isinstance(parsed, bool),
                "array": isinstance(parsed, list),
                "object": isinstance(parsed, dict),
                "null": parsed is None,
            }
            if not matches[kind]:
                return value
            value = parsed
        if isinstance(value, dict):
            properties = schema.get("properties", {})
            return {key: coerce(item, properties.get(key, {})) for key, item in value.items()}
        if isinstance(value, list):
            return [coerce(item, schema.get("items", {})) for item in value]
        return value

    for block in content_blocks:
        if block.get("type") == "tool_use" and block.get("name") in schemas:
            block["input"] = coerce(block.get("input", {}), schemas[block["name"]])
    return content_blocks
