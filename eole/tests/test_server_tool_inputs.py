"""Tool argument types must survive Qwen XML output and Anthropic translation."""

import unittest
from types import SimpleNamespace

from eole.server.model import Model
from eole.server.anthropic import AnthropicTool
from eole.server.utils import _normalize_developer_role
from eole.server.tool_parsing import _coerce_tool_inputs_from_schema, _parse_anthropic_response_content
from eole.server.openai_chat import (
    OpenAIFunctionCall,
    OpenAIMessage,
    _OpenAIStreamParser,
    OpenAIToolCall,
    _openai_messages_for_template,
    _parse_openai_complete_response,
    _parse_openai_response_content,
    _prepare_openai_tool_request,
)


class TestServerToolInputs(unittest.TestCase):
    def test_model_limits_ignore_previous_request_generation_budget(self):
        model = Model()
        model.loaded = True
        model.config = SimpleNamespace(max_length=2048, context_length=32768)
        model.engine = SimpleNamespace(predictor=SimpleNamespace(max_length=32000, context_length=32768))
        self.assertEqual(model.get_model_limits(), (2048, 30720))

    def test_qwen_xml_integer_arguments(self):
        raw = "<tool_call><function=add><parameter=a>19</parameter><parameter=b>23</parameter></function></tool_call>"
        blocks, reason = _parse_anthropic_response_content(raw)
        tool = AnthropicTool(
            name="add",
            input_schema={"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}},
        )
        _coerce_tool_inputs_from_schema(blocks, [tool])
        self.assertEqual(reason, "tool_use")
        self.assertEqual(blocks[0]["input"], {"a": 19, "b": 23})

    def test_preserves_strings_and_invalid_values(self):
        blocks = [
            {
                "type": "tool_use",
                "name": "test",
                "input": {"path": "123", "bad_int": "true", "bad_json": "oops", "untyped": "42"},
            }
        ]
        tool = AnthropicTool(
            name="test",
            input_schema={
                "properties": {
                    "path": {"type": "string"},
                    "bad_int": {"type": "integer"},
                    "bad_json": {"type": "object"},
                }
            },
        )
        _coerce_tool_inputs_from_schema(blocks, [tool])
        self.assertEqual(blocks[0]["input"], {"path": "123", "bad_int": "true", "bad_json": "oops", "untyped": "42"})

    def test_nested_objects_arrays_and_existing_json_values(self):
        blocks = [
            {
                "type": "tool_use",
                "name": "test",
                "input": {"options": '{"enabled":"true","values":["1","2"]}', "count": 3},
            }
        ]
        tool = AnthropicTool(
            name="test",
            input_schema={
                "properties": {
                    "options": {
                        "type": "object",
                        "properties": {
                            "enabled": {"type": "boolean"},
                            "values": {"type": "array", "items": {"type": "integer"}},
                        },
                    },
                    "count": {"type": "integer"},
                }
            },
        )
        _coerce_tool_inputs_from_schema(blocks, [tool])
        self.assertEqual(blocks[0]["input"], {"options": {"enabled": True, "values": [1, 2]}, "count": 3})

    def test_no_schema_leaves_tool_and_text_blocks_unchanged(self):
        blocks = [{"type": "text", "text": "hello"}, {"type": "tool_use", "name": "unknown", "input": {"a": "19"}}]
        self.assertEqual(_coerce_tool_inputs_from_schema(blocks, None), blocks)
        self.assertEqual(blocks[1]["input"], {"a": "19"})

    def test_qwen_xml_becomes_multiple_openai_tool_calls(self):
        raw = """<tool_call>
<function=bash><parameter=command>ls</parameter></function>
</tool_call>
<tool_call>
<function=read><parameter=filePath>C:/Development/BloxWeaver/README.md</parameter></function>
</tool_call>"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "parameters": {
                        "type": "object",
                        "properties": {"filePath": {"type": "string"}},
                        "required": ["filePath"],
                    },
                },
            },
        ]

        content, calls, reason = _parse_openai_response_content(raw, tools)

        self.assertIsNone(content)
        self.assertEqual(reason, "tool_calls")
        self.assertEqual([call.function.name for call in calls], ["bash", "read"])
        self.assertEqual(calls[0].function.arguments, '{"command":"ls"}')
        self.assertEqual(calls[1].function.arguments, '{"filePath":"C:/Development/BloxWeaver/README.md"}')
        self.assertTrue(all(call.id.startswith("call_") for call in calls))

    def test_openai_plain_text_response_is_unchanged(self):
        content, calls, reason = _parse_openai_response_content("Hello!", None)
        self.assertEqual(content, "Hello!")
        self.assertEqual(calls, [])
        self.assertEqual(reason, "stop")

    def test_developer_role_maps_to_system_for_qwen_template(self):
        messages = [{"role": "developer", "content": "Follow these instructions."}]
        normalized = _normalize_developer_role(messages, "{% if message.role == 'system' %}")
        self.assertEqual(normalized, [{"role": "system", "content": "Follow these instructions."}])
        self.assertEqual(messages[0]["role"], "developer")

    def test_developer_role_is_preserved_for_gpt_oss_template(self):
        messages = [{"role": "developer", "content": "Follow these instructions."}]
        normalized = _normalize_developer_role(messages, "<|channel|>analysis<|message|>")
        self.assertIs(normalized, messages)
        self.assertEqual(normalized[0]["role"], "developer")

    def test_openai_stream_parser_handles_every_tag_boundary(self):
        source = "before<think>reasoning</think>after"
        expected = [("text", "before"), ("reasoning", "reasoning"), ("text", "after")]
        for size in range(1, len(source) + 1):
            with self.subTest(size=size):
                parser = _OpenAIStreamParser()
                parts = []
                for offset in range(0, len(source), size):
                    parts.extend(parser.feed(source[offset : offset + size]))
                parts.extend(parser.feed("", final=True))
                combined = []
                for kind, value in parts:
                    if combined and combined[-1][0] == kind:
                        combined[-1] = (kind, combined[-1][1] + value)
                    else:
                        combined.append((kind, value))
                self.assertEqual(combined, expected)

    def test_openai_stream_parser_handles_template_opened_thinking(self):
        source = "reasoning</think>answer"
        expected = [("reasoning", "reasoning"), ("text", "answer")]
        for size in range(1, len(source) + 1):
            with self.subTest(size=size):
                parser = _OpenAIStreamParser(thinking=True)
                parts = []
                for offset in range(0, len(source), size):
                    parts.extend(parser.feed(source[offset : offset + size]))
                parts.extend(parser.feed("", final=True))
                combined = []
                for kind, value in parts:
                    if combined and combined[-1][0] == kind:
                        combined[-1] = (kind, combined[-1][1] + value)
                    else:
                        combined.append((kind, value))
                self.assertEqual(combined, expected)

    def test_openai_stream_parser_buffers_tool_across_every_boundary(self):
        tool = '<tool_call>{"name":"bash","arguments":{"command":"ls"}}</tool_call>'
        source = f"before{tool}after"
        expected = [("text", "before"), ("tool", tool), ("text", "after")]
        for size in range(1, len(source) + 1):
            with self.subTest(size=size):
                parser = _OpenAIStreamParser(parse_tools=True)
                parts = []
                for offset in range(0, len(source), size):
                    parts.extend(parser.feed(source[offset : offset + size]))
                parts.extend(parser.feed("", final=True))
                combined = []
                for kind, value in parts:
                    if combined and combined[-1][0] == kind:
                        combined[-1] = (kind, combined[-1][1] + value)
                    else:
                        combined.append((kind, value))
                self.assertEqual(combined, expected)

    def test_openai_stream_parser_normalizes_newlines_in_buffered_tool(self):
        raw = (
            "<tool_call>｟newline｠<function=bash>｟newline｠"
            "<parameter=command>｟newline｠pwd; ls -la｟newline｠</parameter>｟newline｠"
            "</function>｟newline｠</tool_call>"
        )
        parser = _OpenAIStreamParser(parse_tools=True)

        parts = parser.feed(raw, final=True)

        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0][0], "tool")
        self.assertNotIn("｟newline｠", parts[0][1])
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                },
            }
        ]
        content, calls, reason = _parse_openai_response_content(parts[0][1], tools)
        self.assertIsNone(content)
        self.assertEqual(reason, "tool_calls")
        self.assertEqual(calls[0].function.arguments, '{"command":"pwd; ls -la"}')

    def test_openai_stream_parser_preserves_literal_closing_tag(self):
        parser = _OpenAIStreamParser()
        self.assertEqual(parser.feed("literal </think> text", final=True), [("text", "literal </think> text")])

    def test_openai_complete_response_preserves_tool_inside_reasoning(self):
        raw = (
            "<think>Use the tool."
            "<tool_call><function=bash><parameter=command>ls</parameter></function></tool_call>"
            "</think>"
        )
        tools = [{"type": "function", "function": {"name": "bash", "parameters": {}}}]

        content, reasoning, calls, reason = _parse_openai_complete_response(raw, tools)

        self.assertIsNone(content)
        self.assertEqual(reasoning, "Use the tool.")
        self.assertEqual(reason, "tool_calls")
        self.assertEqual(calls[0].function.name, "bash")
        self.assertEqual(calls[0].function.arguments, '{"command":"ls"}')

    def test_openai_tool_history_survives_template_conversion(self):
        messages = [
            OpenAIMessage(
                role="assistant",
                reasoning_content="I should inspect files.",
                tool_calls=[
                    OpenAIToolCall(
                        id="call_123",
                        function=OpenAIFunctionCall(name="bash", arguments='{"command":"ls"}'),
                    )
                ],
            ),
            OpenAIMessage(role="tool", tool_call_id="call_123", name="bash", content="README.md"),
        ]

        rendered = _openai_messages_for_template(messages)

        self.assertEqual(rendered[0]["tool_calls"][0]["function"]["arguments"], {"command": "ls"})
        self.assertEqual(rendered[0]["reasoning_content"], "I should inspect files.")
        self.assertEqual(rendered[1]["tool_call_id"], "call_123")
        self.assertEqual(rendered[1]["name"], "bash")

    def test_invalid_openai_argument_history_is_not_rewritten(self):
        messages = [
            OpenAIMessage(
                role="assistant",
                tool_calls=[
                    OpenAIToolCall(
                        id="call_123",
                        function=OpenAIFunctionCall(name="bash", arguments='{"command":'),
                    )
                ],
            )
        ]
        rendered = _openai_messages_for_template(messages)
        self.assertNotIn("tool_calls", rendered[0])
        self.assertIn(r'"arguments":"{\"command\":"', rendered[0]["content"])

    def test_named_openai_tool_choice_filters_and_requires_tool(self):
        tools = [
            {"type": "function", "function": {"name": "bash", "parameters": {}}},
            {"type": "function", "function": {"name": "read", "parameters": {}}},
        ]
        messages, selected = _prepare_openai_tool_request(
            [{"role": "user", "content": "Inspect files."}],
            tools,
            {"type": "function", "function": {"name": "read"}},
        )
        self.assertEqual([tool["function"]["name"] for tool in selected], ["read"])
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("must call the read tool", messages[0]["content"])

    def test_openai_tool_choice_none_removes_tools(self):
        messages = [{"role": "user", "content": "Do not call tools."}]
        tools = [{"type": "function", "function": {"name": "bash", "parameters": {}}}]
        self.assertEqual(_prepare_openai_tool_request(messages, tools, "none"), (messages, None))


if __name__ == "__main__":
    unittest.main()
