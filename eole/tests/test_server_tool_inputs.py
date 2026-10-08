"""Tool argument types must survive Qwen XML output and Anthropic translation."""

import unittest
from types import SimpleNamespace

from eole.bin.run.serve import (
    AnthropicTool,
    Model,
    OpenAIFunctionCall,
    OpenAIMessage,
    OpenAIToolCall,
    _coerce_tool_inputs_from_schema,
    _openai_messages_for_template,
    _parse_anthropic_response_content,
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

    def test_openai_tool_history_survives_template_conversion(self):
        messages = [
            OpenAIMessage(
                role="assistant",
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
