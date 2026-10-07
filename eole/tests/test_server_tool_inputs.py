"""Tool argument types must survive Qwen XML output and Anthropic translation."""

import unittest
from types import SimpleNamespace

from eole.bin.run.serve import AnthropicTool, Model, _coerce_tool_inputs_from_schema, _parse_anthropic_response_content


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


if __name__ == "__main__":
    unittest.main()
