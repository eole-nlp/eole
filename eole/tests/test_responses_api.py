"""Responses protocol tests use a fake engine, never a GPU or remote API."""

import json
import threading
import unittest
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from eole.bin.run.responses import (
    OutputParser,
    ResponsesRequest,
    register_responses,
    responses_messages,
)


class FakeModel:
    def __init__(self):
        self.engine = self
        self.chunks = ["Hello ", "world"]
        self.fail = False
        self.stats = None
        self.started = threading.Event()
        self.release = None

    def get_model_limits(self):
        return 128, 1024

    def count_tokens(self, text):
        return len(text.split())

    def apply_chat_template(self, messages, **kwargs):
        self.messages, self.kwargs = messages, kwargs
        return " ".join(str(m.get("content", "")) for m in messages)

    def infer_list_stream(self, prompt, settings, generation_stats=None):
        self.settings = settings
        for chunk in self.chunks:
            yield chunk
        if self.fail:
            raise RuntimeError("engine failed")
        if generation_stats is not None and self.stats is not None:
            generation_stats.update(self.stats)


class TestResponsesAPI(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()

        async def load(_):
            pass

        app = FastAPI()
        register_responses(app, SimpleNamespace(models={"qwen": self.model}, maybe_load_model=load))
        self.client = TestClient(app)

    def post(self, **kwargs):
        return self.client.post("/v1/responses", json={"model": "qwen", "input": "Hello", **kwargs})

    def events(self, **kwargs):
        response = self.post(stream=True, **kwargs)
        self.assertEqual(response.status_code, 200)
        return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]

    def test_text_stream_and_final_output_match(self):
        events = self.events()
        self.assertEqual([e["sequence_number"] for e in events], list(range(len(events))))
        self.assertEqual(events[0]["type"], "response.created")
        self.assertEqual(events[-1]["type"], "response.completed")
        deltas = [e["delta"] for e in events if e["type"] == "response.output_text.delta"]
        self.assertEqual(deltas, ["Hello ", "world"])
        self.assertEqual(events[-1]["response"]["output"][0]["content"][0]["text"], "".join(deltas))
        self.assertEqual(self.post().json()["output"][0]["content"][0]["text"], "".join(deltas))

    def test_function_tool_round_trip(self):
        self.model.chunks = list(
            "<think>hidden</think><tool_call><function=add>"
            "<parameter=a>19</parameter><parameter=b>23</parameter></function></tool_call>"
        )
        tools = [
            {
                "type": "function",
                "name": "add",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                },
            }
        ]
        events = self.events(tools=tools)
        call = events[-1]["response"]["output"][0]
        self.assertEqual(call["type"], "function_call")
        self.assertEqual(json.loads(call["arguments"]), {"a": 19, "b": 23})
        self.assertIn("response.function_call_arguments.delta", [e["type"] for e in events])
        self.model.chunks = ["42"]
        response = self.post(
            input=[
                {"role": "user", "content": [{"type": "input_text", "text": "Add"}]},
                call,
                {
                    "type": "function_call_output",
                    "call_id": call["call_id"],
                    "output": "42",
                },
            ],
            tools=tools,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            self.model.messages[-1],
            {"role": "tool", "tool_call_id": call["call_id"], "content": "42"},
        )
        self.assertEqual(
            self.model.messages[-2]["tool_calls"][0]["function"]["arguments"],
            {"a": 19, "b": 23},
        )

    def test_newline_wrapped_tool_arguments_are_typed_in_json_and_sse(self):
        raw = (
            "<tool_call>｟newline｠<function=add>｟newline｠"
            "<parameter=a>｟newline｠19｟newline｠</parameter>｟newline｠"
            "<parameter=b>｟newline｠23｟newline｠</parameter>｟newline｠"
            "<parameter=label>123</parameter>"
            "<parameter=script>first｟newline｠second</parameter>"
            "</function>｟newline｠</tool_call>"
        )
        tools = [
            {
                "type": "function",
                "name": "add",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                        "label": {"type": "string"},
                        "script": {"type": "string"},
                    },
                },
            }
        ]
        self.model.chunks = list(raw)
        expected = {"a": 19, "b": 23, "label": "123", "script": "first\nsecond"}
        for stream in (False, True):
            with self.subTest(stream=stream):
                if stream:
                    events = self.events(tools=tools)
                    result = events[-1]["response"]
                    delta = next(e["delta"] for e in events if e["type"] == "response.function_call_arguments.delta")
                    self.assertEqual(json.loads(delta), expected)
                else:
                    response = self.post(tools=tools)
                    self.assertEqual(response.status_code, 200)
                    result = response.json()
                self.assertEqual(json.loads(result["output"][0]["arguments"]), expected)

    def test_buffered_tool_normalizes_sentinels_at_every_chunk_boundary(self):
        raw = '<tool_call>｟newline｠{"name":"add","arguments":{"a":19,"b":23}}｟newline｠</tool_call>'
        expected = raw.replace("｟newline｠", "\n")
        for size in range(1, len(raw) + 1):
            with self.subTest(size=size):
                parser = OutputParser()
                parts = []
                for offset in range(0, len(raw), size):
                    parts.extend(parser.feed(raw[offset : offset + size]))
                parts.extend(parser.feed("", final=True))
                self.assertEqual(parts, [("tool", expected)])

    def test_custom_tool_round_trip(self):
        patch = "*** Begin Patch\n*** End Patch"
        self.model.chunks = [
            "<tool_call>" + json.dumps({"name": "apply_patch", "arguments": {"input": patch}}) + "</tool_call>"
        ]
        tools = [
            {
                "type": "custom",
                "name": "apply_patch",
                "format": {
                    "type": "grammar",
                    "syntax": "lark",
                    "definition": "start: /.+/",
                },
            }
        ]
        events = self.events(tools=tools)
        call = events[-1]["response"]["output"][0]
        self.assertEqual(call["input"], patch)
        self.assertEqual(call["type"], "custom_tool_call")
        self.model.chunks = ["Done"]
        self.assertEqual(
            self.post(
                input=[
                    call,
                    {
                        "type": "custom_tool_call_output",
                        "call_id": call["call_id"],
                        "output": "ok",
                    },
                ],
                tools=tools,
            ).status_code,
            200,
        )

    def test_namespace_tool_round_trip(self):
        self.model.chunks = ['<tool_call>{"name":"files.read","arguments":{"path":"a.txt"}}</tool_call>']
        tools = [
            {
                "type": "namespace",
                "name": "files",
                "tools": [
                    {
                        "type": "function",
                        "name": "read",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                        },
                    }
                ],
            }
        ]
        call = self.events(tools=tools)[-1]["response"]["output"][0]
        self.assertEqual((call["namespace"], call["name"]), ("files", "read"))
        self.model.chunks = ["done"]
        self.assertEqual(
            self.post(
                input=[
                    call,
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": "contents",
                    },
                ],
                tools=tools,
            ).status_code,
            200,
        )
        self.assertEqual(self.model.messages[0]["tool_calls"][0]["function"]["name"], "files.read")

    def test_generation_budget_is_incomplete(self):
        events = self.events(max_output_tokens=1)
        self.assertEqual(events[-1]["type"], "response.incomplete")
        self.assertEqual(
            events[-1]["response"]["incomplete_details"],
            {"reason": "max_output_tokens"},
        )
        self.assertEqual(self.post(max_output_tokens=1).json()["status"], "incomplete")

    def test_eos_at_budget_boundary_is_completed(self):
        self.model.stats = {"output_tokens": 1, "stopped_on_eos": True}
        events = self.events(max_output_tokens=1)
        self.assertEqual(events[-1]["type"], "response.completed")
        self.assertEqual(events[-1]["response"]["usage"]["output_tokens"], 1)

    def test_forced_tool_and_disabled_parallel_calls(self):
        tool = {
            "type": "function",
            "name": "add",
            "parameters": {"type": "object", "properties": {}},
        }
        events = self.events(tools=[tool], tool_choice={"type": "function", "name": "add"})
        self.assertEqual(events[-1]["type"], "response.failed")
        self.model.chunks = ['<tool_call>{"name":"add","arguments":{}}</tool_call>' * 2]
        self.assertEqual(
            self.events(tools=[tool], parallel_tool_calls=False)[-1]["type"],
            "response.failed",
        )

    def test_streaming_text_is_available_before_generation_finishes(self):
        parser = OutputParser()
        self.assertEqual(parser.feed("hello "), [("text", "hello ")])
        self.assertEqual(parser.feed("<thi"), [])
        self.assertEqual(parser.feed("nk>hidden</think>world"), [("text", "world")])

    def test_eos_stream_metadata_counts_and_suppresses_terminal_token(self):
        from eole.predict.streamer import GenerationStreamer

        streamer = GenerationStreamer(
            {"tgt": SimpleNamespace(ids_to_tokens=["hello", "<eos>"])},
            eos_token_ids=[1],
        )
        streamer.put([0])
        streamer.put([1])
        streamer.end()
        self.assertEqual(list(streamer), ["hello"])
        self.assertEqual((streamer.token_count, streamer.last_token_id), (2, 1))

    def test_engine_error_never_completes(self):
        self.model.fail = True
        events = self.events()
        self.assertEqual(events[-1]["type"], "response.failed")
        self.assertNotIn("response.completed", [e["type"] for e in events])
        self.assertEqual(self.post().status_code, 500)

    def test_unsupported_features_and_limits(self):
        for params in [
            dict(previous_response_id="resp_old"),
            dict(store=True),
            dict(tools=[{"type": "web_search"}]),
            dict(input=[{"role": "user", "content": [{"type": "input_image"}]}]),
            dict(max_output_tokens=129),
            dict(input="word " * 1025),
            dict(text={"format": {"type": "json_schema"}}),
            dict(background=True),
            dict(context_management=[{"type": "compaction"}]),
        ]:
            with self.subTest(params=params):
                self.assertEqual(self.post(**params).status_code, 400)
        self.assertEqual(self.post(model="missing").status_code, 404)

    def test_malformed_or_undeclared_tool_calls_fail(self):
        for text in [
            "<tool_call>{",
            '<tool_call>{"name":"unknown","arguments":{}}</tool_call>',
        ]:
            self.model.chunks = [text]
            self.assertEqual(self.events()[-1]["type"], "response.failed")

    def test_developer_and_instructions(self):
        req = ResponsesRequest(
            model="qwen",
            instructions="first",
            input=[
                {"role": "developer", "content": "second"},
                {"role": "user", "content": "third"},
            ],
        )
        self.assertEqual(
            responses_messages(req),
            [
                {"role": "system", "content": "first\n\nsecond"},
                {"role": "user", "content": "third"},
            ],
        )

    def test_multiple_developer_messages_render_with_qwen_system_constraint(self):
        from eole.bin.run.serve import Model

        model = Model()
        model.config = SimpleNamespace(
            chat_template=(
                "{% for m in messages %}"
                "{% if m.role == 'system' and not loop.first %}"
                "{{ raise_exception('System message must be at the beginning.') }}"
                "{% endif %}{{ m.role }}:{{ m.content }};{% endfor %}"
            )
        )
        self.model.apply_chat_template = model.apply_chat_template
        items = [
            {"role": "developer", "content": [{"type": "input_text", "text": "policy"}]},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "developer", "content": "updated policy"},
            {"role": "user", "content": "next question"},
        ]
        response = self.post(instructions="base", input=items)
        self.assertEqual(response.status_code, 200)
        messages = responses_messages(ResponsesRequest(model="qwen", instructions="base", input=items))
        self.assertEqual(messages[0], {"role": "system", "content": "base\n\npolicy\n\nupdated policy"})
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])

    def test_template_failure_is_an_explained_400(self):
        from jinja2.exceptions import TemplateError

        def fail(*args, **kwargs):
            raise TemplateError("invalid message history")

        self.model.apply_chat_template = fail
        response = self.post()
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["message"], "invalid message history")

    def test_tag_boundaries_and_literal_angle_brackets(self):
        source = 'a < b\n<think>secret</think>ok<tool_call>{"name":"f","arguments":{}}</tool_call>after'
        expected = None
        for size in range(1, len(source) + 1):
            parser = OutputParser()
            pairs = []
            for i in range(0, len(source), size):
                pairs.extend(parser.feed(source[i : i + size]))
            pairs.extend(parser.feed("", final=True))
            actual = (
                "".join(v for k, v in pairs if k == "text"),
                [v for k, v in pairs if k == "tool"],
            )
            if expected is None:
                expected = actual
            self.assertEqual(actual, expected)
        self.assertEqual(expected[0], "a < b\nokafter")


if __name__ == "__main__":
    unittest.main()
