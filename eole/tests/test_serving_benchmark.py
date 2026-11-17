"""Verify the load benchmark counts reported tokens and handles Responses SSE."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

path = Path(__file__).resolve().parents[2] / "recipes/serving/benchmark_responses.py"
spec = importlib.util.spec_from_file_location("serving_benchmark", path)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class Content:
    def __init__(self, events):
        self.events = events

    async def __aiter__(self):
        for event in self.events:
            yield b"event: ignored\n"
            yield ("data: " + json.dumps(event) + "\n").encode()
            yield b"\n"


class Response:
    status = 200

    def __init__(self, events):
        self.content = Content(events)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class Session:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.payloads = []

    def post(self, url, json):
        self.payloads.append(json)
        return Response(next(self.responses))


def terminal(output=None):
    return {
        "type": "response.completed",
        "response": {
            "status": "completed",
            "output": output or [],
            "usage": {"output_tokens": 7},
        },
    }


class BenchmarkTests(unittest.IsolatedAsyncioTestCase):
    async def test_tokens_come_from_usage_not_chunk_count(self):
        events = [
            {"type": "response.created"},
            {"type": "response.output_text.delta", "delta": "hello"},
            {"type": "response.output_text.delta", "delta": " world"},
            terminal(),
        ]
        _, stats = await benchmark.request(Session(events), "http://local/responses", {})
        self.assertEqual(stats["output_tokens"], 7)
        self.assertEqual(stats["output_events"], 2)
        self.assertEqual(len(stats["inter_chunk_seconds"]), 1)
        self.assertLessEqual(stats["first_output_seconds"], stats["seconds"])

    async def test_missing_terminal_or_usage_fails(self):
        with self.assertRaisesRegex(RuntimeError, "terminal"):
            await benchmark.request(Session([]), "", {})
        event = terminal()
        del event["response"]["usage"]
        with self.assertRaisesRegex(RuntimeError, "output_tokens"):
            await benchmark.request(Session([event]), "", {})

    async def test_tool_round_trip_preserves_call_id(self):
        call = {
            "type": "function_call",
            "name": "read_fixture",
            "call_id": "call_123",
            "arguments": '{"path":"counter.py"}',
        }
        session = Session([terminal([call])], [terminal()])
        args = SimpleNamespace(model="test", base_url="http://local/v1", max_tokens=128, tool_delay=0)
        row = await benchmark.job(session, args, 1, "tool")
        self.assertEqual(len(row["turns"]), 2)
        result = session.payloads[1]["input"][-1]
        self.assertEqual(result["call_id"], "call_123")
        self.assertEqual(result["output"], benchmark.FIXTURE)
        self.assertEqual(session.payloads[1]["tool_choice"], "none")

    async def test_unknown_tool_is_not_executed(self):
        session = Session([terminal([{"type": "function_call", "name": "shell"}])])
        args = SimpleNamespace(model="test", base_url="http://local/v1", max_tokens=128, tool_delay=0)
        with self.assertRaisesRegex(RuntimeError, "read_fixture"):
            await benchmark.job(session, args, 1, "tool")
