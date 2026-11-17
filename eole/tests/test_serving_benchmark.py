"""Verify the load benchmark counts reported tokens and handles Responses SSE."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

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
            "usage": {"input_tokens": 123, "output_tokens": 7},
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
        self.assertEqual(stats["input_tokens"], 123)
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


# Load the sibling benchmark with its CLI-style import resolved explicitly.

agent_spec = importlib.util.spec_from_file_location("agentic_benchmark", path.with_name("benchmark_agentic.py"))
agentic = importlib.util.module_from_spec(agent_spec)
with patch.dict("sys.modules", benchmark_responses=benchmark):
    agent_spec.loader.exec_module(agentic)


class AgenticBenchmarkTests(unittest.IsolatedAsyncioTestCase):
    async def test_repeated_tool_turns_keep_history_and_call_ids(self):
        calls = [
            dict(type="function_call", name="read_fixture", call_id=f"call_{i}", arguments='{"path":"counter.py"}')
            for i in range(3)
        ]
        session = Session(*([terminal([call])] for call in calls), [terminal()])
        args = SimpleNamespace(
            model="test", base_url="http://local/v1", max_tokens=128, context_lines=100, tool_rounds=3, tool_delay=0
        )
        row = await agentic.agent(session, args, 1, 0)
        self.assertEqual(len(row["turns"]), 4)
        self.assertEqual(row["scenario"], "long")
        final_history = session.payloads[-1]["input"]
        outputs = [item for item in final_history if item.get("type") == "function_call_output"]
        self.assertEqual([item["call_id"] for item in outputs], [f"call_{i}" for i in range(3)])
        self.assertEqual(session.payloads[-1]["tool_choice"], "none")
        self.assertIn("helper_0099", final_history[0]["content"])

    async def test_disconnect_waits_for_real_text_and_closes_response(self):
        closed = []

        class ClosingResponse(Response):
            async def __aexit__(self, *args):
                closed.append(True)

        class ClosingSession:
            def post(self, *args, **kwargs):
                return ClosingResponse(
                    [{"type": "response.created"}, {"type": "response.output_text.delta", "delta": "hello"}]
                )

        result = await agentic.cancelled_request(ClosingSession(), "", {})
        self.assertTrue(result["cancelled"])
        self.assertEqual(closed, [True])

    async def test_cancellation_probe_rejects_terminal_without_text(self):
        with self.assertRaisesRegex(RuntimeError, "no text"):
            await agentic.cancelled_request(Session([terminal()]), "", {})

    async def test_agent_rejects_missing_prompt_usage(self):
        event = terminal()
        del event["response"]["usage"]["input_tokens"]
        args = SimpleNamespace(
            model="test", base_url="http://local/v1", max_tokens=128, context_lines=4, tool_rounds=0, tool_delay=0
        )
        with self.assertRaisesRegex(RuntimeError, "input_tokens"):
            await agentic.agent(Session([event]), args, 0, 0)
