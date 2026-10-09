#!/usr/bin/env python
"""Check live Responses text, SSE, and a synthetic add-tool round trip."""

import argparse
import json
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:5000/v1")
    parser.add_argument("--model", default="qwen3.8-27B")
    args = parser.parse_args()

    def post(payload):
        request = urllib.request.Request(
            args.base_url.rstrip("/") + "/responses",
            data=json.dumps({"model": args.model, "store": False, **payload}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            body = response.read().decode()
        if payload.get("stream"):
            events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]
            assert events[0]["type"] == "response.created", events
            assert events[-1]["type"] == "response.completed", events[-1]
            assert [e["sequence_number"] for e in events] == list(range(len(events)))
            return events[-1]["response"], events
        result = json.loads(body)
        assert result["status"] == "completed", result
        return result, []

    def text(response):
        return "".join(
            part["text"] for item in response["output"] if item["type"] == "message" for part in item["content"]
        )

    for stream in (False, True):
        response, events = post({"input": "Reply with a short greeting.", "stream": stream, "max_output_tokens": 128})
        assert text(response).strip(), response
        if stream:
            assert any(e["type"] == "response.output_text.delta" for e in events), events
        print(f"Text stream={stream}: PASS")

    tools = [
        {
            "type": "function",
            "name": "add",
            "description": "Add two integers.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        }
    ]
    history = [
        {"role": "user", "content": "Use the add tool to compute 19 + 23. After its result, answer with the number."}
    ]
    response, events = post({"input": history, "tools": tools, "tool_choice": "required", "stream": True})
    calls = [item for item in response["output"] if item["type"] == "function_call"]
    assert len(calls) == 1 and calls[0]["name"] == "add", response
    values = json.loads(calls[0]["arguments"])
    assert values == {"a": 19, "b": 23}, values
    assert any(e["type"] == "response.function_call_arguments.delta" for e in events)
    history += response["output"] + [
        {"type": "function_call_output", "call_id": calls[0]["call_id"], "output": str(values["a"] + values["b"])}
    ]
    final, _ = post({"input": history, "tools": tools, "tool_choice": "none", "stream": True})
    assert "42" in text(final), final
    print("Typed add tool and replayed tool result: PASS")


if __name__ == "__main__":
    main()
