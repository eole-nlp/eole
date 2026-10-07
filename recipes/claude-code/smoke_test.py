"""Exercise live Eole Anthropic API behavior with a harmless synthetic tool."""

import argparse
import json
import re
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def request(base, route, payload=None, timeout=600, stream=False):
    data = None if payload is None else json.dumps(payload).encode()
    req = Request(
        base.rstrip("/") + route,
        data=data,
        headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"},
    )
    try:
        with urlopen(req, timeout=timeout) as response:
            raw = response.read().decode()
    except HTTPError as exc:
        raise RuntimeError(f"{route}: HTTP {exc.code}: {exc.read().decode()}") from exc
    if stream:
        return [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: ")]
    result = json.loads(raw)
    require(not result.get("error"), f"{route}: {result}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    parser.add_argument("--model", default="qwen3.8-27B")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    def call(route, payload=None, stream=False):
        return request(args.base_url, route, payload, args.timeout, stream)

    call("/health")
    models = call("/v1/models")
    require(any(item["id"] == args.model for item in models["data"]), "Model absent from discovery")
    messages = [{"role": "user", "content": "Say hello in one short sentence."}]
    payload = {"model": args.model, "max_tokens": 256, "messages": messages}
    count = call("/v1/messages/count_tokens", payload)
    require(count.get("input_tokens", 0) > 0, "No prompt token count")
    reply = call("/v1/messages", payload)
    require(any(b.get("type") == "text" and b.get("text", "").strip() for b in reply["content"]), "No text response")
    print("PASS health, discovery, token count, text response")

    tool = {
        "name": "add",
        "description": "Add two integers.",
        "input_schema": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    }
    history = [
        {"role": "user", "content": "Use the add tool with a=19 and b=23. After its result, reply with only the sum."}
    ]
    tool_payload = {
        **payload,
        "messages": list(history),
        "tools": [tool],
        "tool_choice": {"type": "tool", "name": "add"},
    }
    reply = call("/v1/messages", tool_payload)
    calls = [b for b in reply["content"] if b.get("type") == "tool_use"]
    require(reply.get("stop_reason") == "tool_use" and len(calls) == 1, "Expected one tool-use block")
    block = calls[0]
    require(block.get("name") == "add" and block.get("input") == {"a": 19, "b": 23}, "Incorrect tool name/arguments")
    history.extend(
        [
            {"role": "assistant", "content": reply["content"]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": block["id"], "content": "42"}]},
        ]
    )
    final = call("/v1/messages", {**payload, "messages": history, "tools": [tool]})
    text = " ".join(b.get("text", "") for b in final["content"] if b.get("type") == "text")
    require(
        final.get("stop_reason") != "tool_use" and re.search(r"\b42\b", text),
        "Tool result was not completed with the expected sum",
    )
    print("PASS tool-use / tool-result round trip")

    events = call("/v1/messages", {**payload, "stream": True}, stream=True)
    types = [event.get("type") for event in events]
    require(types and types[0] == "message_start" and types[-1] == "message_stop", "Incomplete SSE lifecycle")
    require(
        "content_block_start" in types and "content_block_delta" in types and "content_block_stop" in types,
        "Missing SSE content events",
    )
    require(any(e.get("delta", {}).get("text", "").strip() for e in events), "Empty streamed text")
    print("PASS typed text SSE (buffered server implementation)")

    events = call("/v1/messages", {**tool_payload, "stream": True}, stream=True)
    starts = [
        e["content_block"]
        for e in events
        if e.get("type") == "content_block_start" and e.get("content_block", {}).get("type") == "tool_use"
    ]
    require(len(starts) == 1 and starts[0].get("name") == "add", "Missing streamed add tool block")
    fragments = "".join(
        e.get("delta", {}).get("partial_json", "")
        for e in events
        if e.get("delta", {}).get("type") == "input_json_delta"
    )
    require(fragments and json.loads(fragments) == {"a": 19, "b": 23}, "Incorrect streamed tool arguments")
    require(
        any(e.get("delta", {}).get("stop_reason") == "tool_use" for e in events), "Missing streamed tool stop reason"
    )
    require(events[-1].get("type") == "message_stop", "Incomplete streamed tool lifecycle")
    print("PASS typed tool-use SSE")


if __name__ == "__main__":
    main()
