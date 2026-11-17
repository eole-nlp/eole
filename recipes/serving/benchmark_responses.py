"""Measure streamed Responses requests and a synthetic coding-tool round trip.

Run against a dedicated server: the benchmark intentionally creates concurrent
load. It never executes model-generated code or reads user files.
"""

import argparse
import asyncio
import json
import math
import os
from pathlib import Path
from time import perf_counter


async def read_events(content):
    """Parse SSE records, retaining multiline data and arbitrary HTTP fragments."""
    data = []
    async for raw in content:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield json.loads("\n".join(data))
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
    if data:
        yield json.loads("\n".join(data))


async def request(session, url, payload):
    started = perf_counter()
    arrivals = []
    final = None
    async with session.post(url, json={**payload, "eole_metrics": True}) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {(await response.text())[:500]}")
        async for event in read_events(response.content):
            kind = event.get("type", "")
            if kind in (
                "response.output_text.delta",
                "response.function_call_arguments.delta",
                "response.reasoning_text.delta",
                "response.reasoning_summary_text.delta",
            ) and event.get("delta"):
                arrivals.append(perf_counter() - started)
            if kind in ("error", "response.failed"):
                raise RuntimeError(f"Stream failed: {event}")
            if kind in ("response.completed", "response.incomplete"):
                final = event["response"]
    elapsed = perf_counter() - started
    if final is None:
        raise RuntimeError("Stream ended without a terminal Responses event")
    tokens = final.get("usage", {}).get("output_tokens")
    if not isinstance(tokens, int) or tokens < 0:
        raise RuntimeError("Terminal response must report output_tokens; SSE chunks are not tokens")
    metrics = final.get("eole_metrics")
    if metrics is None:
        raise RuntimeError("Missing server decode timings; use the matching Eole server revision")
    if metrics.get("decode_tokens") != max(0, tokens - 1):
        raise RuntimeError("Decode token count must exclude exactly the first generated token")
    if tokens > 1 and (not metrics.get("decode_seconds") or metrics["decode_seconds"] <= 0):
        raise RuntimeError("Missing or invalid first-to-last-token decode duration")
    return final, dict(
        seconds=elapsed,
        first_output_seconds=arrivals[0] if arrivals else None,
        output_tokens=tokens,
        input_tokens=final.get("usage", {}).get("input_tokens"),
        status=final["status"],
        decode=metrics,
        output_events=len(arrivals),
        inter_chunk_seconds=[b - a for a, b in zip(arrivals, arrivals[1:])],
    )


TOOL = {
    "type": "function",
    "name": "read_fixture",
    "description": "Read the synthetic counter.py fixture for this benchmark.",
    "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string", "enum": ["counter.py"]}},
        "required": ["path"],
        "additionalProperties": False,
    },
}
FIXTURE = "def count_lines(text):\n    return len(text.split('\\n'))\n"


async def job(session, args, identifier, scenario):
    started = perf_counter()
    turns = []
    base = dict(model=args.model, stream=True, store=False, temperature=0, max_output_tokens=args.max_tokens)
    # A distinct suffix prevents identical full-prompt cache hits between jobs;
    # the shared instruction prefix and tool schema remain reusable.
    history = [
        {
            "role": "user",
            "content": (
                "Read counter.py with read_fixture, then fix count_lines so empty text has zero lines. "
                "After the tool result, return a concise patch and two tests. "
                if scenario == "tool"
                else "Write a Python function that counts nonempty lines, with a docstring and two tests. "
            )
            + f"Benchmark request {identifier}.",
        }
    ]
    url = args.base_url.rstrip("/") + "/responses"
    final, timing = await request(
        session,
        url,
        {
            **base,
            "input": history,
            **(
                {"tools": [TOOL], "tool_choice": "required", "parallel_tool_calls": False} if scenario == "tool" else {}
            ),
        },
    )
    turns.append(timing)
    if scenario == "tool":
        calls = [item for item in final["output"] if item["type"] == "function_call"]
        if len(calls) != 1 or calls[0]["name"] != "read_fixture":
            raise RuntimeError("Expected exactly one read_fixture call (increase --max-tokens if truncated)")
        call = calls[0]
        if json.loads(call["arguments"]) != {"path": "counter.py"}:
            raise RuntimeError("Model requested an unexpected fixture")
        await asyncio.sleep(args.tool_delay)
        history += final["output"]
        history.append({"type": "function_call_output", "call_id": call["call_id"], "output": FIXTURE})
        _, timing = await request(session, url, {**base, "input": history, "tools": [TOOL], "tool_choice": "none"})
        turns.append(timing)
    return dict(id=identifier, scenario=scenario, seconds=perf_counter() - started, turns=turns)


def percentile(values, quantile):
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * quantile) - 1)] if values else None


def summarize(rows, seconds):
    turns = [turn for row in rows for turn in row["turns"]]
    first = [t["first_output_seconds"] for t in turns if t["first_output_seconds"] is not None]
    gaps = [gap for t in turns for gap in t["inter_chunk_seconds"]]
    tokens = sum(t["output_tokens"] for t in turns)
    decoded = [t["decode"] for t in turns if t["decode"].get("decode_tokens_per_second") is not None]
    decode_tokens = sum(d["decode_tokens"] for d in decoded)
    return dict(
        decode_tokens=decode_tokens,
        request_decode_p50_tokens_per_second=percentile([d["decode_tokens_per_second"] for d in decoded], 0.5),
        request_decode_p10_tokens_per_second=percentile([d["decode_tokens_per_second"] for d in decoded], 0.1),
        jobs_per_minute=len(rows) * 60 / seconds,
        jobs=len(rows),
        requests=len(turns),
        output_tokens=tokens,
        wall_seconds=seconds,
        end_to_end_output_tokens_per_second=tokens / seconds,
        first_output_p50_seconds=percentile(first, 0.5),
        first_output_p95_seconds=percentile(first, 0.95),
        request_p95_seconds=percentile([t["seconds"] for t in turns], 0.95),
        job_p50_seconds=percentile([r["seconds"] for r in rows], 0.5),
        job_p95_seconds=percentile([r["seconds"] for r in rows], 0.95),
        inter_chunk_p95_seconds=percentile(gaps, 0.95),
        incomplete_requests=sum(t["status"] == "incomplete" for t in turns),
    )


async def run(args):
    import aiohttp

    headers = {"Authorization": f"Bearer {os.environ[args.api_key_env]}"} if args.api_key_env else {}
    report = dict(
        model=args.model,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        tool_delay=args.tool_delay,
        note="Inter-chunk gaps are not per-token latencies; no coding-quality claim.",
        cells=[],
    )
    async with aiohttp.ClientSession(
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=args.timeout),
        connector=aiohttp.TCPConnector(limit=max(args.concurrency)),
    ) as session:
        # Warm model and HTTP path separately from the measured cells.
        await job(session, args, "warmup", "text")
        for scenario in args.scenarios:
            for concurrency in args.concurrency:
                rows = []
                semaphore = asyncio.Semaphore(concurrency)

                async def bounded(identifier):
                    async with semaphore:
                        row = await job(session, args, identifier, scenario)
                        rows.append(row)

                started = perf_counter()
                tasks = [asyncio.create_task(bounded(i)) for i in range(concurrency * args.repeats)]
                try:
                    await asyncio.gather(*tasks)
                finally:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                cell = dict(
                    scenario=scenario, concurrency=concurrency, **summarize(rows, perf_counter() - started), rows=rows
                )
                report["cells"].append(cell)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2))
                print(json.dumps({k: v for k, v in cell.items() if k != "rows"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:5019/v1")
    parser.add_argument("--model", default="qwen3.8-27B")
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--scenarios", nargs="+", choices=["text", "tool"], default=["text", "tool"])
    parser.add_argument("--repeats", type=int, default=2, help="Jobs per concurrent worker in each cell")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--tool-delay", type=float, default=0.05)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--api-key-env", help="Environment variable containing a key; its value is never saved")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.concurrency) < 1 or args.repeats < 1 or args.max_tokens < 1 or args.tool_delay < 0 or args.timeout <= 0:
        parser.error("Counts/timeouts must be positive; tool delay must be nonnegative")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
