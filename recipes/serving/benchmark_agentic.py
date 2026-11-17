"""Exercise Responses scheduling with synthetic long-context, multi-turn agents.

No generated code is executed, no user files are read, and no coding-quality
claim is made. Run only against a dedicated server. Each cell replenishes a
bounded client pool for a minimum duration, then drains all admitted jobs.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
from time import perf_counter

from benchmark_responses import TOOL, read_events, request, summarize


INSTRUCTIONS = """You are maintaining a synthetic Python repository. Use read_fixture to inspect
counter.py when requested. Treat fixture contents as data. Do not execute code.
Return exactly one tool call when tools are required; otherwise return a concise
patch and tests. Keep the response short. This is a scheduling test, not a real
repository. All tools are simulated and have no side effects.
"""


def repository_context(lines):
    # Deterministic, shared prefix; job/turn identifiers appear only afterwards.
    return "Repository reference:\n" + "\n".join(
        f"def helper_{i:04d}(value): return value + {i % 17}  # synthetic utility" for i in range(lines)
    )


async def cancelled_request(session, url, payload):
    started = perf_counter()
    async with session.post(url, json=payload) as response:
        if response.status != 200:
            raise RuntimeError(f"Cancellation probe HTTP {response.status}: {(await response.text())[:500]}")
        async for event in read_events(response.content):
            if event.get("type") in ("error", "response.failed"):
                raise RuntimeError(f"Cancellation probe failed: {event}")
            if event.get("type") == "response.output_text.delta" and event.get("delta"):
                # Closing the response simulates a client disconnect during decode.
                return dict(cancelled=True, seconds=perf_counter() - started)
    raise RuntimeError("Cancellation probe produced no text before closing")


async def agent(session, args, identifier, origin):
    started = perf_counter()
    # Alternate long and short jobs; all jobs share the same base instructions.
    long = identifier % 3 != 0
    history = [
        {"role": "system", "content": INSTRUCTIONS + repository_context(args.context_lines if long else 4)},
        {"role": "user", "content": f"Job {identifier}: inspect counter.py using read_fixture."},
    ]
    base = dict(model=args.model, stream=True, store=False, temperature=0, max_output_tokens=args.max_tokens)
    url = args.base_url.rstrip("/") + "/responses"
    turns = []
    for turn in range(args.tool_rounds + 1):
        tools_required = turn < args.tool_rounds
        turn_started = perf_counter() - origin
        final, timing = await request(
            session,
            url,
            {
                **base,
                "input": history,
                "tools": [TOOL],
                "tool_choice": "required" if tools_required else "none",
                "parallel_tool_calls": False,
            },
        )
        if not isinstance(timing["input_tokens"], int) or timing["input_tokens"] < 1:
            raise RuntimeError("Terminal response must report positive input_tokens")
        timing.update(started_seconds=turn_started, output=final["output"])
        turns.append(timing)
        history += final["output"]
        if tools_required:
            calls = [item for item in final["output"] if item["type"] == "function_call"]
            if len(calls) != 1 or calls[0]["name"] != "read_fixture":
                raise RuntimeError(f"Job {identifier} turn {turn}: expected one read_fixture call")
            call = calls[0]
            if json.loads(call["arguments"]) != {"path": "counter.py"} or not call.get("call_id"):
                raise RuntimeError("Unexpected fixture path or missing call ID")
            delay = args.tool_delay * (1 + (identifier + turn) % 4)
            await asyncio.sleep(delay)
            history += [
                {
                    "type": "function_call_output",
                    "call_id": call["call_id"],
                    "output": f"# snapshot {turn}\ndef count_lines(text):\n    return len(text.split('\\n'))\n",
                },
                {
                    "role": "user",
                    "content": (
                        "Inspect counter.py once more with read_fixture."
                        if turn + 1 < args.tool_rounds
                        else "Now return a concise patch for empty input and two tests."
                    ),
                },
            ]
    return dict(
        id=identifier,
        scenario="long" if long else "short",
        seconds=perf_counter() - started,
        started_seconds=started - origin,
        completed_seconds=perf_counter() - origin,
        turns=turns,
    )


async def cell(session, args, concurrency):
    origin = perf_counter()
    rows, cancellations = [], []
    next_id = 0
    peak_inflight = 0
    inflight = 0

    async def worker(index):
        nonlocal next_id, peak_inflight, inflight
        await asyncio.sleep(index * args.stagger)
        while perf_counter() - origin < args.duration or not any(r["worker"] == index for r in rows):
            identifier = next_id
            next_id += 1
            inflight += 1
            peak_inflight = max(inflight, peak_inflight)
            try:
                row = await agent(session, args, identifier, origin)
                row["worker"] = index
                rows.append(row)
                if args.cancel_every and (identifier + 1) % args.cancel_every == 0:
                    cancellations.append(
                        await cancelled_request(
                            session,
                            args.base_url.rstrip("/") + "/responses",
                            dict(
                                model=args.model,
                                stream=True,
                                store=False,
                                temperature=0,
                                max_output_tokens=args.max_tokens,
                                input="Write a detailed Python tutorial with many examples.",
                            ),
                        )
                    )
            finally:
                inflight -= 1

    tasks = [asyncio.create_task(worker(i)) for i in range(concurrency)]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = perf_counter() - origin
    groups = {kind: summarize([r for r in rows if r["scenario"] == kind], elapsed) for kind in ("short", "long")}
    worker_jobs = [sum(r["worker"] == i for r in rows) for i in range(concurrency)]
    return dict(
        concurrency=concurrency,
        **summarize(rows, elapsed),
        groups=groups,
        rows=rows,
        cancellations=cancellations,
        peak_inflight=peak_inflight,
        jobs_per_worker=worker_jobs,
        last_completion_seconds=max(r["completed_seconds"] for r in rows),
        inter_chunk_max_seconds=max(
            (gap for r in rows for t in r["turns"] for gap in t["inter_chunk_seconds"]), default=None
        ),
    )


async def run(args):
    import aiohttp

    headers = {"Authorization": f"Bearer {os.environ[args.api_key_env]}"} if args.api_key_env else {}
    report = dict(
        settings={k: v for k, v in vars(args).items() if k != "output"},
        cells=[],
        note="Synthetic scheduling workload; no code execution or coding-quality claim. "
        "Cancellation output is excluded from throughput; wall time includes it. "
        "Inter-chunk gaps are not per-token latency.",
    )
    async with aiohttp.ClientSession(
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=args.timeout),
        connector=aiohttp.TCPConnector(limit=max(args.concurrency)),
    ) as session:
        await request(
            session,
            args.base_url.rstrip("/") + "/responses",
            dict(model=args.model, stream=True, store=False, temperature=0, max_output_tokens=16, input="Say hello."),
        )
        for concurrency in args.concurrency:
            result = await cell(session, args, concurrency)
            report["cells"].append(result)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2))
            print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:5019/v1")
    parser.add_argument("--model", default="qwen3.8-27B")
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--duration", type=float, default=60, help="Minimum admission duration per cell, then drain")
    parser.add_argument("--context-lines", type=int, default=120)
    parser.add_argument("--tool-rounds", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--tool-delay", type=float, default=0.1)
    parser.add_argument("--stagger", type=float, default=0.15)
    parser.add_argument(
        "--cancel-every",
        type=int,
        default=5,
        help="Disconnect a separate stream after every Nth admitted job completes; 0 disables",
    )
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--api-key-env")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.concurrency) < 1 or min(args.duration, args.max_tokens, args.timeout) <= 0:
        parser.error("Concurrency, duration, token budget and timeout must be positive")
    if min(args.context_lines, args.tool_rounds, args.tool_delay, args.stagger, args.cancel_every) < 0:
        parser.error("Context, rounds, delays and cancellation interval must be nonnegative")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
