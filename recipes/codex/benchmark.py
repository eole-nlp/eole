"""Run small, executable coding tasks through Codex and a local Eole server."""

import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
from urllib.parse import urlsplit

from aiohttp import ClientSession, ClientTimeout, web

HERE = Path(__file__).resolve().parent
TASKS = ("cart", "intervals")
CONTENT_EVENTS = {
    "response.output_text.delta",
    "response.function_call_arguments.delta",
    "response.custom_tool_call_input.delta",
}
TERMINAL_EVENTS = {"response.completed", "response.incomplete", "response.failed"}


class ResponsesRelay:
    """Forward SSE incrementally and time content-bearing events per request."""

    def __init__(self, client, upstream):
        self.client = client
        self.upstream = upstream
        self.requests = {}

    async def forward(self, request):
        body = await request.read()
        started = perf_counter()
        row = {"status": None, "first_content_seconds": None, "terminal": None}
        self.requests.setdefault(request.match_info["job"], []).append(row)
        async with self.client.post(
            self.upstream + "/responses", data=body, headers={"Content-Type": "application/json"}
        ) as upstream:
            row["status"] = upstream.status
            response = web.StreamResponse(status=upstream.status, headers={"Content-Type": upstream.content_type})
            await response.prepare(request)
            try:
                async for raw in upstream.content:
                    if raw.startswith(b"data: {"):
                        event = json.loads(raw[6:])
                        kind = event["type"]
                        if kind in CONTENT_EVENTS and row["first_content_seconds"] is None:
                            row["first_content_seconds"] = perf_counter() - started
                        if kind in TERMINAL_EVENTS:
                            row["terminal"] = kind
                            row["usage"] = event.get("response", {}).get("usage")
                    await response.write(raw)
                await response.write_eof()
            finally:
                row["seconds"] = perf_counter() - started
        return response


def grade(project, task, log):
    """Run the original tests with read-only files, no network and no host home."""
    source = HERE / "tasks" / task
    edited = project / f"{task}.py"
    if edited.is_symlink() or not edited.is_file() or edited.stat().st_size > 65536:
        return False
    command = [
        "bwrap",
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--clearenv",
        "--ro-bind",
        "/usr",
        "/usr",
        "--symlink",
        "usr/lib",
        "/lib",
        "--symlink",
        "usr/lib64",
        "/lib64",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--ro-bind",
        str(edited),
        f"/work/{edited.name}",
        "--ro-bind",
        str(source / f"test_{task}.py.txt"),
        f"/work/test_{task}.py",
        "--chdir",
        "/work",
        "/usr/bin/python3",
        "-B",
        "-m",
        "unittest",
        "discover",
        "-v",
    ]
    try:
        with log.open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=10, check=False)
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0


async def run_job(args, relay, relay_url, job_id, task):
    job = args.output / job_id
    job.mkdir()
    project = job / "repo"
    source = HERE / "tasks" / task
    project.mkdir()
    for name in ("README.md", f"{task}.py"):
        shutil.copyfile(source / name, project / name)
    # Keep the intentionally failing fixture tests out of repository test discovery.
    shutil.copyfile(source / f"test_{task}.py.txt", project / f"test_{task}.py")
    subprocess.run(["git", "-C", str(project), "init", "-q"], check=True)
    home = job / "codex"
    subprocess.run(
        [
            sys.executable,
            str(HERE / "configure.py"),
            "--output-dir",
            str(home),
            "--base-url",
            f"{relay_url}/{job_id}/v1",
            "--context-window",
            str(args.context_window),
            "--output-reserve",
            "2048",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    # Codex gets an isolated home, no credentials, and workspace-write with its
    # default disabled network access for commands.
    temp_home = job / "home"
    temp_home.mkdir()
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(temp_home),
        "CODEX_HOME": str(home),
        "LANG": "C.UTF-8",
    }
    command = [
        args.codex,
        "exec",
        "--cd",
        str(project),
        "--sandbox",
        "workspace-write",
        "-c",
        'approval_policy="never"',
        "--ephemeral",
        "--json",
        (project / "README.md").read_text(),
    ]
    started = perf_counter()
    with (job / "events.jsonl").open("w") as events, (job / "stderr.log").open("w") as stderr:
        process = await asyncio.create_subprocess_exec(
            *command,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=events,
            stderr=stderr,
        )
        try:
            await asyncio.wait_for(process.wait(), args.timeout)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
    seconds = perf_counter() - started
    requests = relay.requests.get(job_id, [])
    passed = await asyncio.to_thread(grade, project, task, job / "grade.log")
    transport_ok = bool(requests) and all(
        row["status"] == 200 and row["terminal"] == "response.completed" for row in requests
    )
    record = {
        "task": task,
        "job": job_id,
        "passed": passed,
        "exit_code": process.returncode,
        "transport_ok": transport_ok,
        "seconds": seconds,
        "requests": requests,
    }
    print(f"{job_id}: tests={'pass' if passed else 'fail'}, CLI={process.returncode}, {seconds:.1f}s", flush=True)
    return record


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1] if ordered else None


async def benchmark(args):
    args.output.mkdir(parents=True)
    version = subprocess.check_output([args.codex, "--version"], text=True).strip()
    async with ClientSession(timeout=ClientTimeout(total=args.timeout)) as client:
        relay = ResponsesRelay(client, args.base_url)
        app = web.Application(client_max_size=16 * 1024**2)
        app.router.add_post("/{job}/v1/responses", relay.forward)
        runner = web.AppRunner(app)
        await runner.setup()
        try:
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            relay_url = f"http://127.0.0.1:{runner.addresses[0][1]}"
            result = {"codex": version, "context_window": args.context_window, "cells": []}
            if not args.no_warmup:
                await run_job(args, relay, relay_url, "warmup", TASKS[0])
            for concurrency in args.concurrency:
                jobs = []
                started = perf_counter()
                for wave in range(args.repeats * len(TASKS) // math.gcd(concurrency, len(TASKS))):
                    jobs.extend(
                        await asyncio.gather(
                            *(
                                run_job(
                                    args,
                                    relay,
                                    relay_url,
                                    f"c{concurrency}-{wave}-{index}",
                                    TASKS[(wave * concurrency + index) % len(TASKS)],
                                )
                                for index in range(concurrency)
                            )
                        )
                    )
                elapsed = perf_counter() - started
                first = [
                    row["first_content_seconds"]
                    for job in jobs
                    for row in job["requests"]
                    if row["first_content_seconds"] is not None
                ]
                success = sum(j["passed"] and j["exit_code"] == 0 and j["transport_ok"] for j in jobs)
                result["cells"].append(
                    {
                        "concurrency": concurrency,
                        "jobs": jobs,
                        "passed": success,
                        "wall_seconds": elapsed,
                        "successful_jobs_per_minute": 60 * success / elapsed,
                        "first_content_p50_seconds": percentile(first, 0.5),
                        "first_content_p95_seconds": percentile(first, 0.95),
                        "job_p95_seconds": percentile([job["seconds"] for job in jobs], 0.95),
                    }
                )
                (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        finally:
            await runner.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:5019/v1")
    parser.add_argument("--codex", default="codex")
    parser.add_argument("--output", required=True, type=lambda path: Path(path).expanduser().resolve())
    parser.add_argument("--context-window", type=int, default=12288)
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--repeats", type=int, default=1, help="Balanced task waves per cell")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()
    upstream = urlsplit(args.base_url)
    if upstream.scheme != "http" or upstream.hostname != "127.0.0.1" or upstream.path != "/v1":
        parser.error("base-url must be http://127.0.0.1:PORT/v1")
    if args.context_window <= 2048 or min(args.repeats, args.timeout, *args.concurrency) < 1:
        parser.error("use a context window above 2048 and positive concurrency, repeats and timeout")
    if len(set(args.concurrency)) != len(args.concurrency) or args.output.exists():
        parser.error("use distinct concurrency values and a fresh output directory")
    for command in (args.codex, "git", "bwrap"):
        if not shutil.which(command):
            parser.error(f"required executable not found: {command}")
    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
