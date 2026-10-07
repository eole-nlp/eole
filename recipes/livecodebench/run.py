"""Prepare LiveCodeBench, generate through Eole's API, or grade saved code."""

import argparse
from datetime import date, datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import Request, urlopen

import yaml

UPSTREAM_COMMIT = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def request_json(url, payload, timeout):
    request = Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def check_upstream(path):
    revision = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_COMMIT:
        raise ValueError(f"LiveCodeBench must be checked out at {UPSTREAM_COMMIT}, found {revision}")
    if subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True).strip():
        raise ValueError("LiveCodeBench checkout must be clean")
    sys.path.insert(0, str(path))
    # Upstream prompt module reads its few-shot files relative to the checkout.
    os.chdir(path)


def prepare(config, output):
    from datasets import load_dataset
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem

    # Import only the official code-generation prompt module. Its package
    # initializer imports unrelated provider SDKs (including removed constants).
    spec = importlib.util.spec_from_file_location(
        "lcb_code_generation_prompt", Path("lcb_runner/prompts/code_generation.py")
    )
    prompt_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prompt_module)
    format_prompt_generation = prompt_module.format_prompt_generation
    from lcb_runner.lm_styles import LMStyle

    dataset = load_dataset(
        "livecodebench/code_generation_lite",
        split="test",
        version_tag=config["release_version"],
        trust_remote_code=True,
    )
    start = datetime.fromisoformat(config["start_date"]) if config.get("start_date") else None
    end = datetime.fromisoformat(config["end_date"]) if config.get("end_date") else None
    # Filter before decoding private tests; upstream's loader decodes the entire
    # release first, even for a small smoke subset. Match its inclusive dates.
    problems = []
    for row in dataset:
        contest_date = datetime.fromisoformat(row["contest_date"])
        if (start and contest_date < start) or (end and contest_date > end):
            continue
        problems.append(CodeGenerationProblem(**row))
        if config.get("limit") and len(problems) >= config["limit"]:
            break
    if not problems:
        raise ValueError("No problems in selected release/date window")
    tasks = []
    for problem in problems:
        tasks.append(
            {
                "id": f"{problem.platform.value}/{problem.question_id}",
                "difficulty": problem.difficulty.value,
                "messages": format_prompt_generation(problem, LMStyle.OpenAIChat),
                "evaluation_sample": problem.get_evaluation_sample(),
            }
        )
    # Keep hidden tests in a separate file: never include them in API requests.
    save(
        output / "tasks.json",
        {
            "upstream_commit": UPSTREAM_COMMIT,
            "selection": {key: config.get(key) for key in ("release_version", "start_date", "end_date", "limit")},
            "tasks": tasks,
        },
    )


def generate(config, output):
    tasks_path = output / "tasks.json"
    dataset = json.loads(tasks_path.read_text())
    destination = output / "generations.json"
    if destination.exists():
        raise ValueError("generations.json already exists; use a fresh output directory or move it before rerunning")
    artifact = {
        "tasks_sha256": hashlib.sha256(tasks_path.read_bytes()).hexdigest(),
        "upstream_commit": UPSTREAM_COMMIT,
        "config": config,
        "records": [],
    }
    for task in dataset["tasks"]:
        payload = {
            "model": config["model_id"],
            "messages": task["messages"],
            "stream": False,
            "max_tokens": config["max_tokens"],
            "temperature": 0.0,
            "top_p": 1.0,
        }
        started = time.perf_counter()
        response = request_json(
            config["base_url"].rstrip("/") + "/chat/completions", payload, config["request_timeout"]
        )
        elapsed = time.perf_counter() - started
        choice = response["choices"][0]
        text = choice["message"]["content"]
        if not isinstance(text, str):
            raise ValueError(f"Non-text completion for {task['id']}")
        artifact["records"].append(
            {
                "id": task["id"],
                "text": text,
                "wall_seconds": elapsed,
                "finish_reason": choice.get("finish_reason"),
                "usage": response.get("usage"),
            }
        )
        save(destination, artifact)
        print(f"{len(artifact['records'])}/{len(dataset['tasks'])}: {task['id']} ({elapsed:.2f}s)", flush=True)


def evaluate(config, output):
    from lcb_runner.evaluation import codegen_metrics
    from lcb_runner.lm_styles import LMStyle
    from lcb_runner.utils.extraction_utils import extract_code

    tasks_path = output / "tasks.json"
    dataset = json.loads(tasks_path.read_text())
    generations = json.loads((output / "generations.json").read_text())
    if generations["tasks_sha256"] != hashlib.sha256(tasks_path.read_bytes()).hexdigest():
        raise ValueError("Generations do not match the prepared dataset")
    tasks, records = dataset["tasks"], generations["records"]
    if [t["id"] for t in tasks] != [r["id"] for r in records]:
        raise ValueError("Incomplete or reordered generations; all selected tasks are required")
    codes = [[extract_code(r["text"], LMStyle.OpenAIChat)] for r in records]
    metrics, results, metadata = codegen_metrics(
        [t["evaluation_sample"] for t in tasks],
        codes,
        k_list=[1],
        num_process_evaluate=config["num_process_evaluate"],
        timeout=config["test_timeout"],
    )
    passed = [
        bool(result) and bool(result[0]) and all(value > 0 for value in result[0])
        for _, result in sorted(results.items())
    ]
    if len(passed) != len(tasks):
        raise ValueError("Evaluator returned incomplete results")
    by_difficulty = {}
    for difficulty in sorted({t["difficulty"] for t in tasks}):
        subset = [ok for t, ok in zip(tasks, passed) if t["difficulty"] == difficulty]
        by_difficulty[difficulty] = {"tasks": len(subset), "pass@1": sum(subset) / len(subset)}
    save(
        output / "evaluation.json",
        {
            "metrics": metrics,
            "by_difficulty": by_difficulty,
            "selection": dataset["selection"],
            "upstream_commit": UPSTREAM_COMMIT,
            "request_wall_seconds_total": sum(r["wall_seconds"] for r in records),
            "passed": sum(passed),
            "tasks": len(tasks),
            "results": results,
            "metadata": metadata,
            "codes": codes,
            "evaluation_config": config,
        },
    )
    print(json.dumps({"pass@1": metrics["pass@1"], "by_difficulty": by_difficulty}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "generate", "evaluate"))
    parser.add_argument("--config", "-c", type=Path, required=True)
    parser.add_argument("--allow-code-execution", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(os.path.expandvars(args.config.read_text()))
    if config.get("recipe_type") != "livecodebench":
        parser.error("Expected recipe_type: livecodebench")
    required = {
        "recipe_type",
        "upstream_dir",
        "output_dir",
        "release_version",
        "base_url",
        "model_id",
        "max_tokens",
        "request_timeout",
        "num_process_evaluate",
        "test_timeout",
    }
    allowed = required | {"start_date", "end_date", "limit"}
    if required - config.keys() or config.keys() - allowed:
        parser.error("Missing or unknown configuration keys")
    if config["release_version"] not in {f"release_v{i}" for i in range(1, 7)}:
        parser.error("Select an explicit release_v1 through release_v6")
    for key in ("max_tokens", "request_timeout", "num_process_evaluate", "test_timeout", "limit"):
        value = config.get(key)
        if value is None and key == "limit":
            continue
        if type(value) is not int or value < 1:
            parser.error(f"{key} must be a positive integer")
    dates = [date.fromisoformat(config[key]) if config.get(key) else None for key in ("start_date", "end_date")]
    if all(dates) and dates[0] > dates[1]:
        parser.error("start_date must not be after end_date")
    if args.phase == "evaluate" and not args.allow_code_execution:
        parser.error("Grading executes generated code; use an isolated environment and --allow-code-execution")
    output = Path(config["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.phase != "generate":
        check_upstream(Path(config["upstream_dir"]).resolve())
    if args.phase == "prepare" and (output / "tasks.json").exists():
        parser.error("tasks.json already exists; use a fresh output directory")
    {"prepare": prepare, "generate": generate, "evaluate": evaluate}[args.phase](config, output)


if __name__ == "__main__":
    main()
