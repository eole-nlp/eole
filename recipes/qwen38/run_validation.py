"""Run sequential GPU benchmarks and live API checks from a GPU-enabled shell."""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", type=Path, default=Path("recipes/qwen38/validation.yaml"))
    args = parser.parse_args()
    import yaml

    from eole.config.recipes import Qwen38ValidationConfig

    settings = Qwen38ValidationConfig(**yaml.safe_load(os.path.expandvars(args.config.read_text()))).model_dump()
    model_path = settings["model_path"]
    if not model_path or "$" in model_path:
        parser.error("Set QWEN38_MODEL or edit model_path in the validation YAML")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Run this command in a terminal where PyTorch can use the GPU.")
    root = Path(__file__).resolve().parents[2]
    output = Path(settings["output_dir"]).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output_dir in validation.yaml to avoid mixing results from different runs.")
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, QWEN38_MODEL=str(Path(model_path).resolve()))
    environment["PYTHONPATH"] = str(root) + os.pathsep + environment.get("PYTHONPATH", "")
    # Keep initial comparison eager; the README documents a separate compiled comparison.
    environment["EOLE_TORCH_COMPILE"] = "0"
    environment["EOLE_COMPILE_MODE"] = "0"
    print(f"Results: {output}", flush=True)

    def run(command, name, cwd=root, env=environment):
        print(f"Running {name}", flush=True)
        with (output / (name + ".log")).open("w") as log:
            try:
                subprocess.run(
                    command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                log.flush()
                print("\n".join((output / (name + ".log")).read_text().splitlines()[-25:]), flush=True)
                raise

    (output / "environment.json").write_text(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "model": environment["QWEN38_MODEL"],
            },
            indent=2,
        )
    )
    run([sys.executable, str(root / "recipes/qwen38/check_checkpoint.py"), environment["QWEN38_MODEL"]], "checkpoint")
    for mode in ["baseline", "mtp"]:
        run(
            [
                sys.executable,
                str(root / "recipes/qwen38/benchmark.py"),
                "--config",
                settings[mode + "_config"],
                "--output",
                str(output / (mode + ".json")),
            ],
            mode,
        )
    baseline = json.loads((output / "baseline.json").read_text())
    speculative = json.loads((output / "mtp.json").read_text())
    comparison = {
        "output_text_matches": [a["text"] == b["text"] for a, b in zip(baseline["results"], speculative["results"])],
        "baseline_wall_seconds": [r["wall_seconds"] for r in baseline["results"]],
        "mtp_wall_seconds": [r["wall_seconds"] for r in speculative["results"]],
    }
    (output / "comparison.json").write_text(json.dumps(comparison, indent=2))
    # Refuse to collide with another service; do not stop an existing process.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", settings["port"]))
    base = "http://127.0.0.1:" + str(settings["port"])
    with (output / "server.log").open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "eole.bin.main",
                "serve",
                "-c",
                settings["server_config"],
                "--host",
                "127.0.0.1",
                "--port",
                str(settings["port"]),
            ],
            cwd=root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 600
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited; inspect {output / 'server.log'}")
                try:
                    with urlopen(base + "/health", timeout=2) as response:
                        if json.load(response).get("status") == "ok":
                            break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("Server did not become ready within 10 minutes")
                time.sleep(2)
            run(
                [
                    sys.executable,
                    str(root / "recipes/claude-code/smoke_test.py"),
                    "--base-url",
                    base,
                    "--model",
                    settings["model_id"],
                ],
                "api",
            )
            if settings.get("claude", False):
                scratch = output / "claude-scratch"
                scratch.mkdir(exist_ok=True)
                (scratch / "sentinel.txt").write_text("EOLE_LOCAL_TOOL_CHECK_7391\n")
                client_env = dict(
                    environment,
                    ANTHROPIC_BASE_URL=base,
                    ANTHROPIC_AUTH_TOKEN="eole-local",
                    ANTHROPIC_MODEL=settings["model_id"],
                    ANTHROPIC_DEFAULT_OPUS_MODEL=settings["model_id"],
                    ANTHROPIC_DEFAULT_SONNET_MODEL=settings["model_id"],
                    ANTHROPIC_DEFAULT_HAIKU_MODEL=settings["model_id"],
                    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
                    CLAUDE_CODE_MAX_OUTPUT_TOKENS="2048",
                    CLAUDE_CONFIG_DIR=str(output / "claude-config"),
                )
                for provider_flag in ["CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY"]:
                    client_env.pop(provider_flag, None)
                client_env.pop("ANTHROPIC_API_KEY", None)
                run(["claude", "--version"], "claude-version", cwd=scratch, env=client_env)
                run(
                    [
                        "claude",
                        "-p",
                        "Read sentinel.txt using the Read tool and return its exact contents.",
                        "--model",
                        settings["model_id"],
                        "--allowedTools",
                        "Read",
                        "--output-format",
                        "json",
                    ],
                    "claude",
                    cwd=scratch,
                    env=client_env,
                )
                reply = json.loads((output / "claude.log").read_text())
                if reply.get("is_error") or "EOLE_LOCAL_TOOL_CHECK_7391" not in reply.get("result", ""):
                    raise RuntimeError("Claude Code did not return the file sentinel; inspect client/server logs.")
                # The prompt did not contain the sentinel. Inspect server logs for Read tool blocks as well.
            log.flush()
            if "MTP draft acceptance:" not in (output / "server.log").read_text():
                raise RuntimeError("Live API run did not demonstrate MTP acceptance; inspect server logs.")
            (output / "SUCCESS").write_text(
                "GPU benchmark and API checks passed. Review comparison and server/client logs.\n"
            )
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    print(f"Completed. Results: {output}", flush=True)


if __name__ == "__main__":
    main()
