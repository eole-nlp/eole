"""Measure repeated warm inference using the same YAML as eole predict."""

import argparse
import json
import logging
import os
import time
from pathlib import Path

from check_checkpoint import inspect_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    import torch
    import yaml
    from eole.config.run import PredictConfig
    from eole.server.model import Model
    from eole.inference_engine import InferenceEnginePY
    from eole.utils.logging import logger

    config = PredictConfig(**yaml.safe_load(os.path.expandvars(args.config.read_text())))
    mode = "mtp" if config.self_speculative_decoding else "baseline"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable; run in a GPU-enabled terminal.")
    if config.batch_size != 1 or config.beam_size != 1 or config.top_k != 1:
        raise ValueError("Use the single-sequence greedy recipe configuration.")
    prompts = [line.rstrip("\n") for line in Path(config.src).read_text().splitlines() if line.strip()]
    if len(prompts) != 1:
        raise ValueError("Benchmark expects exactly one preformatted prompt line in config.src.")
    if len(config.model_path) != 1:
        raise ValueError("Use exactly one checkpoint for this single-GPU benchmark.")
    metadata = inspect_checkpoint(config.model_path[0])
    events = []

    class Capture(logging.Handler):
        def emit(self, record):
            events.append(record.getMessage())

    engine = InferenceEnginePY(config)
    capture = Capture()
    logger.addHandler(capture)
    model = Model(model_id="qwen3.8-27B")
    model.config = config
    model.engine = engine
    model.loaded = True
    results = []
    try:
        engine.infer_list(prompts)  # Exclude warmup / compilation.
        backend_diagnostics = [line for line in events if line.startswith("Inference ")]
        events.clear()
        for index in range(args.runs):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            _, _, predictions = engine.infer_list(prompts)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            text = predictions[0][0].replace("｟newline｠", "\n")
            results.append(
                {
                    "run": index + 1,
                    "wall_seconds": elapsed,
                    "retokenized_tokens_approx": model.count_tokens(text),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "text": text,
                }
            )
        if mode == "mtp" and (
            any("using normal decoding" in line for line in events)
            or not any("MTP draft acceptance:" in line for line in events)
        ):
            raise RuntimeError("Speculation was not demonstrated; inspect fallback / checkpoint / generation length.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "mode": mode,
                    "checkpoint": metadata,
                    "config_path": str(args.config),
                    "gpu": torch.cuda.get_device_name(),
                    "torch": torch.__version__,
                    "compile": os.environ.get("EOLE_TORCH_COMPILE", "0"),
                    "compile_mode": os.environ.get("EOLE_COMPILE_MODE", "0"),
                    "prompt": prompts[0],
                    "max_tokens": config.max_length,
                    "draft_tokens": config.self_speculative_num_tokens,
                    "backend": config.self_attn_backend,
                    "results": results,
                    "diagnostics": events,
                    "backend_diagnostics": backend_diagnostics,
                },
                indent=2,
            )
        )
        print(f"Saved {args.output}")
    finally:
        logger.removeHandler(capture)
        engine.terminate()


if __name__ == "__main__":
    main()
