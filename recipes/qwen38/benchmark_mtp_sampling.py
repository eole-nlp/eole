"""Measure ordinary or MTP sampling with optional vocabulary quantization."""

import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--head", choices=["none", "int8", "int4"], default="none")
    parser.add_argument("--no-mtp", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--context-length", type=int, default=64000)
    args = parser.parse_args()
    if args.tokens < 2:
        parser.error("--tokens must be at least 2")

    import torch
    from eole import EOLE_COMPILE_MODE, EOLE_TORCH_COMPILE
    from eole.config.run import PredictConfig
    from eole.inference_engine import InferenceEnginePY

    fixture_path = Path(__file__).with_name("mtp-sampling-fixtures.json")
    bundle = json.loads(fixture_path.read_text())
    config = PredictConfig(
        model_path=args.model,
        vocabulary_quantization=args.head,
        src="dummy",
        compute_dtype="bf16",
        gpu_ranks=[0],
        self_attn_backend="flash",
        batch_size=1,
        beam_size=1,
        top_k=20,
        top_p=0.95,
        temperature=0.6,
        presence_penalty=1.0,
        frequency_penalty=0.0,
        seed=args.seeds[0],
        max_length=args.tokens,
        min_length=0,
        max_length_ratio=0,
        context_length=args.context_length,
        prefill_chunk_size=1024,
        prefill_cache_size=0,
        self_speculative_decoding=not args.no_mtp,
        self_speculative_num_tokens=3,
        fuse_kvq=True,
        fuse_gate=False,
        report_time=True,
    )
    report = {
        "mtp": not args.no_mtp,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "compile_mode": EOLE_COMPILE_MODE,
        "compile_enabled": EOLE_TORCH_COMPILE,
        "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "fixture_revision": bundle["revision"],
        "config": config.model_dump(mode="python"),
        "results": [],
    }

    # Exact serialization of these text-only Qwen3.8 fixtures with thinking off.
    prompts = {
        fixture["name"]: "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n" for message in fixture["messages"]
        )
        + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        for fixture in bundle["fixtures"]
    }
    engine = InferenceEnginePY(config)
    try:
        if args.head == "none" and (
            not isinstance(engine.predictor.model.generator, torch.nn.Linear)
            or engine.predictor.model.generator.weight.dtype != torch.bfloat16
        ):
            raise ValueError("This benchmark requires the original dense BF16 vocabulary head")
        engine.predictor.max_length = min(128, args.tokens)
        engine.infer_list([next(iter(prompts.values()))])
        engine.predictor.max_length = args.tokens
        for seed in args.seeds:
            for name, prompt in prompts.items():
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                predictor = engine.predictor
                predictor.decode_timings = []
                started = time.perf_counter()
                _, _, texts = engine.infer_list([prompt])
                torch.cuda.synchronize()
                tokens, seconds = predictor.decode_timings[-1]
                if not args.no_mtp and not predictor._speculative_drafted_tokens:
                    raise RuntimeError("MTP fell back; benchmark is invalid")
                row = {
                    "fixture": name,
                    "seed": seed,
                    "decode_tokens": tokens,
                    "seconds": seconds,
                    "tps": tokens / seconds,
                    "wall_seconds": time.perf_counter() - started,
                    "accepted": predictor._speculative_accepted_tokens,
                    "drafted": predictor._speculative_drafted_tokens,
                    "sha256": hashlib.sha256(texts[0][0].encode()).hexdigest(),
                    "text": texts[0][0],
                }
                report["results"].append(row)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2, default=str))
                print("RESULT", json.dumps({key: value for key, value in row.items() if key != "text"}), flush=True)
    finally:
        engine.terminate()


if __name__ == "__main__":
    main()
