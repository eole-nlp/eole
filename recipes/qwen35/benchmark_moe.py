"""Compare resident and experimental CPU INT8 / GPU MoE inference."""

import argparse
import json
from pathlib import Path
import statistics
import time

from eole.config.run import PredictConfig
from eole.inference_engine import InferenceEnginePY
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=["resident", "hybrid_int8"], default="resident")
    parser.add_argument("--gpu-experts", type=int, default=16)
    parser.add_argument(
        "--no-hybrid-graph",
        "--no-expert-graph",
        dest="no_expert_graph",
        action="store_true",
        help="Disable hybrid attention/router and shared-expert GPU graphs",
    )
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.tokens < 2 or args.repeats < 1:
        parser.error("--tokens must be at least 2 and --repeats must be positive")
    config = PredictConfig(
        model_path=args.model,
        src="dummy",
        compute_dtype="bf16",
        gpu_ranks=[0],
        self_attn_backend="flash",
        batch_size=1,
        beam_size=1,
        top_k=1,
        top_p=0.0,
        temperature=1.0,
        seed=42,
        max_length=args.tokens,
        min_length=0,
        max_length_ratio=0,
        context_length=4096,
        dynamic_shapes=False,
        moe_expert_cuda_graph=not args.no_expert_graph,
        prefill_chunk_size=32 if args.backend == "hybrid_int8" else 128,
        prefill_cache_size=0,
        self_speculative_decoding=False,
        fuse_kvq=True,
        fuse_gate=True,
        report_time=True,
        moe_expert_backend=args.backend,
        moe_gpu_experts=args.gpu_experts,
    )
    prompt = (
        "<|im_start|>user\nWrite a Python function for binary search, then give an example."
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    started = time.perf_counter()
    engine = InferenceEnginePY(config)
    report = {
        "backend": args.backend,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "load_seconds": time.perf_counter() - started,
        "config": json.loads(json.dumps(config.model_dump(mode="python"), default=str)),
        "prompt": prompt,
        "runs": [],
    }
    try:
        engine.infer_list([prompt], settings={"max_length": min(16, args.tokens)})
        for _ in range(args.repeats):
            torch.cuda.reset_peak_memory_stats()
            engine.predictor.decode_timings = []
            _, _, predictions = engine.infer_list([prompt], settings={"max_length": args.tokens})
            torch.cuda.synchronize()
            tokens, seconds = engine.predictor.decode_timings[-1]
            report["runs"].append(
                {
                    "decode_tokens": tokens,
                    "decode_seconds": seconds,
                    "tokens_per_second": tokens / seconds,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                    "text": predictions[0][0],
                }
            )
            report["median_tokens_per_second"] = statistics.median(run["tokens_per_second"] for run in report["runs"])
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2))
    finally:
        engine.terminate()


if __name__ == "__main__":
    main()
