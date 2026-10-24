"""Paired full/restricted MTP draft-head benchmark on one unchanged target model."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import time


PROMPTS = {
    "orwell": "Generate a 500 word text talking about George Orwell.",
    "code": "Write a Python LRU cache without functools. Include get and put, explain the design, "
    "and provide thorough tests and complexity analysis.",
    "french": "Rédigez un article de 800 mots en français sur les avantages et les limites des "
    "énergies renouvelables. Donnez des exemples concrets et une conclusion nuancée.",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-vocab", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--context-length", type=int, default=64000)
    args = parser.parse_args()
    if args.runs < 1 or args.tokens < 2:
        parser.error("--runs must be positive and --tokens must be at least 2")

    import torch
    from eole import EOLE_COMPILE_MODE, EOLE_TORCH_COMPILE
    from eole.config.run import PredictConfig
    from eole.inference_engine import InferenceEnginePY

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
        seed=1,
        max_length=args.tokens,
        min_length=0,
        max_length_ratio=0,
        context_length=args.context_length,
        self_speculative_decoding=True,
        self_speculative_num_tokens=3,
        self_speculative_draft_vocab=str(args.draft_vocab),
        fuse_kvq=True,
        fuse_gate=False,
        report_time=True,
    )
    engine = InferenceEnginePY(config)
    model = engine.predictor.model
    restricted = model._mtp_draft_head
    generator_pointer = model.generator.weight.data_ptr()
    prompts = {
        name: f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        for name, text in PROMPTS.items()
    }
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "compile_mode": EOLE_COMPILE_MODE,
        "torch_compile": EOLE_TORCH_COMPILE,
        "cuda": torch.version.cuda,
        "gpu_capability": torch.cuda.get_device_capability(),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
        ).strip(),
        "model": args.model,
        "context_length": args.context_length,
        "output_tokens": args.tokens,
        "draft_tokens": 3,
        "draft_vocab_size": restricted.token_ids.numel(),
        "draft_vocab_sha256": hashlib.sha256(args.draft_vocab.read_bytes()).hexdigest(),
        "additional_weight_bytes": restricted.weight.numel() * restricted.weight.element_size(),
        "runs_per_arm": args.runs,
        "config": dict(
            compute_dtype="bf16",
            beam_size=1,
            batch_size=1,
            top_k=1,
            top_p=0.0,
            temperature=1.0,
            seed=1,
            fuse_kvq=True,
            fuse_gate=False,
            report_time=True,
        ),
        "warmups": [],
        "results": [],
    }
    try:
        # Keep the additional head allocated in both arms to equalize memory pressure.
        for name, prompt in prompts.items():
            for arm, head in (("full", None), ("restricted", restricted)):
                model._mtp_draft_head = head
                engine.predictor.warmup_time = []
                started = time.perf_counter()
                engine.infer_list([prompt])
                if not engine.predictor._speculative_drafted_tokens:
                    raise RuntimeError("MTP fell back to normal decoding; benchmark is invalid")
                row = dict(
                    prompt=name,
                    arm=arm,
                    wall_seconds=time.perf_counter() - started,
                    warmup_seconds=sum(engine.predictor.warmup_time),
                )
                report["warmups"].append(row)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2))
                print("WARMUP", json.dumps(row), flush=True)
        for repeat in range(args.runs):
            for name, prompt in prompts.items():
                # Reverse order on alternate repeats to limit thermal/order bias.
                arms = [("full", None), ("restricted", restricted)]
                if repeat % 2:
                    arms.reverse()
                for arm, head in arms:
                    model._mtp_draft_head = head
                    predictor = engine.predictor
                    predictor.decode_timings = []
                    predictor.warmup_time = []
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    _, _, predictions = engine.infer_list([prompt])
                    torch.cuda.synchronize()
                    wall = time.perf_counter() - start
                    predictor = engine.predictor
                    if not predictor._speculative_drafted_tokens:
                        raise RuntimeError("MTP fell back to normal decoding; benchmark is invalid")
                    tokens, seconds = predictor.decode_timings[-1]
                    phases = {
                        key: sum(begin.elapsed_time(end) for begin, end in events)
                        for key, events in predictor._mtp_profile_events.items()
                    }
                    row = dict(
                        arm=arm,
                        prompt=name,
                        repeat=repeat,
                        decode_tokens=tokens,
                        decode_seconds=seconds,
                        tokens_per_second=tokens / seconds,
                        wall_seconds=wall,
                        warmup_seconds=sum(predictor.warmup_time),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        output_sha256=hashlib.sha256(predictions[0][0].encode()).hexdigest(),
                        accepted=predictor._speculative_accepted_tokens,
                        drafted=predictor._speculative_drafted_tokens,
                        phases_ms=phases,
                        cycles=len(predictor._mtp_profile_events["draft"]),
                        text=predictions[0][0],
                    )
                    assert model.generator.weight.data_ptr() == generator_pointer
                    report["results"].append(row)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(report, indent=2))
                    print("RESULT", json.dumps({k: v for k, v in row.items() if k != "text"}), flush=True)
        with torch.inference_mode():
            hidden = torch.randn(
                1, restricted.weight.size(1), device=restricted.weight.device, dtype=restricted.weight.dtype
            )
            projection_times = {}
            for arm, function in (
                ("full", lambda: model.generator(hidden).argmax(-1, keepdim=True)),
                ("restricted", lambda: restricted(hidden)),
            ):
                for _ in range(10):
                    function()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(100):
                    function()
                end.record()
                end.synchronize()
                projection_times[arm] = start.elapsed_time(end) / 100
            report["projection_ms_per_call"] = projection_times
            args.output.write_text(json.dumps(report, indent=2))
            print("PROJECTION", json.dumps(projection_times), flush=True)
    finally:
        engine.terminate()


if __name__ == "__main__":
    main()
