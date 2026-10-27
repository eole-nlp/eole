"""Measure sampling with optional vocabulary quantization or restricted MTP drafts."""

import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--greedy", action="store_true", help="Use greedy target selection, retaining presence penalty")
    parser.add_argument("--fixtures", nargs="+", help="Run only the named fixtures")
    parser.add_argument("--head", choices=["none", "int8", "int4"], default="none")
    parser.add_argument("--draft-vocab", type=Path, help="Optional restricted MTP draft token-ID JSON")
    parser.add_argument("--paired-draft-vocab", action="store_true", help="Alternate full/restricted draft arms")
    parser.add_argument("--no-mtp", action="store_true")
    parser.add_argument("--fuse-kvq", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fuse-gate", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--context-length", type=int, default=64000)
    parser.add_argument("--warmup-tokens", type=int, default=128)
    parser.add_argument("--warmup-all-prompts", action="store_true")
    args = parser.parse_args()
    if args.paired_draft_vocab and not args.draft_vocab:
        parser.error("--paired-draft-vocab requires --draft-vocab")
    if args.draft_vocab and args.no_mtp:
        parser.error("--draft-vocab requires MTP")
    if args.warmup_tokens < 1:
        parser.error("--warmup-tokens must be positive")
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
        top_k=1 if args.greedy else 20,
        top_p=0.0 if args.greedy else 0.95,
        temperature=1.0 if args.greedy else 0.6,
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
        self_speculative_draft_vocab=str(args.draft_vocab) if args.draft_vocab else None,
        fuse_kvq=args.fuse_kvq,
        fuse_gate=args.fuse_gate,
        report_time=True,
    )
    report = {
        "mtp": not args.no_mtp,
        "warmup_tokens": min(args.warmup_tokens, args.tokens),
        "warmup_all_prompts": args.warmup_all_prompts,
        "paired_draft_vocab": args.paired_draft_vocab,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "compile_mode": EOLE_COMPILE_MODE,
        "compile_enabled": EOLE_TORCH_COMPILE,
        "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "fixture_revision": bundle["revision"],
        "draft_vocab_sha256": hashlib.sha256(args.draft_vocab.read_bytes()).hexdigest() if args.draft_vocab else None,
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
    if args.fixtures:
        unknown = set(args.fixtures) - prompts.keys()
        if unknown:
            parser.error(f"Unknown fixtures: {sorted(unknown)}")
        prompts = {name: prompt for name, prompt in prompts.items() if name in args.fixtures}
    engine = InferenceEnginePY(config)
    try:
        if args.head == "none" and (
            not isinstance(engine.predictor.model.generator, torch.nn.Linear)
            or engine.predictor.model.generator.weight.dtype != torch.bfloat16
        ):
            raise ValueError("This benchmark requires the original dense BF16 vocabulary head")
        model = engine.predictor.model
        restricted = getattr(model, "_mtp_draft_head", None)
        arms = ["full", "restricted"] if args.paired_draft_vocab else ["restricted" if restricted else "full"]
        engine.predictor.max_length = min(args.warmup_tokens, args.tokens)
        warmup_prompts = list(prompts.values()) if args.warmup_all_prompts else [next(iter(prompts.values()))]
        for arm in arms:
            model._mtp_draft_head = restricted if arm == "restricted" else None
            for prompt in warmup_prompts:
                if args.warmup_all_prompts:
                    torch.manual_seed(args.seeds[0])
                    torch.cuda.manual_seed_all(args.seeds[0])
                engine.infer_list([prompt])
        engine.predictor.max_length = args.tokens
        for repeat, seed in enumerate(args.seeds):
            for name, prompt in prompts.items():
                for arm in arms if repeat % 2 == 0 else arms[::-1]:
                    model._mtp_draft_head = restricted if arm == "restricted" else None
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
                        "arm": arm,
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
