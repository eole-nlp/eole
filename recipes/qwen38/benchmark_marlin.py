"""Compare Marlin builds with warmed greedy decoding and optional quantized heads."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--head", choices=("none", "int8", "int4"), default="none")
    parser.add_argument("--mtp", action="store_true")
    parser.add_argument("--fuse-gate", action="store_true", help="Pack gate/up projections before Marlin repacking")
    parser.add_argument("--no-fuse-kvq", action="store_true", help="Disable self-attention KVQ and GDN QKV/Z packing")
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--prompt-repeat", type=int, default=0, help="Repeat a paragraph to measure longer prefill")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if args.tokens < 2 or args.runs < 1 or args.prompt_repeat < 0:
        parser.error("Use at least two output tokens, one run and a nonnegative prompt repeat count")

    import torch
    from eole.config.run import PredictConfig
    from eole.inference_engine import InferenceEnginePY
    from eole.predict.generator import GeneratorLM

    options = {}
    if args.head != "none":
        if "vocabulary_quantization" not in PredictConfig.model_fields:
            parser.error("INT8/INT4 heads require the separate vocabulary quantization implementation (PR #481)")
        options["vocabulary_quantization"] = args.head
    # Retain decode timing, but exclude the overhead of per-phase MTP CUDA events.
    # This override lives only in this benchmark process.
    GeneratorLM._mtp_profile_enabled = property(lambda self: False, lambda self, value: None)
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
        min_length=0,
        max_length=args.tokens,
        max_length_ratio=0,
        context_length=64000,
        prefill_chunk_size=1024,
        prefill_cache_size=0,
        self_speculative_decoding=args.mtp,
        self_speculative_num_tokens=3,
        fuse_kvq=not args.no_fuse_kvq,
        fuse_gate=args.fuse_gate,
        report_time=True,
        seed=42,
        **options,
    )
    content = "Generate a 500 word text talking about George Orwell."
    if args.prompt_repeat:
        content = (
            "George Orwell wrote novels and essays about politics, language, individual freedom, "
            "and the abuse of power. " * args.prompt_repeat
        ) + "Generate a 500 word text talking about George Orwell."
    prompt = f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    report = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "compile": os.environ.get("EOLE_TORCH_COMPILE", "0"),
        "compile_mode": os.environ.get("EOLE_COMPILE_MODE", "0"),
        "head": args.head,
        "fuse_kvq": not args.no_fuse_kvq,
        "fuse_gate": args.fuse_gate,
        "prompt": prompt,
        "config": config.model_dump(mode="python"),
        "results": [],
    }
    engine = InferenceEnginePY(config)
    try:
        for index in range(2 + args.runs):
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            predictor = engine.predictor
            started = time.perf_counter()
            _, _, texts = engine.infer_list([prompt])
            torch.cuda.synchronize()
            if index < 2:
                continue
            tokens, seconds = predictor.decode_timings[-1]
            if args.mtp and not predictor._speculative_drafted_tokens:
                raise RuntimeError("MTP fell back to ordinary decoding")
            row = {
                "repeat": index - 2,
                "tokens": tokens,
                "seconds": seconds,
                "tps": tokens / seconds,
                "first_step_seconds": sum(predictor.step0_time),
                "wall_seconds": time.perf_counter() - started,
                "accepted": predictor._speculative_accepted_tokens,
                "drafted": predictor._speculative_drafted_tokens,
                "text_sha256": hashlib.sha256(texts[0][0].encode()).hexdigest(),
                "text": texts[0][0],
            }
            report["results"].append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
            print("RESULT", json.dumps({k: v for k, v in row.items() if k != "text"}), flush=True)
    finally:
        engine.terminate()


if __name__ == "__main__":
    main()
