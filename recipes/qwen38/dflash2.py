"""Single-GPU greedy DFlash2 prototype, with ordinary/MTP controls."""

import argparse

from eole.config.run import PredictConfig
from eole.inference_engine import InferenceEnginePY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Converted Eole target checkpoint")
    parser.add_argument("--drafter", help="Local incoai/Qwen3.8-27B-DFlash2 HF checkpoint")
    parser.add_argument("--mode", choices=["none", "mtp", "dflash2"], default="dflash2")
    parser.add_argument("--drafter-quantization", choices=["none", "int8"], default="none")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--prompt",
        default="Write a Python function that performs binary search in a sorted list. Explain its time complexity.",
    )
    args = parser.parse_args()
    if args.mode == "dflash2" and not args.drafter:
        parser.error("--drafter is required for DFlash2")
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
        max_length=args.max_tokens,
        context_length=args.context_length,
        prefill_cache_size=0,
        prefill_chunk_size=1024,
        self_speculative_decoding=args.mode != "none",
        self_speculative_num_tokens=3,
        dflash2_model=args.drafter if args.mode == "dflash2" else None,
        dflash2_quantization=args.drafter_quantization,
        fuse_kvq=True,
        fuse_gate=True,
        report_time=True,
        vocabulary_quantization="none",
    )
    prompt = f"<|im_start|>user\n{args.prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    engine = InferenceEnginePY(config)
    try:
        for _ in range(args.repeats):
            _, _, predictions = engine.infer_list([prompt])
            print(predictions[0][0].replace("｟newline｠", "\n"))
    finally:
        engine.terminate()


if __name__ == "__main__":
    main()
