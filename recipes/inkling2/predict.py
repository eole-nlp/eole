"""Run greedy text or image prompts with the official Inkling 2 Mini checkpoint."""

import argparse
import re

from jinja2.sandbox import ImmutableSandboxedEnvironment

from eole.config.run import PredictConfig
from eole.inference_engine import InferenceEnginePY


def render_prompt(template, prompt, reasoning_effort):
    # Generation tags mark spans for HF's training mask. They don't emit any text.
    template = re.sub(r"{%-?\s*(?:end)?generation\s*-?%}", "", template)
    env = ImmutableSandboxedEnvironment()
    return env.from_string(template).render(
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort=reasoning_effort,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="thinkingmachines/Inkling-2-Mini")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--kernel-backend", choices=("eole", "vllm"), default="eole")
    parser.add_argument("--image", action="append", default=[], help="Image path. Repeat to attach multiple images.")
    parser.add_argument("--dspark", action="store_true", help="Use the bundled DSpark drafter.")
    parser.add_argument(
        "--fused-relative-attention", action="store_true", help="Enable the fused Blackwell relative-attention decoder."
    )
    parser.add_argument("--draft-tokens", type=int, help="Maximum DSpark proposals per verification.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--context-length", type=int, default=0)
    parser.add_argument("--reasoning-effort", choices=("none", "minimal", "low", "medium", "high"), default="none")
    parser.add_argument("prompt")
    args = parser.parse_args()
    if args.draft_tokens is not None and (not args.dspark or args.draft_tokens < 1):
        parser.error("--draft-tokens requires --dspark and a positive count")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")

    config = PredictConfig(
        model_path=[args.model],
        src="dummy",
        batch_size=1,
        beam_size=1,
        top_k=1,
        max_length=args.max_new_tokens,
        max_length_ratio=0,
        context_length=args.context_length,
        gpu_ranks=[args.gpu],
        compute_dtype="bf16",
        report_time=True,
        dspark=args.dspark,
        inkling_kernel_backend=args.kernel_backend,
        relative_attention_backend="fused" if args.fused_relative_attention else "default",
        data_type="image" if args.image else "text",
        **({"self_speculative_num_tokens": args.draft_tokens} if args.draft_tokens is not None else {}),
    )
    engine = InferenceEnginePY(config)
    try:
        marker = "EOLEINKLINGIMAGEPLACEHOLDER"
        text = marker + "\n" + args.prompt if args.image else args.prompt
        prompt = render_prompt(config.chat_template, text, args.reasoning_effort)
        if args.image:
            images = {f"image{i}": path for i, path in enumerate(args.image)}
            if getattr(config.model, "adapter", None) != "inkling2":
                parser.error("Image inputs require a checkpoint converted with the vision encoder.")
            markers = "".join(f"{{{key}}}" for key in images)
            # The corpus expands only these placeholders and preserves literal user braces.
            prompt = prompt.replace("{", "{{").replace("}", "}}")
            prompt = {"text": prompt.replace(marker, markers, 1), "images": images}
        _, _, predictions = engine.infer_list([prompt])
        print(predictions[0][0])
    finally:
        engine.terminate()


if __name__ == "__main__":
    main()
