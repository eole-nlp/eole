"""Run greedy text generation with the official Inkling 2 Mini checkpoint."""

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
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--context-length", type=int, default=0)
    parser.add_argument("--reasoning-effort", choices=("none", "minimal", "low", "medium", "high"), default="none")
    parser.add_argument("prompt")
    args = parser.parse_args()
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
    )
    engine = InferenceEnginePY(config)
    try:
        prompt = render_prompt(config.chat_template, args.prompt, args.reasoning_effort)
        _, _, predictions = engine.infer_list([prompt])
        print(predictions[0][0])
    finally:
        engine.terminate()


if __name__ == "__main__":
    main()
