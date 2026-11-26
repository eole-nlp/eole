"""Run greedy text generation with the official Inkling 2 Mini checkpoint."""

import argparse
import json
from pathlib import Path
import re
import sys
from time import perf_counter

from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer
import torch

from eole.models.inkling2 import Inkling2Mini


def render_prompt(template, prompt, reasoning_effort):
    # Generation tags mark spans for HF's training mask. They don't emit any text.
    template = re.sub(r"{%-?\s*(?:end)?generation\s*-?%}", "", template)
    env = ImmutableSandboxedEnvironment()
    return env.from_string(template).render(
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort=reasoning_effort,
    )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Local thinkingmachines/Inkling-2-Mini snapshot")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prefill-chunk", type=int, default=128)
    parser.add_argument("--reasoning-effort", choices=("none", "minimal", "low", "medium", "high"), default="none")
    parser.add_argument("prompt")
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.prefill_chunk < 1:
        parser.error("The generation limit and prefill chunk size must be positive.")

    device = torch.device(args.device)
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    eos = json.loads((args.model / "generation_config.json").read_text())["eos_token_id"]
    prompt = render_prompt((args.model / "chat_template.jinja").read_text(), args.prompt, args.reasoning_effort)
    tokens = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False).ids], device=device)

    start = perf_counter()
    model = Inkling2Mini.from_checkpoint(args.model, device)
    print(
        f"Loaded {len(model.layers)} physical layers with {model.num_visits} visits in {perf_counter() - start:.2f}s",
        file=sys.stderr,
    )
    cache = model.new_cache()
    start = perf_counter()
    for chunk in tokens.split(args.prefill_chunk, dim=1):
        logits = model(chunk, cache)
    next_token = logits.argmax(-1)
    generated = [next_token.item()]
    first_token = perf_counter()
    while len(generated) < args.max_new_tokens and generated[-1] != eos:
        next_token = model(next_token, cache).argmax(-1)
        generated.append(next_token.item())
    decode = perf_counter() - first_token

    print(tokenizer.decode(generated, skip_special_tokens=False))
    print(
        f"Prompt {tokens.size(1)} tokens, first token {first_token - start:.2f}s, output {len(generated)} tokens, "
        f"decode {(len(generated) - 1) / max(decode, 1e-6):.1f} tok/s",
        file=sys.stderr,
    )
    if device.type == "cuda":
        print(f"Peak PyTorch allocation {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB", file=sys.stderr)


if __name__ == "__main__":
    main()
