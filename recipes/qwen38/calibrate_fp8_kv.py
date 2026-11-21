"""Measure post-RoPE K/V ranges on representative text prompts."""

import argparse
import json
from pathlib import Path

import torch

from eole import EOLE_TORCH_COMPILE
from eole.config.run import PredictConfig
from eole.inference_engine import InferenceEnginePY


def observe(attention, maxima):
    forward = attention.flash_attn_with_kvcache

    def record(query, key_cache, value_cache, key, value, **kwargs):
        start = int(kwargs["cache_seqlens"][0])
        if start < 0 or start + key.size(1) > key_cache.size(1):
            raise ValueError("Calibration prompt exceeds the configured context length")
        for i, tensor in enumerate((key, value)):
            maxima[i] = torch.maximum(maxima[i], tensor.float().abs().amax())
        return forward(query, key_cache, value_cache, key, value, **kwargs)

    attention.flash_attn_with_kvcache = record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", type=Path, required=True, help="JSON array of preformatted model input strings")
    parser.add_argument("--output", type=Path, required=True, help="JSON config fragment")
    parser.add_argument("--context-length", type=int, default=8192)
    args = parser.parse_args()
    if EOLE_TORCH_COMPILE:
        parser.error("Set EOLE_TORCH_COMPILE=0 for calibration")
    prompts = json.loads(args.prompts.read_text())
    if not isinstance(prompts, list) or not prompts or any(not isinstance(p, str) or not p.strip() for p in prompts):
        parser.error("--prompts must contain a nonempty JSON array of text prompts")
    engine = InferenceEnginePY(
        PredictConfig(
            model_path=args.model,
            src="dummy",
            gpu_ranks=[0],
            compute_dtype="bf16",
            batch_size=1,
            beam_size=1,
            top_k=1,
            min_length=0,
            max_length=1,
            max_length_ratio=0,
            self_attn_backend="flash",
            dynamic_shapes=False,
            context_length=args.context_length,
            prefill_cache_size=0,
            prefill_chunk_size=512,
        )
    )
    try:
        maxima = []
        decoder = engine.predictor.model.decoder
        device = next(decoder.parameters()).device
        with torch.inference_mode():
            for layer in decoder.transformer_layers:
                if layer.layer_type == "linear_attention":
                    maxima.append(None)
                else:
                    pair = [torch.zeros((), device=device) for _ in range(2)]
                    maxima.append(pair)
                    observe(layer.self_attn, pair)
            for prompt in prompts:
                engine.infer_list([prompt])
            ranges = [None if pair is None else [float(x) for x in pair] for pair in maxima]
            if not any(pair is not None for pair in ranges) or any(
                pair is not None and not all(x > 0 for x in pair) for pair in ranges
            ):
                raise RuntimeError("Calibration requires nonzero K and V observations in every attention layer")
            # Keep 10% headroom above the largest observed magnitude.
            scales = [None if pair is None else [x * 1.1 / 448.0 for x in pair] for pair in ranges]
        args.output.write_text(json.dumps({"kv_cache_scales": scales}, indent=2, allow_nan=False) + "\n")
    finally:
        engine.terminate()


if __name__ == "__main__":
    main()
