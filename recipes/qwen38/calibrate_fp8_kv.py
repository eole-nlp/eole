"""Measure post-RoPE K/V ranges on representative text prompts."""

import argparse
import json
from pathlib import Path

import torch

from eole import EOLE_TORCH_COMPILE
from eole.config.run import PredictConfig
from eole.inference_engine import InferenceEnginePY


PERCENTILES = (0.50, 0.95, 0.99, 0.999)
SAMPLE_STRIDE = 127


def summarize(samples, maximum):
    values = torch.cat(samples)
    quantiles = torch.quantile(values, torch.tensor(PERCENTILES))
    return {
        "max": maximum,
        "sample_count": values.numel(),
        **{f"p{100 * q:g}": float(value) for q, value in zip(PERCENTILES, quantiles)},
    }


def observe(attention, maxima, samples):
    forward = attention.flash_attn_with_kvcache

    def record(query, key_cache, value_cache, key, value, **kwargs):
        start = int(kwargs["cache_seqlens"][0])
        if start < 0 or start + key.size(1) > key_cache.size(1):
            raise ValueError("Calibration prompt exceeds the configured context length")
        for i, tensor in enumerate((key, value)):
            magnitudes = tensor.float().abs()
            maxima[i] = torch.maximum(maxima[i], magnitudes.amax())
            if samples is not None:
                values = magnitudes.reshape(-1)
                offset = len(samples[i]) % min(SAMPLE_STRIDE, values.numel())
                samples[i].append(values[offset::SAMPLE_STRIDE].cpu())
        return forward(query, key_cache, value_cache, key, value, **kwargs)

    attention.flash_attn_with_kvcache = record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", type=Path, required=True, help="JSON array of preformatted model input strings")
    parser.add_argument("--output", type=Path, required=True, help="JSON config fragment")
    parser.add_argument("--report", type=Path, help="Optional JSON report of per-layer K/V magnitude percentiles")
    parser.add_argument("--context-length", type=int, default=8192)
    args = parser.parse_args()
    if args.report is not None and args.report.resolve() == args.output.resolve():
        parser.error("--report and --output must be different files")
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
        samples = []
        decoder = engine.predictor.model.decoder
        device = next(decoder.parameters()).device
        with torch.inference_mode():
            for layer in decoder.transformer_layers:
                if layer.layer_type == "linear_attention":
                    maxima.append(None)
                    samples.append(None)
                else:
                    pair = [torch.zeros((), device=device) for _ in range(2)]
                    maxima.append(pair)
                    sample_pair = ([], []) if args.report is not None else None
                    samples.append(sample_pair)
                    observe(layer.self_attn, pair, sample_pair)
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
        if args.report is not None:
            layers = [
                (
                    None
                    if pair is None
                    else {
                        "key": summarize(pair[0], limits[0]),
                        "value": summarize(pair[1], limits[1]),
                    }
                )
                for pair, limits in zip(samples, ranges)
            ]
            args.report.write_text(json.dumps({"layers": layers}, indent=2, allow_nan=False) + "\n")
    finally:
        engine.terminate()


if __name__ == "__main__":
    main()
