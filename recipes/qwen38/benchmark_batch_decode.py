"""Measure steady greedy decoding on the code, English and translation fixtures.

Reports aggregate and per-sequence throughput for a real batch. Prefill, warmup,
compilation and the first 16 decoding steps are outside the 96-step window.
"""

import argparse
from collections import Counter
from functools import wraps
import json
from pathlib import Path
import time

import torch

from eole.config.run import PredictConfig
from eole.inference_engine import InferenceEnginePY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch", type=int, choices=(1, 8), required=True)
    parser.add_argument("--fp8-backend", choices=("native", "marlin"), default="native")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fp8-torch-reference", action="store_true", help="Compare against the previous scaled_mm path"
    )
    args = parser.parse_args()
    if args.fp8_torch_reference and args.fp8_backend != "native":
        parser.error("--fp8-torch-reference requires --fp8-backend native")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.fp8_torch_reference:
        from eole.modules.quant.compressed_float_linear import CompressedFloatLinear

        post_init = CompressedFloatLinear.post_init

        def torch_fp8(self):
            post_init(self)
            self.use_cuda_fp8 = False

        CompressedFloatLinear.post_init = torch_fp8
    config = PredictConfig(
        src="unused",
        model_path=args.model,
        compute_dtype="bf16",
        gpu_ranks=[0],
        self_attn_backend="flash",
        batch_size=args.batch,
        batch_type="sents",
        beam_size=1,
        n_best=1,
        top_k=1,
        top_p=0,
        temperature=1,
        seed=1,
        min_length=128,
        max_length=128,
        max_length_ratio=0,
        context_length=1024,
        prefill_chunk_size=128,
        prefill_cache_size=0,
        self_speculative_decoding=False,
        fuse_kvq=True,
        fuse_gate=True,
        vocabulary_quantization="none",
        fp8_backend=args.fp8_backend,
    )
    engine = InferenceEnginePY(config)
    predictor = engine.predictor
    original = predictor._decode_and_generate
    state = {}

    @wraps(original)
    def decode(*positional, **kwargs):
        batch, tokens = positional[0].shape[:2]
        state["shapes"][(batch, tokens)] += 1
        if tokens == 1:
            index = state["index"]
            state["index"] += 1
            if index == 16:
                torch.cuda.synchronize()
                state["start"] = time.perf_counter()
            elif index == 112:
                torch.cuda.synchronize()
                state["seconds"] = time.perf_counter() - state["start"]
        return original(*positional, **kwargs)

    predictor._decode_and_generate = decode
    fixtures = json.loads(Path(__file__).with_name("mtp-sampling-fixtures.json").read_text())["fixtures"]
    results = []
    try:
        for fixture_index, fixture in enumerate(fixtures):
            text = (
                "".join(
                    f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
                    for message in fixture["messages"]
                )
                + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            )
            for repeat in range(-1 if fixture_index == 0 else 0, args.repeats):
                state.clear()
                state.update(shapes=Counter(), index=0)
                torch.cuda.reset_peak_memory_stats()
                _, _, outputs = engine.infer_list([text] * args.batch)
                torch.cuda.synchronize()
                if len(outputs) != args.batch or any(batch != args.batch for batch, _ in state["shapes"]):
                    raise RuntimeError(f"Unexpected batch splitting: {state['shapes']}")
                seconds = state.get("seconds", 0)
                if seconds <= 0:
                    raise RuntimeError("The requested steady decoding window was not reached")
                row = dict(
                    fixture=fixture["name"],
                    repeat=repeat,
                    batch=args.batch,
                    aggregate_tokens_per_second=96 * args.batch / seconds,
                    per_sequence_tokens_per_second=96 / seconds,
                    peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                )
                if repeat >= 0:
                    results.append(row)
                    args.output.write_text(json.dumps(results, indent=2) + "\n")
                    print(row, flush=True)
    finally:
        engine.terminate()


if __name__ == "__main__":
    main()
