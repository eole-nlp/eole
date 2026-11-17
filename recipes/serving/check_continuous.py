"""Check mixed-age admission/cancellation/slot reuse against solo Qwen decoding.

This probes the eager scheduler core, not the HTTP server. Output tokens are
compared exactly for these short prompts; no universal bit-identity is claimed.
"""

import argparse
import json
import time
from pathlib import Path
import torch
from eole.config.run import PredictConfig
from eole.inference_engine import InferenceEnginePY
from eole.predict.continuous import ContinuousGreedy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", action="store_true", help="Also time eager banks at 1/2/4/8 slots")
    args = parser.parse_args()

    from eole import EOLE_TORCH_COMPILE

    if EOLE_TORCH_COMPILE:
        raise ValueError("Use EOLE_TORCH_COMPILE=0 for this eager core probe")

    cfg = PredictConfig(
        model_path=args.model,
        src="dummy",
        compute_dtype="bf16",
        gpu_ranks=[0],
        self_attn_backend="flash",
        dynamic_shapes=False,
        batch_size=1,
        beam_size=1,
        top_k=1,
        top_p=0.0,
        temperature=1.0,
        max_length=64,
        context_length=1024,
        prefill_cache_size=0,
        prefill_chunk_size=256,
        self_speculative_decoding=False,
        fuse_kvq=True,
        fuse_gate=True,
        report_time=False,
    )
    engine = InferenceEnginePY(cfg)
    p = engine.predictor
    prompts = [
        "Write a Python binary search function.",
        "Tell a story about a lighthouse keeper who finds an unexpected visitor.",
        "Translate into English: 这个工具可以读取文件并计算每一行的长度。",
    ]
    ids = []
    for text in prompts:
        prompt = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        batch, _ = next(iter(engine._build_inference_iterator(src=[prompt])))
        ids.append(batch["src"][0].tolist())

    @torch.inference_mode()
    def solo(prompt, budget):
        p.model.decoder._disable_cache()
        src = torch.tensor([prompt], device="cuda")
        src_len = torch.tensor([len(prompt)], device="cuda")
        result = []
        for index in range(budget):
            inputs = src if index == 0 else torch.tensor([[result[-1]]], device="cuda")
            scores, _ = p._decode_and_generate(
                inputs, None, src_len=src_len, step=0 if index == 0 else len(prompt) + index - 1
            )
            token = scores.argmax(-1).item()
            result.append(token)
            if token in p._tgt_eos_idx:
                break
        return result

    try:
        expected = {}
        for name, prompt, budget in [("a", ids[0], 32), ("c", ids[2], 24), ("d", ids[1], 16), ("e", ids[2], 20)]:
            expected[name] = solo(prompt, budget)
        print("SOLO_READY", flush=True)
        runner = ContinuousGreedy(p, slots=2, max_pending=8)
        runner.submit("a", ids[0], 32)
        runner.submit("b", ids[1], 48)
        runner.submit("d", ids[1], 16)
        timeline = []
        started = time.perf_counter()
        for tick in range(100):
            if tick == 5:
                runner.submit("c", ids[2], 24)
            if tick == 8:
                runner.cancel("b")
                runner.submit("e", ids[2], 20)
            events = runner.tick()
            timeline.append(
                dict(tick=tick, events=events, active={str(k): v.request_id for k, v in runner.active.items()})
            )
            if not runner.pending and not runner.active:
                break
        torch.cuda.synchronize()
        comparisons = {}
        for name, want in expected.items():
            got = runner.requests[name].tokens
            mismatch = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), None)
            comparisons[name] = dict(
                equal=got == want,
                first_mismatch=mismatch,
                actual=got,
                expected=want,
                status=runner.requests[name].status,
            )
        report = dict(
            comparisons=comparisons,
            cancelled=runner.requests["b"].status,
            timeline=timeline,
            seconds=time.perf_counter() - started,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print(
            json.dumps(
                {k: {"equal": v["equal"], "first_mismatch": v["first_mismatch"]} for k, v in comparisons.items()}
            ),
            flush=True,
        )
        assert report["cancelled"] == "cancelled"
        assert all(v["equal"] for v in comparisons.values()), "Mixed execution differs; inspect per-token comparison"
        runner.close()
        if args.benchmark:
            report["eager_banks"] = []
            for slots in (1, 2, 4, 8):
                runner = ContinuousGreedy(p, slots=slots)
                for i in range(slots * 2):
                    runner.submit(str(i), ids[i % len(ids)], 64)
                torch.cuda.synchronize()
                start = time.perf_counter()
                while runner.pending or runner.active:
                    runner.tick()
                torch.cuda.synchronize()
                seconds = time.perf_counter() - start
                tokens = sum(len(r.tokens) for r in runner.requests.values())
                cell = dict(slots=slots, tokens=tokens, seconds=seconds, aggregate_tps=tokens / seconds)
                report["eager_banks"].append(cell)
                args.output.write_text(json.dumps(report, indent=2))
                print("EAGER_BANK", json.dumps(cell), flush=True)
                runner.close()
    finally:
        engine.terminate()


if __name__ == "__main__":
    main()
