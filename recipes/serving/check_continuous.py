"""Long mixed-age greedy check and idle-prefill probe on a local Qwen checkpoint.

Solo references replay each request's actual prefill chunk boundaries. Matching
finite cases is not a universal promise of cross-batch bit-identical arithmetic.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from eole.config.run import PredictConfig
from eole.inference_engine import InferenceEnginePY
from eole.predict.continuous import ContinuousGreedy
from eole.utils.request_cache import RequestCache


def describe(logits, hidden):
    """Small diagnostics plus CPU hidden states, only for this correctness probe."""
    probabilities = logits.log_softmax(-1)
    values, indices = probabilities.topk(2, dim=-1)
    return [
        dict(
            selected=int(probabilities[i].argmax()),
            top_ids=indices[i].tolist(),
            top_log_probs=values[i].float().tolist(),
            hidden=hidden[i].detach().cpu().clone(),
        )
        for i in range(logits.size(0))
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiled-step", action="store_true")
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--prefill-chunk-size", type=int, default=128)
    parser.add_argument("--prefill-idle-chunk-size", type=int)
    parser.add_argument("--reference-mode", choices=["matching", "eager", "isolation"], default="matching")
    parser.add_argument("--prefill-sweep", action="store_true", help="Only measure idle chunks 128/256/512/1024")
    parser.add_argument("--allow-divergence", action="store_true", help="Write divergent cases without a nonzero exit")
    args = parser.parse_args()
    from eole import EOLE_TORCH_COMPILE

    if EOLE_TORCH_COMPILE != args.compiled_step:
        parser.error("Set EOLE_TORCH_COMPILE=1 with --compiled-step, otherwise use 0")
    if args.tokens < 16 or args.slots < 2:
        parser.error("Use at least 16 output tokens and two slots for this probe")
    config = PredictConfig(
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
        max_length=args.tokens,
        context_length=args.context_length,
        prefill_cache_size=0,
        prefill_chunk_size=256,
        self_speculative_decoding=False,
        vocabulary_quantization="none",
        fuse_kvq=True,
        fuse_gate=True,
        report_time=False,
    )
    engine = InferenceEnginePY(config)
    predictor = engine.predictor
    runner = None
    reference_runner = None
    report = dict(
        torch=torch.__version__,
        slots=args.slots,
        tokens=args.tokens,
        context_length=args.context_length,
        reference_mode=args.reference_mode,
        active_chunk=args.prefill_chunk_size,
        idle_chunk=args.prefill_idle_chunk_size,
    )

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))

    def encode(text):
        prompt = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        batch, _ = next(iter(engine._build_inference_iterator(src=[prompt])))
        return batch["src"][0].tolist()

    @torch.inference_mode()
    def solo(ids, chunks, budget):
        decoder = predictor.model.decoder
        decoder._disable_cache()
        src = torch.tensor([ids], device="cuda")
        result, records, offset = [], [], 0
        for length in chunks:
            part = src[:, offset : offset + length]
            emb = predictor.model.tgt_emb(part, step=offset)
            mask = torch.zeros(1, 1, length, device=emb.device, dtype=torch.bool)
            if offset == 0:
                decoder._init_cache(emb, mask)
            hidden, _ = decoder._forward_eager(emb, tgt_pad_mask=mask)
            offset += length
        if offset != len(ids):
            raise RuntimeError("Incomplete recorded prefill")
        for index in range(budget):
            if index:
                token = torch.tensor([[result[-1]]], device="cuda")
                emb = predictor.model.tgt_emb(token, step=len(ids) + index - 1)
                mask = torch.zeros(1, 1, 1, device=emb.device, dtype=torch.bool)
                hidden, _ = decoder._forward_eager(emb, tgt_pad_mask=mask)
            logits = predictor.model.generator(hidden[:, -1])
            record = describe(logits, hidden[:, -1])[0]
            result.append(record["selected"])
            records.append(record)
            if result[-1] in predictor._tgt_eos_idx:
                break
        return result, records

    def compiled_solo(identifier, ids, chunks, budget):
        records = []

        def reference_hook(module, inputs, logits):
            records.extend(describe(logits, inputs[0]))

        handle = predictor.model.generator.register_forward_hook(reference_hook)
        try:
            request = reference_runner.submit(identifier, ids, budget)
            while reference_runner.has_work:
                index = len(request.prefill_chunks)
                if index < len(chunks):
                    reference_runner.prefill_chunk_size = chunks[index]
                    reference_runner.prefill_idle_chunk_size = chunks[index]
                reference_runner.tick()
            if request.prefill_chunks != chunks:
                raise RuntimeError("Compiled solo used different prefill boundaries")
            tokens = list(request.tokens)
            reference_runner.forget(identifier)
            return tokens, records
        finally:
            handle.remove()

    @torch.inference_mode()
    def isolated_reference(identifier, ids, chunks, actual, records):
        # Keep the target's batch width/slot and prefix, but remove other requests.
        decoder = predictor.model.decoder
        decoder._disable_cache()
        src = torch.tensor([ids], device="cuda")
        offset = 0
        for length in chunks:
            part = src[:, offset : offset + length]
            emb = predictor.model.tgt_emb(part, step=offset)
            mask = torch.zeros(1, 1, length, device=emb.device, dtype=torch.bool)
            if offset == 0:
                decoder._init_cache(emb, mask)
            hidden, _ = decoder._forward_eager(emb, tgt_pad_mask=mask)
            offset += length
        values = describe(predictor.model.generator(hidden[:, -1]), hidden[:, -1])
        expected, observations = [values[0]["selected"]], [values[0]]
        slot = records[0]["slot"]
        reference_runner.bank.copy_row(slot, RequestCache.take(decoder))
        for index in range(1, len(actual)):
            width = records[index]["width"]
            if width == 1:
                cache = reference_runner.row_caches[slot]
                graph = reference_runner.single_graphs[slot]
                inputs = reference_runner.next_tokens[slot : slot + 1]
                selected = 0
            else:
                cache = reference_runner.bank
                graph = reference_runner.step_graph
                inputs = reference_runner.next_tokens
                inputs.zero_()
                for other in range(args.slots):
                    if other != slot:
                        cache.metadata["cache_seqlens"][other] = 0
                selected = slot
            cache.bind()
            inputs[selected, 0] = actual[index - 1]
            emb = predictor.model.tgt_emb(inputs, step=1)
            if graph is None:
                hidden, _ = decoder.forward_step(emb)
            else:
                hidden, _ = graph(
                    emb, tgt_pad_mask=torch.zeros(inputs.size(0), 1, 1, device=emb.device, dtype=torch.bool)
                )
            values = describe(predictor.model.generator(hidden[:, -1]), hidden[:, -1])
            expected.append(values[selected]["selected"])
            observations.append(values[selected])
        return expected, observations

    try:
        runner = ContinuousGreedy(
            predictor,
            slots=args.slots,
            max_pending=32,
            prefill_chunk_size=args.prefill_chunk_size,
            prefill_idle_chunk_size=args.prefill_idle_chunk_size,
            compiled=args.compiled_step,
        )
        report["graphs"] = dict(
            batch=bool(runner.step_graph and runner.step_graph.graph),
            single=[bool(g and g.graph) for g in runner.single_graphs],
        )
        if args.prefill_sweep:
            long_ids = encode(("def sample(x): return x + 1\n" * 180) + "Explain this code.")
            report["prompt_tokens"] = len(long_ids)
            report["idle_prefill"] = []
            for chunk in (128, 256, 512, 1024):
                runner.prefill_idle_chunk_size = chunk
                for repeat in range(3):
                    identifier = f"sweep-{chunk}-{repeat}"
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    runner.submit(identifier, long_ids, 1)
                    while runner.has_work:
                        runner.tick()
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - started
                    row = dict(
                        chunk=chunk, repeat=repeat, seconds=seconds, chunks=runner.requests[identifier].prefill_chunks
                    )
                    runner.forget(identifier)
                    report["idle_prefill"].append(row)
                    save()
                    print("PREFILL", json.dumps(row), flush=True)
            return

        prompts = [
            "Write a Python command-line program that inventories files and stores checksums in SQLite. Include tests.",
            ("# sample data line: alpha beta gamma\n" * 150) + "Summarize the sample data.",
            "Write a mystery story about a lighthouse keeper and an unexpected visitor.",
            "Translate into English: 这个工具可以读取文件并计算每一行的长度。请解释如何测试它。",
            "Explain binary search with Python code and edge-case tests.",
            "Write a short poem about the ocean.",
            ("def add(a, b): return a + b\n" * 60) + "Write tests and document edge cases.",
            "Describe how to implement a reliable job queue with retries and cancellation.",
            "Write a French explanation of Python generators, with examples.",
            "Write a JSON parser tutorial with several complete code examples.",
        ]
        ids = [encode(text) for text in prompts]
        for index in range(6):
            runner.submit(f"r{index}", ids[index], args.tokens)
        runner.cancel("r5")
        cancelled = {"r5"}
        actual_records = {f"r{i}": [] for i in range(10)}
        captured = []
        assigned_slots = {}
        original_emit = runner._emit

        def observe_slot(request, token, events):
            assigned_slots[request.request_id] = request.slot
            original_emit(request, token, events)

        runner._emit = observe_slot

        def hook(module, inputs, logits):
            captured.append(describe(logits, inputs[0]))

        handle = predictor.model.generator.register_forward_hook(hook)
        timeline, ticks = [], dict(single=0, batch=0)
        try:
            for tick in range(args.tokens * 12 + 100):
                if tick == 20:
                    for index in range(6, 10):
                        runner.submit(f"r{index}", ids[index], args.tokens)
                for identifier in ("r1", "r2"):
                    request = runner.requests.get(identifier)
                    if request is None or identifier in cancelled:
                        continue
                    condition = identifier == "r1" and request.status == "prefilling" and request.prefilled > 0
                    condition |= identifier == "r2" and request.status == "running" and len(request.tokens) >= 8
                    if condition:
                        runner.cancel(identifier)
                        cancelled.add(identifier)
                before = dict(runner.active)
                captured.clear()
                if before:
                    ticks["single" if len(before) == 1 else "batch"] += 1
                events = runner.tick()
                decode_records = captured[0] if before else None
                for identifier, token, status in events:
                    if identifier in (r.request_id for r in before.values()):
                        slot = next(slot for slot, request in before.items() if request.request_id == identifier)
                        record = decode_records[0 if len(before) == 1 else slot]
                    else:
                        record = captured[-1][0]
                        slot = assigned_slots[identifier]
                    record["slot"] = slot
                    record["width"] = (
                        len(decode_records) if identifier in (r.request_id for r in before.values()) else 1
                    )
                    actual_records[identifier].append(record)
                timeline.append(
                    dict(
                        tick=tick,
                        events=events,
                        active={str(slot): r.request_id for slot, r in runner.active.items()},
                        prefill=runner.prefilling.request_id if runner.prefilling else None,
                    )
                )
                if tick >= 20 and not runner.has_work:
                    break
            else:
                raise RuntimeError("Scheduler did not finish within the tick budget")
        finally:
            handle.remove()
        results = {
            key: dict(
                status=r.status,
                tokens=r.tokens,
                prefill_chunks=r.prefill_chunks,
                decode_widths=[v["width"] for v in actual_records[key]],
                slot=assigned_slots.get(key),
            )
            for key, r in runner.requests.items()
        }
        runner.close()
        runner = None
        report.update(cancelled=sorted(cancelled), ticks=ticks, timeline=timeline, results=results)
        save()
        if cancelled != {"r1", "r2", "r5"}:
            raise RuntimeError("Not all queued/prefill/decode cancellation paths were exercised")
        report["comparisons"] = {}
        if args.compiled_step and args.reference_mode in ("matching", "isolation"):
            reference_runner = ContinuousGreedy(
                predictor,
                slots=args.slots if args.reference_mode == "isolation" else 1,
                max_pending=1,
                prefill_chunk_size=args.prefill_chunk_size,
                compiled=True,
            )
        if args.reference_mode == "isolation" and reference_runner is None:
            reference_runner = ContinuousGreedy(predictor, slots=args.slots, prefill_chunk_size=args.prefill_chunk_size)
        for index in range(10):
            key = f"r{index}"
            if key in cancelled:
                continue
            result = results[key]
            if args.reference_mode == "isolation":
                expected, reference = isolated_reference(
                    key, ids[index], result["prefill_chunks"], result["tokens"], actual_records[key]
                )
            elif reference_runner is not None:
                expected, reference = compiled_solo(key, ids[index], result["prefill_chunks"], args.tokens)
            else:
                expected, reference = solo(ids[index], result["prefill_chunks"], args.tokens)
            actual = result["tokens"]
            mismatch = next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b), None)
            if mismatch is None and len(actual) != len(expected):
                mismatch = min(len(actual), len(expected))
            row = dict(
                equal=actual == expected,
                first_mismatch=mismatch,
                actual_tokens=len(actual),
                expected_tokens=len(expected),
            )
            if mismatch is not None and mismatch < min(len(actual_records[key]), len(reference)):
                left, right = actual_records[key][mismatch], reference[mismatch]
                a, b = left["hidden"].float(), right["hidden"].float()
                row["hidden_relative_rms"] = float(
                    ((a - b).square().mean() / b.square().mean().clamp_min(1e-20)).sqrt()
                )
                row["actual_top"] = {k: v for k, v in left.items() if k != "hidden"}
                row["reference_top"] = {k: v for k, v in right.items() if k != "hidden"}
            report["comparisons"][key] = row
            save()
            print("COMPARE", key, json.dumps(row), flush=True)
        if not args.allow_divergence and not all(row["equal"] for row in report["comparisons"].values()):
            raise RuntimeError("Long mixed execution diverged; inspect first-mismatch diagnostics")
    finally:
        if reference_runner is not None:
            reference_runner.close()
        if runner is not None:
            runner.close()
        engine.terminate()


if __name__ == "__main__":
    main()
