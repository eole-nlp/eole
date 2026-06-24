#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from comet import download_model, load_from_checkpoint

from eole.models.model import EncoderScoringModel
from eole.scorers.eole_comet import EoleCometScorer, EoleCometKiwiScorer
from eole.utils.scorer import build_segment_rows


def parse_args():
    p = argparse.ArgumentParser(description="Compare unbabel-comet and native EOLE COMET parity.")
    p.add_argument("--src", required=True, help="Source text file (one segment per line)")
    p.add_argument("--mt", required=True, help="MT hypothesis file (one segment per line)")
    p.add_argument("--ref", default=None, help="Reference file for reference-based COMET")
    p.add_argument("--hf-model", required=True, help="Baseline model ID (e.g. Unbabel/wmt22-cometkiwi-da)")
    p.add_argument("--eole-model", required=True, help="Converted EOLE COMET model directory")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cpu", help="Device for native EOLE scoring, e.g. cpu, mps, cuda:0")
    p.add_argument("--num-workers", type=int, default=1, help="DataLoader workers for Unbabel COMET baseline")
    p.add_argument("--output", default=None, help="Optional path for JSON report")
    return p.parse_args()


def read_lines(path):
    return Path(path).read_text(encoding="utf-8").splitlines()


def _as_list(value):
    if value is None:
        return None
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _metadata_get(metadata, key):
    if metadata is None:
        return None
    if hasattr(metadata, key):
        return getattr(metadata, key)
    if isinstance(metadata, dict):
        return metadata.get(key)
    return None


def _span_mismatches(baseline_spans, eole_spans, limit=10):
    mismatches = []
    for idx, (baseline_sentence, eole_sentence) in enumerate(zip(baseline_spans, eole_spans)):
        if len(baseline_sentence) != len(eole_sentence):
            mismatches.append(
                {
                    "index": idx,
                    "reason": "span_count",
                    "baseline": baseline_sentence,
                    "eole": eole_sentence,
                }
            )
            if len(mismatches) >= limit:
                break
            continue
        for span_idx, (baseline_span, eole_span) in enumerate(zip(baseline_sentence, eole_sentence)):
            baseline_core = {k: baseline_span.get(k) for k in ["text", "severity", "start", "end"]}
            eole_core = {k: eole_span.get(k) for k in ["text", "severity", "start", "end"]}
            if baseline_core != eole_core:
                mismatches.append(
                    {
                        "index": idx,
                        "span_index": span_idx,
                        "reason": "span_fields",
                        "baseline": baseline_span,
                        "eole": eole_span,
                    }
                )
                if len(mismatches) >= limit:
                    return mismatches
    if len(baseline_spans) != len(eole_spans):
        mismatches.append(
            {
                "reason": "sentence_count",
                "baseline_count": len(baseline_spans),
                "eole_count": len(eole_spans),
            }
        )
    return mismatches


def _numeric_diff(a, b):
    if a is None or b is None:
        return None
    dif = [abs(float(x) - float(y)) for x, y in zip(a, b)]
    if not dif:
        return {"mae": 0.0, "max_abs": 0.0}
    return {"mae": float(sum(dif) / len(dif)), "max_abs": float(max(dif))}


def main():
    args = parse_args()
    src = read_lines(args.src)
    mt = read_lines(args.mt)
    ref = read_lines(args.ref) if args.ref else None
    if len(src) != len(mt):
        raise ValueError("src and mt files must have same number of lines")
    if ref is not None and len(ref) != len(src):
        raise ValueError("ref must have same number of lines as src")

    baseline = load_from_checkpoint(download_model(args.hf_model))
    if ref is None:
        data = [{"src": s, "mt": m} for s, m in zip(src, mt)]
        eole_scorer = EoleCometKiwiScorer(
            SimpleNamespace(comet_model=args.eole_model, comet_batch_size=args.batch_size)
        )
    else:
        data = [{"src": s, "mt": m, "ref": r} for s, m, r in zip(src, mt, ref)]
        eole_scorer = EoleCometScorer(SimpleNamespace(comet_model=args.eole_model, comet_batch_size=args.batch_size))

    baseline_out = baseline.predict(
        data, batch_size=args.batch_size, gpus=0, num_workers=args.num_workers, progress_bar=False
    )

    runtime = EncoderScoringModel.from_model_dir(args.eole_model, device=args.device)
    tokenizer = eole_scorer._build_sp_transform(runtime, args.eole_model)
    runtime_rows = build_segment_rows(mt, src, ref, tokenizer, runtime, encode_texts_fn=eole_scorer._encode_texts)
    eole_scores = []
    eole_metadata = None
    for i in range(0, len(runtime_rows), args.batch_size):
        chunk = runtime_rows[i : i + args.batch_size]
        if getattr(runtime, "class_identifier", None) == "xcomet_metric":
            chunk_out = runtime.predict_xcomet(chunk)
            eole_scores.extend(chunk_out["scores"])
            if eole_metadata is None:
                eole_metadata = {key: [] for key in chunk_out["metadata"]}
            for key, value in chunk_out["metadata"].items():
                eole_metadata[key].extend(value)
        else:
            eole_scores.extend(runtime.predict_scores(chunk).tolist())
    eole_system = float(sum(eole_scores) / len(eole_scores))

    dif = [abs(a - b) for a, b in zip(baseline_out.scores, eole_scores)]
    report = {
        "hf_model": args.hf_model,
        "eole_model": args.eole_model,
        "n": len(src),
        "baseline_system": float(baseline_out.system_score),
        "eole_system": float(eole_system),
        "delta_system": float(abs(baseline_out.system_score - eole_system)),
        "sentence_mae": float(sum(dif) / len(dif)),
        "sentence_max_abs": float(max(dif)),
    }

    if getattr(runtime, "class_identifier", None) == "xcomet_metric":
        baseline_metadata = getattr(baseline_out, "metadata", None)
        baseline_spans = _metadata_get(baseline_metadata, "error_spans") or []
        eole_spans = (eole_metadata or {}).get("error_spans", [])
        baseline_mqm = _as_list(_metadata_get(baseline_metadata, "mqm_scores"))
        eole_mqm = (eole_metadata or {}).get("mqm_scores")
        report["xcomet"] = {
            "mqm_scores": _numeric_diff(baseline_mqm, eole_mqm),
            "span_sentence_mismatches": len(_span_mismatches(baseline_spans, eole_spans, limit=len(src))),
            "span_mismatch_examples": _span_mismatches(baseline_spans, eole_spans),
        }

    print(json.dumps(report, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
