#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from comet import download_model, load_from_checkpoint

from eole.models.model import EncoderScoringModel
from eole.scorers.eole_comet import EoleCometScorer, EoleCometKiwiScorer


def parse_args():
    p = argparse.ArgumentParser(description="Compare unbabel-comet and native EOLE COMET parity.")
    p.add_argument("--src", required=True, help="Source text file (one segment per line)")
    p.add_argument("--mt", required=True, help="MT hypothesis file (one segment per line)")
    p.add_argument("--ref", default=None, help="Reference file for reference-based COMET")
    p.add_argument("--hf-model", required=True, help="Baseline model ID (e.g. Unbabel/wmt22-cometkiwi-da)")
    p.add_argument("--eole-model", required=True, help="Converted EOLE COMET model directory")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output", default=None, help="Optional path for JSON report")
    return p.parse_args()


def read_lines(path):
    return Path(path).read_text(encoding="utf-8").splitlines()


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

    baseline_out = baseline.predict(data, batch_size=args.batch_size, gpus=0, num_workers=1, progress_bar=False)
    eole_system = eole_scorer.compute_score(mt, ref, src)

    runtime = EncoderScoringModel.from_model_dir(args.eole_model, device="cpu")
    tokenizer = eole_scorer._build_sp_transform(runtime, args.eole_model)
    src_ids = eole_scorer._encode_texts(src, tokenizer, runtime)
    mt_ids = eole_scorer._encode_texts(mt, tokenizer, runtime)
    ref_ids = eole_scorer._encode_texts(ref, tokenizer, runtime) if ref is not None else None
    runtime_rows = []
    for i in range(len(src)):
        row = {"src_ids": src_ids[i], "mt_ids": mt_ids[i]}
        if ref_ids is not None:
            row["ref_ids"] = ref_ids[i]
        runtime_rows.append(row)
    eole_scores = []
    for i in range(0, len(runtime_rows), args.batch_size):
        eole_scores.extend(runtime.predict_scores(runtime_rows[i : i + args.batch_size]).tolist())

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

    print(json.dumps(report, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
