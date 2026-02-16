#!/usr/bin/env python
"""
Evaluate Whisper model WER on LibriSpeech test-clean.

Usage:
    cd recipes/whisper/eval
    bash download_librispeech.sh
    python eval_librispeech.py --model_path $EOLE_MODEL_DIR/whisper-base-en-eole

Requires: jiwer, whisper-normalizer
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
from collections import defaultdict

import jiwer
from whisper_normalizer.english import EnglishTextNormalizer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Whisper WER on LibriSpeech test-clean")
    parser.add_argument("--model_path", required=True, help="Path to converted eole Whisper model")
    parser.add_argument(
        "--data_dir",
        default="./LibriSpeech/test-clean",
        help="Path to LibriSpeech test-clean directory",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID (-1 for CPU)")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam search width")
    parser.add_argument(
        "--condition_on_previous_text",
        action="store_true",
        help="Condition next segment on previous segment's text",
    )
    parser.add_argument(
        "--output_dir",
        default="./results",
        help="Directory for output files",
    )
    return parser.parse_args()


def load_references(data_dir):
    """Load reference transcripts from LibriSpeech trans.txt files.

    Returns:
        audio_files: list of FLAC file paths (sorted)
        references: dict mapping FLAC path to reference text
        speakers: dict mapping FLAC path to speaker ID
    """
    trans_files = sorted(glob.glob(os.path.join(data_dir, "*", "*", "*.trans.txt")))
    if not trans_files:
        print(f"Error: No transcript files found in {data_dir}")
        print("Run download_librispeech.sh first.")
        sys.exit(1)

    audio_files = []
    references = {}
    speakers = {}

    for trans_file in trans_files:
        chapter_dir = os.path.dirname(trans_file)
        with open(trans_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                utt_id = parts[0]
                text = parts[1] if len(parts) > 1 else ""
                flac_path = os.path.join(chapter_dir, f"{utt_id}.flac")
                if os.path.exists(flac_path):
                    audio_files.append(flac_path)
                    references[flac_path] = text
                    # Speaker ID is first part of utterance ID
                    speakers[flac_path] = utt_id.split("-")[0]

    audio_files.sort()
    return audio_files, references, speakers


def run_inference(audio_files, args):
    """Run eole predict on the audio files."""
    os.makedirs(args.output_dir, exist_ok=True)

    # Write audio file list
    src_file = os.path.join(args.output_dir, "audio_files.txt")
    with open(src_file, "w") as f:
        for path in audio_files:
            f.write(os.path.abspath(path) + "\n")

    output_file = os.path.join(args.output_dir, "hyp.txt")
    config_path = os.path.join(os.path.dirname(__file__), "eval_config.yaml")

    eole_bin = shutil.which("eole")
    if eole_bin is None:
        print("Error: 'eole' command not found. Is eole installed in the active environment?")
        sys.exit(1)
    cmd = [
        eole_bin,
        "predict",
        "-config",
        config_path,
        "-model_path",
        args.model_path,
        "-src",
        src_file,
        "-output",
        output_file,
        "-beam_size",
        str(args.beam_size),
    ]

    if args.gpu >= 0:
        cmd.extend(["-gpu_ranks", str(args.gpu)])

    if args.condition_on_previous_text:
        cmd.extend(["-condition_on_previous_text"])

    print(f"Running inference on {len(audio_files)} files...")
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Inference failed:\n{result.stderr}")
        sys.exit(1)

    # Read hypotheses
    with open(output_file) as f:
        hypotheses = [line.strip() for line in f]

    if len(hypotheses) != len(audio_files):
        print(f"Warning: got {len(hypotheses)} hypotheses for {len(audio_files)} audio files")

    return hypotheses


def compute_wer(references, hypotheses, normalizer):
    """Compute WER with text normalisation."""
    refs_norm = [normalizer(r) for r in references]
    hyps_norm = [normalizer(h) for h in hypotheses]

    # Filter out pairs where the normalised reference is empty
    pairs = [(r, h) for r, h in zip(refs_norm, hyps_norm) if r.strip()]
    if not pairs:
        return 0.0, [], []

    refs_filtered, hyps_filtered = zip(*pairs)
    refs_filtered = list(refs_filtered)
    hyps_filtered = list(hyps_filtered)

    score = jiwer.wer(refs_filtered, hyps_filtered) * 100
    return score, refs_filtered, hyps_filtered


def main():
    args = parse_args()

    # Load dataset
    audio_files, references, speakers = load_references(args.data_dir)
    print(f"Loaded {len(audio_files)} utterances from {args.data_dir}")

    # Count unique speakers
    unique_speakers = set(speakers.values())
    print(f"Speakers: {len(unique_speakers)}")

    # Run inference
    hypotheses = run_inference(audio_files, args)

    # Prepare reference list in same order as audio_files
    ref_list = [references[f] for f in audio_files]
    hyp_list = hypotheses[: len(audio_files)]

    # Compute overall WER
    normalizer = EnglishTextNormalizer()
    overall_wer, refs_norm, hyps_norm = compute_wer(ref_list, hyp_list, normalizer)

    print(f"\n{'='*60}")
    print(f"LibriSpeech test-clean WER: {overall_wer:.2f}%")
    print(f"{'='*60}")

    # Per-speaker breakdown
    speaker_refs = defaultdict(list)
    speaker_hyps = defaultdict(list)
    for f, ref, hyp in zip(audio_files, ref_list, hyp_list):
        spk = speakers[f]
        speaker_refs[spk].append(ref)
        speaker_hyps[spk].append(hyp)

    print(f"\nPer-speaker WER ({len(unique_speakers)} speakers):")
    print(f"{'Speaker':<10} {'Utts':>5} {'WER':>8}")
    print(f"{'-'*25}")

    speaker_wers = []
    for spk in sorted(speaker_refs.keys()):
        spk_wer, _, _ = compute_wer(speaker_refs[spk], speaker_hyps[spk], normalizer)
        speaker_wers.append((spk, len(speaker_refs[spk]), spk_wer))
        print(f"{spk:<10} {len(speaker_refs[spk]):>5} {spk_wer:>7.2f}%")

    # Save results
    results_file = os.path.join(args.output_dir, "wer_results.txt")
    with open(results_file, "w") as f:
        f.write(f"LibriSpeech test-clean WER: {overall_wer:.2f}%\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Beam size: {args.beam_size}\n")
        f.write(f"Utterances: {len(audio_files)}\n")
        f.write(f"Speakers: {len(unique_speakers)}\n")
        f.write(f"\nPer-speaker:\n")
        for spk, n, w in speaker_wers:
            f.write(f"  {spk}: {w:.2f}% ({n} utts)\n")

    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
