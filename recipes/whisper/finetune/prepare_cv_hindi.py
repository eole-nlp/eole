#!/usr/bin/env python3
"""Prepare Common Voice Hindi data for EOLE Whisper fine-tuning.

Extracts the Common Voice Hindi tar.gz archive, parses TSV metadata,
and creates parallel audio-path / transcription files for training
and validation.

Following the HuggingFace Whisper fine-tuning blog approach:
- train + dev splits combined for training (low-resource scenario)
- test split used for validation

Usage:
    python recipes/whisper/finetune/prepare_cv_hindi.py
"""

import csv
import os
import sys
import tarfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = os.path.join(SCRIPT_DIR, "cv-corpus-24.0-2025-12-05-hi.tar.gz")
EXTRACT_DIR = os.path.join(SCRIPT_DIR, "cv-hi")

# Whisper decoder prefix tokens for Hindi transcription
DECODER_PREFIX = "<|hi|><|transcribe|><|notimestamps|>"


def extract_archive():
    """Extract tar.gz if not already extracted."""
    if os.path.exists(EXTRACT_DIR):
        print(f"Already extracted: {EXTRACT_DIR}")
        return
    print(f"Extracting {ARCHIVE} ...")
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        tar.extractall(path=EXTRACT_DIR)
    print("Done.")


def find_data_root():
    """Find the actual data directory inside the extracted archive."""
    # Common Voice archives typically extract to cv-corpus-VERSION/LANG/
    for root, dirs, files in os.walk(EXTRACT_DIR):
        if "train.tsv" in files:
            return root
    raise FileNotFoundError("Could not find train.tsv in extracted archive")


def parse_tsv(tsv_path, clips_dir):
    """Parse a Common Voice TSV file and return (audio_paths, transcripts).

    Filters out rows with empty sentence fields.
    Audio paths are made absolute for reproducible training.
    """
    audio_paths = []
    transcripts = []
    skipped = 0
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sentence = row.get("sentence", "").strip()
            path = row.get("path", "").strip()
            if not sentence or not path:
                skipped += 1
                continue
            audio_path = os.path.abspath(os.path.join(clips_dir, path))
            audio_paths.append(audio_path)
            transcripts.append(f"{DECODER_PREFIX}{sentence}<|endoftext|>")
    if skipped > 0:
        print(f"  Skipped {skipped} rows with empty sentence/path")
    return audio_paths, transcripts


def write_files(audio_paths, transcripts, prefix):
    """Write audio paths and transcripts to output files."""
    audio_file = os.path.join(SCRIPT_DIR, f"{prefix}_audio_paths.txt")
    text_file = os.path.join(SCRIPT_DIR, f"{prefix}_transcripts.txt")
    with open(audio_file, "w", encoding="utf-8") as f:
        for p in audio_paths:
            f.write(p + "\n")
    with open(text_file, "w", encoding="utf-8") as f:
        for t in transcripts:
            f.write(t + "\n")
    print(f"  {audio_file}: {len(audio_paths)} lines")
    print(f"  {text_file}: {len(transcripts)} lines")


def verify_tokenization():
    """Verify that special tokens are encoded as single token IDs."""
    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("\nSkipping tokenization verification (tokenizers not installed)")
        return

    model_dir = os.environ.get("EOLE_MODEL_DIR", "")
    tokenizer_path = None
    for model_name in ["whisper-small-eole", "whisper-base-eole"]:
        candidate = os.path.join(model_dir, model_name, "tokenizer.json")
        if os.path.exists(candidate):
            tokenizer_path = candidate
            break

    if tokenizer_path is None:
        print("\nSkipping tokenization verification (no whisper tokenizer found)")
        return

    print(f"\nVerifying tokenization with {tokenizer_path} ...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    test_text = f"{DECODER_PREFIX}नमस्ते दुनिया"
    encoded = tokenizer.encode(test_text)

    print(f"  Input:  {test_text}")
    print(f"  Tokens: {encoded.tokens[:8]}...")
    print(f"  IDs:    {encoded.ids[:8]}...")

    # Verify each prefix token encodes as a single ID
    ok = True
    for token_text in ["<|hi|>", "<|transcribe|>", "<|notimestamps|>"]:
        enc = tokenizer.encode(token_text)
        if len(enc.ids) == 1:
            print(f"  OK: {token_text} -> ID {enc.ids[0]}")
        else:
            print(f"  WARNING: {token_text} -> {len(enc.ids)} tokens: {enc.tokens}")
            ok = False
    if ok:
        print("  All prefix tokens encode as single IDs.")


def main():
    if not os.path.exists(ARCHIVE):
        print(f"Error: Archive not found: {ARCHIVE}")
        print(f"Expected at: {ARCHIVE}")
        sys.exit(1)

    extract_archive()
    data_root = find_data_root()
    clips_dir = os.path.join(data_root, "clips")
    print(f"Data root: {data_root}")
    print(f"Clips dir: {clips_dir}")

    # Parse splits
    print("\nParsing TSV files...")
    train_paths, train_texts = parse_tsv(os.path.join(data_root, "train.tsv"), clips_dir)
    dev_paths, dev_texts = parse_tsv(os.path.join(data_root, "dev.tsv"), clips_dir)
    test_paths, test_texts = parse_tsv(os.path.join(data_root, "test.tsv"), clips_dir)

    print("\nSplit sizes:")
    print(f"  train: {len(train_paths)}")
    print(f"  dev:   {len(dev_paths)}")
    print(f"  test:  {len(test_paths)}")

    # Combine train + dev for training
    combined_paths = train_paths + dev_paths
    combined_texts = train_texts + dev_texts
    print(f"  train+dev (combined for training): {len(combined_paths)}")

    # Write output files
    print("\nWriting output files:")
    write_files(combined_paths, combined_texts, "train")
    write_files(test_paths, test_texts, "valid")

    verify_tokenization()
    print("\nDone! Data files ready for training.")


if __name__ == "__main__":
    main()
