#!/usr/bin/env python
"""
Evaluate Whisper model WER/CER on FLEURS multilingual benchmark.

Usage:
    cd recipes/whisper/eval
    python eval_fleurs.py --model_path $MODEL --languages hi_in
    python eval_fleurs.py --model_path $MODEL --languages hi_in,de_de,fr_fr
    python eval_fleurs.py --model_path $MODEL --languages all

Requires: datasets, jiwer, whisper-normalizer
    pip install eole[wer]
"""

import argparse
import os
import shutil
import subprocess
import sys
import wave
from collections import defaultdict

import jiwer
from whisper_normalizer.basic import BasicTextNormalizer
from whisper_normalizer.english import EnglishTextNormalizer

# FLEURS language code -> Whisper language code
# Only languages supported by Whisper are included.
FLEURS_TO_WHISPER = {
    "af_za": "af",
    "am_et": "am",
    "ar_eg": "ar",
    "as_in": "as",
    "az_az": "az",
    "be_by": "be",
    "bg_bg": "bg",
    "bn_in": "bn",
    "bs_ba": "bs",
    "ca_es": "ca",
    "cs_cz": "cs",
    "cy_gb": "cy",
    "da_dk": "da",
    "de_de": "de",
    "el_gr": "el",
    "en_us": "en",
    "es_419": "es",
    "et_ee": "et",
    "fa_ir": "fa",
    "fi_fi": "fi",
    "fil_ph": "tl",
    "fr_fr": "fr",
    "ga_ie": "ga",
    "gl_es": "gl",
    "gu_in": "gu",
    "ha_ng": "ha",
    "he_il": "he",
    "hi_in": "hi",
    "hr_hr": "hr",
    "hu_hu": "hu",
    "hy_am": "hy",
    "id_id": "id",
    "is_is": "is",
    "it_it": "it",
    "ja_jp": "ja",
    "jv_id": "jw",
    "ka_ge": "ka",
    "kk_kz": "kk",
    "km_kh": "km",
    "kn_in": "kn",
    "ko_kr": "ko",
    "lo_la": "lo",
    "lt_lt": "lt",
    "lv_lv": "lv",
    "mi_nz": "mi",
    "mk_mk": "mk",
    "ml_in": "ml",
    "mn_mn": "mn",
    "mr_in": "mr",
    "ms_my": "ms",
    "mt_mt": "mt",
    "my_mm": "my",
    "nb_no": "no",
    "ne_np": "ne",
    "nl_nl": "nl",
    "cmn_hans_cn": "zh",
    "pa_in": "pa",
    "pl_pl": "pl",
    "ps_af": "ps",
    "pt_br": "pt",
    "ro_ro": "ro",
    "ru_ru": "ru",
    "sd_in": "sd",
    "sk_sk": "sk",
    "sl_si": "sl",
    "sn_zw": "sn",
    "so_so": "so",
    "sr_rs": "sr",
    "sv_se": "sv",
    "sw_ke": "sw",
    "ta_in": "ta",
    "te_in": "te",
    "tg_tj": "tg",
    "th_th": "th",
    "tr_tr": "tr",
    "uk_ua": "uk",
    "ur_pk": "ur",
    "uz_uz": "uz",
    "vi_vn": "vi",
    "yo_ng": "yo",
    "yue_hant_hk": "yue",
}

# Languages that use CER instead of WER (character-based scripts)
CER_LANGUAGES = {"zh", "ja", "ko", "yue"}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Whisper WER/CER on FLEURS multilingual benchmark")
    parser.add_argument("--model_path", required=True, help="Path to converted eole Whisper model")
    parser.add_argument(
        "--languages",
        required=True,
        help="Comma-separated FLEURS language codes (e.g. hi_in,de_de) or 'all'",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "dev", "train"],
        help="FLEURS split to evaluate (default: test)",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID (-1 for CPU)")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam search width")
    parser.add_argument("--output_dir", default="./results", help="Output root directory")
    parser.add_argument(
        "--audio_cache_dir",
        default="./fleurs_audio_cache",
        help="Directory for cached WAV files",
    )
    parser.add_argument(
        "--task",
        default=None,
        choices=["transcribe", "translate"],
        help="Task: 'transcribe' or 'translate'. If not set, uses config default.",
    )
    parser.add_argument(
        "--condition_on_previous_text",
        action="store_true",
        help="Condition next segment on previous segment's text",
    )
    return parser.parse_args()


def write_wav(path, audio_array, sample_rate):
    """Write a float32 audio array to a 16-bit PCM WAV file.

    audio_array can be a list, numpy array, or torch tensor of float32 samples.
    """
    import numpy as np

    arr = np.asarray(audio_array, dtype=np.float32).ravel()
    np.clip(arr, -1.0, 1.0, out=arr)
    pcm = (arr * 32767).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def _extract_audio(audio_obj):
    """Extract (array, sample_rate) from a HuggingFace audio object.

    Handles both the legacy dict format (datasets <4.5) and the newer
    AudioDecoder format (datasets >=4.5 with torchcodec).
    """
    if isinstance(audio_obj, dict):
        return audio_obj["array"], audio_obj["sampling_rate"]
    samples = audio_obj.get_all_samples()
    return samples.data.squeeze(0), samples.sample_rate


def load_fleurs_data(fleurs_lang, split, audio_cache_dir):
    """Load FLEURS dataset and cache audio as WAV files.

    Returns:
        audio_files: list of WAV file paths (sorted by ID)
        references: dict mapping WAV path to reference transcription
        genders: dict mapping WAV path to gender (0=male, 1=female)
    """
    from datasets import load_dataset

    cache_dir = os.path.join(audio_cache_dir, fleurs_lang, split)
    os.makedirs(cache_dir, exist_ok=True)

    hf_split = "validation" if split == "dev" else split

    print(f"Loading FLEURS {fleurs_lang}/{hf_split} from HuggingFace...")
    ds = load_dataset(
        "google/fleurs",
        data_dir=fleurs_lang,
        split=hf_split,
        revision="refs/convert/parquet",
    )

    existing = [f for f in os.listdir(cache_dir) if f.endswith(".wav")]
    need_write = len(existing) != len(ds)

    audio_files = []
    references = {}
    genders = {}

    for i, example in enumerate(ds):
        wav_path = os.path.join(cache_dir, f"{i:05d}.wav")
        audio_files.append(wav_path)
        references[wav_path] = example["transcription"]
        genders[wav_path] = example.get("gender", -1)

        if need_write:
            array, sr = _extract_audio(example["audio"])
            write_wav(wav_path, array, sr)

    if need_write:
        print(f"  Wrote {len(audio_files)} WAV files to {cache_dir}")
    else:
        print(f"  Using cached audio ({len(audio_files)} files in {cache_dir})")

    audio_files.sort()
    return audio_files, references, genders


def run_inference(audio_files, whisper_lang, args, lang_output_dir):
    """Run eole predict on the audio files for a given language."""
    os.makedirs(lang_output_dir, exist_ok=True)

    src_file = os.path.join(lang_output_dir, "audio_files.txt")
    with open(src_file, "w") as f:
        for path in audio_files:
            f.write(os.path.abspath(path) + "\n")

    output_file = os.path.join(lang_output_dir, "hyp.txt")
    config_path = os.path.join(os.path.dirname(__file__), "eval_config.yaml")

    eole_bin = shutil.which("eole")
    if eole_bin is None:
        print("Error: 'eole' command not found. Is eole installed?")
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
        "-language",
        whisper_lang,
    ]

    if args.gpu >= 0:
        cmd.extend(["-gpu_ranks", str(args.gpu)])

    if args.condition_on_previous_text:
        cmd.extend(["-condition_on_previous_text"])

    if args.task:
        cmd.extend(["-task", args.task])

    print(f"Running inference on {len(audio_files)} files (lang={whisper_lang})...")
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Inference failed:\n{result.stderr}")
        sys.exit(1)

    with open(output_file) as f:
        hypotheses = [line.strip() for line in f]

    if len(hypotheses) != len(audio_files):
        print(f"Warning: got {len(hypotheses)} hypotheses for {len(audio_files)} audio files")

    return hypotheses


def compute_wer(references, hypotheses, normalizer, metric_fn):
    """Compute WER or CER with text normalisation."""
    refs_norm = [normalizer(r) for r in references]
    hyps_norm = [normalizer(h) for h in hypotheses]

    pairs = [(r, h) for r, h in zip(refs_norm, hyps_norm) if r.strip()]
    if not pairs:
        return 0.0, [], []

    refs_filtered, hyps_filtered = zip(*pairs)
    refs_filtered = list(refs_filtered)
    hyps_filtered = list(hyps_filtered)

    score = metric_fn(refs_filtered, hyps_filtered) * 100
    return score, refs_filtered, hyps_filtered


def evaluate_language(fleurs_lang, whisper_lang, args):
    """Evaluate a single language: load data, run inference, compute metrics.

    Returns:
        dict with language results, or None on failure.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {fleurs_lang} (Whisper code: {whisper_lang})")
    print(f"{'='*60}")

    audio_files, references, genders = load_fleurs_data(fleurs_lang, args.split, args.audio_cache_dir)
    print(f"  {len(audio_files)} utterances loaded")

    lang_output_dir = os.path.join(args.output_dir, fleurs_lang)

    hypotheses = run_inference(audio_files, whisper_lang, args, lang_output_dir)

    use_cer = whisper_lang in CER_LANGUAGES
    metric_name = "CER" if use_cer else "WER"
    metric_fn = jiwer.cer if use_cer else jiwer.wer

    if whisper_lang == "en":
        normalizer = EnglishTextNormalizer()
    else:
        normalizer = BasicTextNormalizer()

    ref_list = [references[f] for f in audio_files]
    hyp_list = hypotheses[: len(audio_files)]
    overall_score, refs_norm, hyps_norm = compute_wer(ref_list, hyp_list, normalizer, metric_fn)

    print(f"\n  FLEURS {fleurs_lang} {metric_name}: {overall_score:.2f}%")

    gender_refs = defaultdict(list)
    gender_hyps = defaultdict(list)
    for f, ref, hyp in zip(audio_files, ref_list, hyp_list):
        g = genders.get(f, -1)
        gender_refs[g].append(ref)
        gender_hyps[g].append(hyp)

    gender_labels = {0: "male", 1: "female", -1: "unknown"}
    gender_results = []
    for g in sorted(gender_refs.keys()):
        g_score, _, _ = compute_wer(gender_refs[g], gender_hyps[g], normalizer, metric_fn)
        label = gender_labels.get(g, f"gender_{g}")
        n = len(gender_refs[g])
        gender_results.append((label, n, g_score))
        print(f"    {label}: {g_score:.2f}% ({n} utts)")

    results_file = os.path.join(lang_output_dir, "wer_results.txt")
    with open(results_file, "w") as f:
        f.write(f"FLEURS {fleurs_lang} ({whisper_lang}) {metric_name}: {overall_score:.2f}%\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Beam size: {args.beam_size}\n")
        f.write(f"Utterances: {len(audio_files)}\n")
        f.write("\nPer-gender:\n")
        for label, n, g_score in gender_results:
            f.write(f"  {label}: {g_score:.2f}% ({n} utts)\n")
    print(f"  Results saved to {results_file}")

    return {
        "fleurs_lang": fleurs_lang,
        "whisper_lang": whisper_lang,
        "metric": metric_name,
        "score": overall_score,
        "n_utterances": len(audio_files),
        "gender_results": gender_results,
    }


def main():
    args = parse_args()

    if args.languages == "all":
        languages = list(FLEURS_TO_WHISPER.keys())
    else:
        languages = [lang.strip() for lang in args.languages.split(",")]

    eval_pairs = []
    for fleurs_lang in languages:
        if fleurs_lang in FLEURS_TO_WHISPER:
            eval_pairs.append((fleurs_lang, FLEURS_TO_WHISPER[fleurs_lang]))
        else:
            print(f"Warning: '{fleurs_lang}' not in FLEURS→Whisper mapping, skipping")

    if not eval_pairs:
        print("Error: No valid languages to evaluate.")
        sys.exit(1)

    print(f"Evaluating {len(eval_pairs)} language(s): {', '.join(f for f, _ in eval_pairs)}")

    results = []
    for fleurs_lang, whisper_lang in eval_pairs:
        result = evaluate_language(fleurs_lang, whisper_lang, args)
        if result:
            results.append(result)

    if len(results) > 1:
        print(f"\n{'='*60}")
        print("FLEURS Evaluation Summary")
        print(f"{'='*60}")
        print(f"{'Language':<16} {'Whisper':<8} {'Metric':<6} {'Score':>8} {'Utts':>6}")
        print(f"{'-'*46}")
        for r in results:
            print(
                f"{r['fleurs_lang']:<16} {r['whisper_lang']:<8} {r['metric']:<6} "
                f"{r['score']:>7.2f}% {r['n_utterances']:>6}"
            )

        avg_score = sum(r["score"] for r in results) / len(results)
        print(f"{'-'*46}")
        print(f"{'Average':<31} {avg_score:>7.2f}%")

    if results:
        os.makedirs(args.output_dir, exist_ok=True)
        summary_file = os.path.join(args.output_dir, "fleurs_summary.txt")
        with open(summary_file, "w") as f:
            f.write("FLEURS Evaluation Summary\n")
            f.write(f"Model: {args.model_path}\n")
            f.write(f"Split: {args.split}\n")
            f.write(f"Beam size: {args.beam_size}\n\n")
            f.write(f"{'Language':<16} {'Whisper':<8} {'Metric':<6} {'Score':>8} {'Utts':>6}\n")
            f.write(f"{'-'*46}\n")
            for r in results:
                f.write(
                    f"{r['fleurs_lang']:<16} {r['whisper_lang']:<8} {r['metric']:<6} "
                    f"{r['score']:>7.2f}% {r['n_utterances']:>6}\n"
                )
            if len(results) > 1:
                avg_score = sum(r["score"] for r in results) / len(results)
                f.write(f"{'-'*46}\n")
                f.write(f"{'Average':<31} {avg_score:>7.2f}%\n")
        print(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    main()
