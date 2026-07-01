from types import SimpleNamespace
import os

import torch

from eole.config.config import Config
from eole.config.data import NestedAllTransformsConfig
from eole.transforms import get_transforms_cls, make_transforms
from eole.transforms.tokenize import BaseTokenizerConfig
from eole.utils.misc import clear_gpu_cache


SCORER_INFERENCE_DEFAULTS = {
    "transforms": [],
    "optional_eos": [],
    "dynamic_shapes": None,
    "n_best": 1,
    "max_length": 0,
    "max_length_ratio": 0,
    "context_length": 0,
    "beam_size": 1,
    "temperature": 1.0,
    "top_k": 0,
    "top_p": 0.0,
    "min_length": 0,
    "ban_unk_token": False,
    "ratio": 0.0,
    "stepwise_penalty": False,
    "dump_beam": "",
    "block_ngram_repeat": 0,
    "ignore_when_blocking": [],
    "replace_unk": False,
    "tgt_file_prefix": False,
    "phrase_table": "",
    "data_type": "text",
    "verbose": False,
    "report_align": False,
    "gold_align": False,
    "seed": -1,
    "estim_only": True,
    "self_attn_backend": "pytorch",
    "fuse_kvq": False,
    "fuse_gate": False,
}


def apply_scorer_inference_defaults(config, model_config, global_scorer=None):
    defaults = dict(SCORER_INFERENCE_DEFAULTS)
    defaults["report_time"] = bool(getattr(config, "report_time", False))
    defaults["with_score"] = bool(getattr(config, "with_score", False))
    for key, value in defaults.items():
        if not hasattr(config, key):
            setattr(config, key, value)
    if not hasattr(model_config, "add_estimator"):
        setattr(model_config, "add_estimator", True)
    if not hasattr(model_config, "estimator_type"):
        setattr(model_config, "estimator_type", "first_token")
    return global_scorer or SimpleNamespace(has_cov_pen=False)


def read_score_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def score_level_output(all_scores, score_level="segment", report_time=False, logger=None):
    if score_level == "system":
        system_scores = [sum(all_scores) / len(all_scores)] if all_scores else []
        if report_time and logger is not None:
            logger.info("Emitted 1 system score")
        return [[]], [system_scores], [[]]

    if report_time and logger is not None:
        logger.info("Emitted %d segment score%s", len(all_scores), "" if len(all_scores) == 1 else "s")
    return [[]], [all_scores], [[]]


class ScorerTransformConfig(Config):
    share_vocab: bool = True
    seed: int = 0
    transforms_configs: NestedAllTransformsConfig


def build_scorer_sentencepiece_transform(
    model,
    model_dir,
    model_file="spiece.model",
    replace_newline_sentinel=True,
):
    tokenizer_model = os.path.join(model_dir, model_file)
    if not os.path.exists(tokenizer_model):
        raise FileNotFoundError(f"No {model_file} found in {model_dir}")

    full_config = ScorerTransformConfig(
        share_vocab=True,
        seed=0,
        transforms_configs=NestedAllTransformsConfig(
            sentencepiece=BaseTokenizerConfig(
                src_subword_model=tokenizer_model,
                tgt_subword_model=tokenizer_model,
                replace_newline_sentinel=replace_newline_sentinel,
                src_subword_nbest=1,
                tgt_subword_nbest=1,
                src_subword_alpha=0.0,
                tgt_subword_alpha=0.0,
                src_subword_vocab="",
                tgt_subword_vocab="",
                src_vocab_threshold=0,
                tgt_vocab_threshold=0,
            )
        ),
    )
    transforms = make_transforms(full_config, get_transforms_cls(["sentencepiece"]), model.vocabs)
    return transforms["sentencepiece"]


def build_template_inputs(src_lines, tgt_lines, ref_lines, input_templates, input_mode):
    if len(src_lines) != len(tgt_lines):
        raise ValueError(f"src and tgt line counts differ: {len(src_lines)} vs {len(tgt_lines)}")
    if input_mode not in input_templates:
        raise ValueError(f"No input template configured for input_mode={input_mode!r}.")

    template = input_templates[input_mode]
    needs_ref = "{ref}" in template
    if needs_ref:
        if ref_lines is None:
            raise ValueError(f"input_mode={input_mode!r} requires --ref.")
        if len(ref_lines) != len(tgt_lines):
            raise ValueError(
                f"src/tgt/ref line counts differ: {len(src_lines)} vs {len(tgt_lines)} vs {len(ref_lines)}"
            )

    refs = ref_lines if ref_lines is not None else [None] * len(tgt_lines)
    return [template.format(src=src, tgt=tgt, ref=ref) for src, tgt, ref in zip(src_lines, tgt_lines, refs)]


def encode_scorer_inputs(texts, tokenizer, model):
    encoded = []
    for text in texts:
        pieces = tokenizer.tokenize_string(text, side="src", is_train=False)
        ids = [model.vocabs["src"].lookup_token(piece) for piece in pieces]
        if getattr(model, "strip_eos", True) and ids and ids[-1] == model.eos_id:
            ids = ids[:-1]
        encoded.append(ids[: model.max_length])
    return encoded


def build_segment_rows(preds, texts_srcs, texts_refs, tokenizer, model, encode_texts_fn):
    if len(texts_srcs) != len(preds):
        raise ValueError(f"preds and texts_srcs lengths differ: {len(preds)} vs {len(texts_srcs)}")
    if texts_refs is not None and len(texts_refs) != len(preds):
        raise ValueError(f"preds and texts_refs lengths differ: {len(preds)} vs {len(texts_refs)}")

    requires_reference = bool(getattr(model, "requires_reference", False))
    input_segments = list(getattr(model, "input_segments", []))
    needs_ref_ids = requires_reference or "ref" in input_segments
    src_ids = encode_texts_fn(texts_srcs, tokenizer, model)
    mt_ids = encode_texts_fn(preds, tokenizer, model)
    ref_ids = encode_texts_fn(texts_refs, tokenizer, model) if needs_ref_ids and texts_refs is not None else None

    def _ids(item):
        return item["ids"] if isinstance(item, dict) else item

    rows = []
    for idx in range(len(preds)):
        mt = mt_ids[idx]
        row = {"src_ids": _ids(src_ids[idx]), "mt_ids": _ids(mt), "mt_text": preds[idx]}
        if isinstance(mt, dict):
            if "offsets" in mt:
                row["mt_offsets"] = mt["offsets"]
            if "decode_token_ids" in mt:
                row["decode_token_ids"] = mt["decode_token_ids"]
        if ref_ids is not None:
            row["ref_ids"] = _ids(ref_ids[idx])
        rows.append(row)
    return rows


def predict_scores_with_oom_retry(model, rows, batch_size, logger=None, progress_callback=None):
    batch_size = max(1, int(batch_size))
    all_scores = []
    i = 0
    while i < len(rows):
        current_bs = min(batch_size, len(rows) - i)
        chunk = rows[i : i + current_bs]
        try:
            scores = model.predict_scores(chunk)
            all_scores.extend(scores.tolist())
            i += current_bs
            if progress_callback is not None:
                progress_callback(i)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if not isinstance(e, torch.cuda.OutOfMemoryError) and "out of memory" not in str(e).lower():
                raise
            if current_bs == 1:
                raise
            clear_gpu_cache()
            batch_size = max(1, current_bs // 2)
            if logger is not None:
                logger.warning("OOM at batch %d; halving to %d", current_bs, batch_size)
    return all_scores
