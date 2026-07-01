"""Predictor that scores parallel (src, tgt) pairs with an EncoderScoringModel.

Plugs into InferenceEnginePY in place of Translator/Encoder/etc when the loaded
model has architecture="transformer_encoder_scorer". Reads --src and --tgt files,
tokenizes both sides via the recipe's `sentencepiece` transform, and writes one
score per line to --output.
"""

import time
import torch

from eole.predict.inference import Inference
from eole.utils.comet_scorer import (
    build_comet_sentencepiece_transform,
    encode_comet_texts_to_ids,
    encode_comet_texts_to_ids_and_offsets,
)
from eole.utils.scorer import (
    apply_scorer_inference_defaults,
    build_segment_rows,
    predict_scores_with_oom_retry,
    read_score_lines,
    score_level_output,
)
from eole.utils.logging import logger as _module_logger

logger = _module_logger


def _resolve_sentencepiece(transforms):
    if isinstance(transforms, dict) and "sentencepiece" in transforms:
        return transforms["sentencepiece"]
    if hasattr(transforms, "transforms"):
        for t in transforms.transforms:
            if hasattr(t, "tokenize_string"):
                return t
    if hasattr(transforms, "__iter__") and not isinstance(transforms, dict):
        for t in transforms:
            if hasattr(t, "tokenize_string"):
                return t
    return None


def _encode_texts_with_tokenizer(texts, tokenizer, model):
    bos_id = model.bos_id
    eos_id = model.eos_id
    max_length = model.max_length
    vocab = model.vocabs["src"]
    encoded = []
    for text in texts:
        pieces = tokenizer.tokenize_string(text, side="src", is_train=False)
        ids = [vocab.lookup_token(piece) for piece in pieces]
        encoded.append(([bos_id] + ids + [eos_id])[:max_length])
    return encoded


class EncoderScorer(Inference):
    def __init__(
        self,
        model,
        vocabs,
        config,
        model_config,
        device_id=0,
        global_scorer=None,
        report_score=True,
        logger=None,  # noqa: A002 — kept for engine API compat; module-level `logger` is used
        return_gold_log_probs=False,
    ):
        global_scorer = apply_scorer_inference_defaults(config, model_config, global_scorer)

        super().__init__(
            model,
            vocabs,
            config,
            model_config,
            device_id=device_id,
            global_scorer=global_scorer,
            report_score=report_score,
            logger=logger,
            return_gold_log_probs=return_gold_log_probs,
        )
        self.config = config

        compute_dtype = getattr(config, "compute_dtype", None)
        if compute_dtype in (torch.float16, torch.bfloat16):
            self.model.to(compute_dtype)
        param_dtype = None
        if hasattr(self.model, "parameters"):
            try:
                param_dtype = next(iter(self.model.parameters())).dtype
            except (StopIteration, TypeError):
                pass
        _module_logger.info(
            "EncoderScorer: batch_size=%s, compute_dtype=%s, model.dtype=%s",
            getattr(config, "batch_size", None),
            compute_dtype,
            param_dtype,
        )

    def update_settings(self, **kwargs):
        return

    def _predict(self, infer_iter, transforms, attn_debug, align_debug, streamer=None):
        if not self.config.src or not self.config.tgt:
            raise ValueError("EncoderScorer requires both --src and --tgt files.")
        if not getattr(self.config, "with_score", False):
            raise ValueError("EncoderScorer requires --with_score to emit estimator scores.")

        src_lines = read_score_lines(self.config.src)
        tgt_lines = read_score_lines(self.config.tgt)
        ref_lines = read_score_lines(self.config.ref) if getattr(self.config, "ref", None) else None
        if len(src_lines) != len(tgt_lines):
            raise ValueError(f"src and tgt line counts differ: {len(src_lines)} vs {len(tgt_lines)}")
        if ref_lines is not None and len(ref_lines) != len(tgt_lines):
            raise ValueError(
                f"src/tgt/ref line counts differ: {len(src_lines)} vs {len(tgt_lines)} vs {len(ref_lines)}"
            )
        if bool(getattr(self.model, "requires_reference", False)):
            if ref_lines is None:
                raise ValueError(
                    "This EncoderScorer model requires references. Provide --ref (with --tgt as hypothesis/mt)."
                )

        tokenizer = _resolve_sentencepiece(transforms)
        encode_texts_fn = _encode_texts_with_tokenizer if tokenizer is not None else None
        if tokenizer is None:
            scoring_type = getattr(self.model, "scoring_type", None)
            if scoring_type == "comet":
                # COMET checkpoints bundle sentencepiece.bpe.model next to config.json,
                # so we can construct the transform directly from the model dir.
                tokenizer = build_comet_sentencepiece_transform(self.model, self.config.model_path[0])
                if getattr(self.model, "class_identifier", None) == "xcomet_metric":
                    encode_texts_fn = encode_comet_texts_to_ids_and_offsets
                else:
                    encode_texts_fn = encode_comet_texts_to_ids
            else:
                raise ValueError(
                    "EncoderScorer could not resolve a tokenizer transform. "
                    "Declare one in the recipe, e.g.:\n"
                    "  transforms: [sentencepiece]\n"
                    "  transforms_configs:\n"
                    "    sentencepiece:\n"
                    "      src_subword_model: <path>\n"
                    "      tgt_subword_model: <path>\n"
                    f"(scoring_type={scoring_type!r} has no auto-fallback.)"
                )
        rows = build_segment_rows(
            tgt_lines, src_lines, ref_lines, tokenizer, self.model, encode_texts_fn=encode_texts_fn
        )

        batch_size = max(1, int(getattr(self.config, "batch_size", 64)))
        t0 = time.time()
        last_logged = 0

        def _log_progress(i):
            nonlocal last_logged
            if self.report_time and i - last_logged >= 10000:
                elapsed = time.time() - t0
                logger.info("Scored %d/%d (%.0f seg/s)", i, len(rows), i / max(elapsed, 1e-6))
                last_logged = i

        all_scores = predict_scores_with_oom_retry(
            self.model,
            rows,
            batch_size,
            logger=logger,
            progress_callback=_log_progress,
        )

        if self.report_time:
            elapsed = time.time() - t0
            logger.info("Scored %d pairs in %.1fs (%.0f seg/s)", len(rows), elapsed, len(rows) / max(elapsed, 1e-6))

        return score_level_output(all_scores, getattr(self.config, "score_level", "segment"), self.report_time, logger)
