import time

import torch

from eole.predict.inference import Inference
from eole.utils.scorer import (
    apply_scorer_inference_defaults,
    build_scorer_sentencepiece_transform,
    build_template_inputs,
    encode_scorer_inputs,
    predict_scores_with_oom_retry,
    read_score_lines,
    score_level_output,
)
from eole.utils.logging import logger as _module_logger

logger = _module_logger


class EncoderDecoderScorer(Inference):
    def __init__(
        self,
        model,
        vocabs,
        config,
        model_config,
        device_id=0,
        global_scorer=None,
        report_score=True,
        logger=None,  # noqa: A002
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

    def update_settings(self, **kwargs):
        return

    def _predict(self, infer_iter, transforms, attn_debug, align_debug, streamer=None):
        if not self.config.src or not self.config.tgt:
            raise ValueError("EncoderDecoderScorer requires both --src and --tgt files.")
        if not getattr(self.config, "with_score", False):
            raise ValueError("EncoderDecoderScorer requires --with_score to emit scores.")

        src_lines = read_score_lines(self.config.src)
        tgt_lines = read_score_lines(self.config.tgt)
        ref_lines = read_score_lines(self.config.ref) if getattr(self.config, "ref", None) else None
        input_mode = "reference" if ref_lines is not None else "qe"
        if hasattr(self.model, "validate_input_mode"):
            self.model.validate_input_mode(input_mode)
        inputs = build_template_inputs(
            src_lines,
            tgt_lines,
            ref_lines,
            getattr(self.model, "input_templates", {}),
            input_mode=input_mode,
        )

        tokenizer = build_scorer_sentencepiece_transform(
            self.model,
            self.config.model_path[0],
            replace_newline_sentinel=getattr(self.config, "metricx_replace_newline_sentinel", True),
        )

        input_ids = encode_scorer_inputs(inputs, tokenizer, self.model)
        rows = [{"input_ids": ids} for ids in input_ids]

        batch_size = max(1, int(getattr(self.config, "batch_size", 64)))
        t0 = time.time()
        all_scores = predict_scores_with_oom_retry(self.model, rows, batch_size, logger=logger)

        if self.report_time:
            elapsed = time.time() - t0
            logger.info("Scored %d pairs in %.1fs (%.0f seg/s)", len(rows), elapsed, len(rows) / max(elapsed, 1e-6))

        return score_level_output(all_scores, getattr(self.config, "score_level", "segment"), self.report_time, logger)
