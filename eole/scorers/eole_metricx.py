import os

import torch

from .scorer import Scorer
from eole.models.model import EncoderDecoderScoringModel
from eole.scorers import register_scorer
from eole.utils.scorer import (
    build_scorer_sentencepiece_transform,
    build_template_inputs,
    encode_scorer_inputs,
    predict_scores_with_oom_retry,
)
from eole.utils.misc import clear_gpu_cache, get_device


def _slug(name):
    return name.replace("/", "--")


def _resolve_model_dir(model_name):
    if model_name and os.path.isdir(model_name):
        return model_name
    model_root = os.environ.get("EOLE_MODEL_DIR")
    if model_root and model_name:
        candidate = os.path.join(model_root, "metricx", _slug(model_name))
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        "EOLE converted MetricX model not found. Convert first with: "
        "`eole convert MetricX --model <hf-model-id> --output <dir>` and set `metricx_model` to that directory."
    )


class _EoleMetricXBase(Scorer):
    uses_gpu = True
    default_model = "google/metricx-24-hybrid-large-v2p6"
    metric_name = "EOLE-METRICX"
    input_mode = "reference"

    def __init__(self, config):
        super().__init__(config)
        self.model_name = getattr(config, "metricx_model", None) or self.default_model

    def _resolve_device(self):
        configured_device = getattr(self.config, "metricx_device", None)
        if configured_device:
            return torch.device(configured_device)
        return get_device()

    def _load_model(self, model_dir):
        device = self._resolve_device()
        compute_dtype = torch.float32 if device.type in {"cpu", "mps"} else torch.float16
        return EncoderDecoderScoringModel.from_model_dir(model_dir, device=device, compute_dtype=compute_dtype)

    def _validate_family(self, model):
        if getattr(model.model_config, "architecture", None) != "transformer_encoder_decoder_scorer":
            raise ValueError(f"{self.metric_name} requires a transformer_encoder_decoder_scorer model.")
        if getattr(model, "scoring_type", None) != "metricx":
            raise ValueError(f"{self.metric_name} requires scoring_type='metricx'.")
        model.validate_input_mode(self.input_mode)

    def compute_score(self, preds, texts_refs, texts_srcs=None):
        if not preds:
            return 0
        if texts_srcs is None:
            texts_srcs = [""] * len(preds)

        model_dir = _resolve_model_dir(self.model_name)
        model = self._load_model(model_dir)
        self._validate_family(model)

        if self.input_mode == "reference" and (texts_refs is None or len(texts_refs) != len(preds)):
            raise ValueError(f"{self.metric_name} requires references, but they were missing or malformed.")

        tokenizer = build_scorer_sentencepiece_transform(model, model_dir)
        inputs = build_template_inputs(
            texts_srcs,
            preds,
            texts_refs,
            model.input_templates,
            input_mode=self.input_mode,
        )
        input_ids = encode_scorer_inputs(inputs, tokenizer, model)
        rows = [{"input_ids": ids} for ids in input_ids]

        batch_size = max(1, int(getattr(self.config, "metricx_batch_size", 8)))
        all_scores = predict_scores_with_oom_retry(model, rows, batch_size)

        del model
        clear_gpu_cache()
        return float(sum(all_scores) / len(all_scores))


@register_scorer(metric="EOLE-METRICX")
class EoleMetricXScorer(_EoleMetricXBase):
    metric_name = "EOLE-METRICX"
    input_mode = "reference"


@register_scorer(metric="EOLE-METRICX-QE")
class EoleMetricXQEScorer(_EoleMetricXBase):
    metric_name = "EOLE-METRICX-QE"
    input_mode = "qe"
