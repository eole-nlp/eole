import os

from .scorer import Scorer
from eole.models.model import EncoderScoringModel
from eole.scorers import register_scorer
from eole.utils.comet_scorer import (
    build_comet_sentencepiece_transform,
    encode_comet_texts_to_ids,
    encode_comet_texts_to_ids_and_offsets,
)
from eole.utils.scorer import (
    build_segment_rows,
    predict_scores_with_oom_retry,
)
from eole.utils.misc import clear_gpu_cache, get_device

import torch


def _slug(name):
    return name.replace("/", "--")


def _resolve_model_dir(model_name):
    if model_name and os.path.isdir(model_name):
        return model_name
    model_root = os.environ.get("EOLE_MODEL_DIR")
    if model_root and model_name:
        candidate = os.path.join(model_root, "comet", _slug(model_name))
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        "EOLE converted COMET model not found. Convert first with: "
        "`eole convert COMET --model <hf-model-id> --output <dir>` and set `comet_model` to that directory."
    )


class _EoleCometBase(Scorer):
    uses_gpu = True
    default_model = None
    metric_name = "EOLE-COMET"
    expect_reference = True
    expected_class_identifier = None

    def __init__(self, config):
        super().__init__(config)
        self.model_name = getattr(config, "comet_model", None) or self.default_model

    def _validate_family(self, model):
        if getattr(model, "scoring_type", "comet") != "comet":
            raise ValueError(
                f"{self.metric_name} requires a transformer_encoder_scorer model with scoring_type='comet'."
            )
        expected = self.expected_class_identifier
        if isinstance(expected, str):
            expected = {expected}
        if expected is not None and getattr(model, "class_identifier", None) not in expected:
            expected_text = ", ".join(f"'{value}'" for value in sorted(expected))
            raise ValueError(
                f"{self.metric_name} expects a converted model with " f"class_identifier in {{{expected_text}}}."
            )
        if self.expect_reference and not model.requires_reference:
            raise ValueError(f"{self.metric_name} expects a reference-based converted model.")
        if not self.expect_reference and model.requires_reference:
            raise ValueError(f"{self.metric_name} expects a reference-free converted model.")

    def _build_sp_transform(self, model, model_dir):
        return build_comet_sentencepiece_transform(model, model_dir)

    def _encode_texts(self, texts, tokenizer, model):
        if getattr(model, "class_identifier", None) == "xcomet_metric":
            return encode_comet_texts_to_ids_and_offsets(texts, tokenizer, model)
        return encode_comet_texts_to_ids(texts, tokenizer, model)

    def _resolve_device(self):
        configured_device = getattr(self.config, "comet_device", None)
        if configured_device:
            return torch.device(configured_device)
        return get_device()

    def compute_score(self, preds, texts_refs, texts_srcs=None):
        if not preds:
            return 0
        if texts_srcs is None:
            texts_srcs = [""] * len(preds)

        model_dir = _resolve_model_dir(self.model_name)
        model = EncoderScoringModel.from_model_dir(model_dir, device=self._resolve_device())
        self._validate_family(model)

        if model.requires_reference and (texts_refs is None or len(texts_refs) != len(preds)):
            raise ValueError("This EOLE-COMET model requires references, but they were missing or malformed.")

        tokenizer = self._build_sp_transform(model, model_dir)
        rows = build_segment_rows(preds, texts_srcs, texts_refs, tokenizer, model, encode_texts_fn=self._encode_texts)

        batch_size = max(1, int(getattr(self.config, "comet_batch_size", 64)))
        all_scores = predict_scores_with_oom_retry(model, rows, batch_size)

        del model
        clear_gpu_cache()
        return float(sum(all_scores) / len(all_scores))


@register_scorer(metric="EOLE-COMET")
class EoleCometScorer(_EoleCometBase):
    default_model = "Unbabel/wmt22-comet-da"
    metric_name = "EOLE-COMET"
    expect_reference = True
    expected_class_identifier = "regression_metric"


@register_scorer(metric="EOLE-COMET-KIWI")
class EoleCometKiwiScorer(_EoleCometBase):
    default_model = "Unbabel/wmt22-cometkiwi-da"
    metric_name = "EOLE-COMET-KIWI"
    expect_reference = False
    expected_class_identifier = {"referenceless_regression_metric", "unified_metric"}


@register_scorer(metric="EOLE-XCOMET")
class EoleXCometScorer(_EoleCometBase):
    default_model = "Unbabel/XCOMET-XL"
    metric_name = "EOLE-XCOMET"
    expect_reference = True
    expected_class_identifier = "xcomet_metric"
