import json
from types import SimpleNamespace
from unittest.mock import patch

import torch

from eole.config.run import PredictConfig


def _base_kwargs():
    return {
        "model_path": "org/model",
        "src": "src.txt",
        "tgt": "mt.txt",
        "output": "scores.txt",
    }


def test_predict_config_accepts_ref_field():
    cfg = PredictConfig(**_base_kwargs(), ref="ref.txt")

    assert cfg.ref == "ref.txt"
    assert cfg.src == "src.txt"
    assert cfg.tgt == "mt.txt"

    dumped = cfg.model_dump()
    assert dumped["ref"] == "ref.txt"


def test_predict_config_ref_defaults_to_none():
    cfg = PredictConfig(**_base_kwargs())

    assert cfg.ref is None
    dumped = cfg.model_dump()
    assert "ref" in dumped
    assert dumped["ref"] is None


def test_predict_config_score_level_defaults_to_segment():
    cfg = PredictConfig(**_base_kwargs())

    assert cfg.score_level == "segment"


def test_predict_config_accepts_system_score_level():
    cfg = PredictConfig(**_base_kwargs(), score_level="system")

    assert cfg.score_level == "system"


def test_predict_config_rejects_invalid_score_level():
    try:
        PredictConfig(**_base_kwargs(), score_level="sentence")
    except ValueError as exc:
        assert "score_level" in str(exc)
    else:
        raise AssertionError("Expected invalid score_level validation error")


def test_predict_config_defaults_comet_encoder_scorer_to_fp16(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"training": {"compute_dtype": "fp32"}, "model": {}}),
        encoding="utf-8",
    )

    model_config = SimpleNamespace(architecture="transformer_encoder_scorer", scoring_type="comet")
    training_config = SimpleNamespace(
        compute_dtype=torch.float32,
        world_size=1,
        gpu_ranks=[],
        quant_type="",
    )

    with (
        patch("eole.config.run.build_model_config", return_value=model_config),
        patch("eole.config.run.TrainingConfig", return_value=training_config),
    ):
        kwargs = {**_base_kwargs(), "model_path": str(model_dir)}
        cfg = PredictConfig(**kwargs)

    assert cfg.compute_dtype == torch.float16


def test_predict_config_preserves_explicit_comet_compute_dtype(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"training": {"compute_dtype": "fp32"}, "model": {}}),
        encoding="utf-8",
    )

    model_config = SimpleNamespace(architecture="transformer_encoder_scorer", scoring_type="comet")
    training_config = SimpleNamespace(
        compute_dtype=torch.float32,
        world_size=1,
        gpu_ranks=[],
        quant_type="",
    )

    with (
        patch("eole.config.run.build_model_config", return_value=model_config),
        patch("eole.config.run.TrainingConfig", return_value=training_config),
    ):
        kwargs = {**_base_kwargs(), "model_path": str(model_dir), "compute_dtype": "fp32"}
        cfg = PredictConfig(**kwargs)

    assert cfg.compute_dtype == torch.float32
