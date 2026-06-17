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
