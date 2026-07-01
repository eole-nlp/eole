import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from eole.models.hf_resolver import resolve_preconverted_eole_hf_repo
from eole.models.model import get_model_class
from eole.predict import build_predictor, get_infer_class
from eole.scorers.eole_comet import _resolve_model_dir as resolve_comet_model_dir
from eole.scorers.eole_metricx import _resolve_model_dir as resolve_metricx_model_dir


class TestHfEoleResolver(unittest.TestCase):
    def test_resolves_preconverted_eole_hf_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_dir = tmp_path / "owner--model"
            config_path = tmp_path / "config.json"
            config_path.write_text(json.dumps({"model": {}, "training": {}}), encoding="utf-8")
            cache_checks = iter([False, True])

            with (
                patch(
                    "eole.models.hf_resolver.list_repo_files",
                    return_value=["config.json", "vocab.json", "model.00.safetensors"],
                ) as list_repo_files,
                patch("eole.models.hf_resolver.hf_hub_download", return_value=str(config_path)) as hf_hub_download,
                patch("eole.models.hf_resolver.snapshot_download") as snapshot_download,
                patch("eole.models.hf_resolver.hf_cache_dir", return_value=str(cache_dir)),
                patch("eole.models.hf_resolver._config_file_is_loadable_eole_model", return_value=True),
                patch(
                    "eole.models.hf_resolver._local_dir_is_loadable_eole_model",
                    side_effect=lambda _: next(cache_checks),
                ),
            ):
                resolved = resolve_preconverted_eole_hf_repo("owner/model", token="token")

            self.assertEqual(resolved, str(cache_dir))
            list_repo_files.assert_called_once_with(repo_id="owner/model", token="token")
            hf_hub_download.assert_called_once_with(
                repo_id="owner/model", filename="config.json", token="token", local_dir=str(cache_dir)
            )
            snapshot_download.assert_called_once_with(repo_id="owner/model", token="token", local_dir=str(cache_dir))

    def test_valid_cache_is_used_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "owner--model"
            cache_dir.mkdir()
            (cache_dir / "config.json").write_text(json.dumps({"model": {}, "training": {}}), encoding="utf-8")
            (cache_dir / "vocab.json").write_text("{}", encoding="utf-8")
            (cache_dir / "model.00.safetensors").write_text("", encoding="utf-8")

            with (
                patch("eole.models.hf_resolver.hf_cache_dir", return_value=str(cache_dir)),
                patch("eole.models.hf_resolver._config_file_is_loadable_eole_model", return_value=True),
                patch("eole.models.hf_resolver.list_repo_files") as list_repo_files,
            ):
                resolved = resolve_preconverted_eole_hf_repo("owner/model")

            self.assertEqual(resolved, str(cache_dir))
            list_repo_files.assert_not_called()

    def test_list_repo_files_failure_without_cache_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "owner--model"

            with (
                patch("eole.models.hf_resolver.hf_cache_dir", return_value=str(cache_dir)),
                patch("eole.models.hf_resolver.list_repo_files", side_effect=RuntimeError("offline")),
            ):
                self.assertIsNone(resolve_preconverted_eole_hf_repo("owner/model"))

    def test_remote_eole_config_download_failure_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "owner--model"

            with (
                patch("eole.models.hf_resolver.hf_cache_dir", return_value=str(cache_dir)),
                patch(
                    "eole.models.hf_resolver.list_repo_files",
                    return_value=["config.json", "vocab.json", "model.00.safetensors"],
                ),
                patch("eole.models.hf_resolver.hf_hub_download", side_effect=RuntimeError("rate limited")),
            ):
                with self.assertRaisesRegex(RuntimeError, "config validation failed"):
                    resolve_preconverted_eole_hf_repo("owner/model")

    def test_remote_eole_download_failure_uses_valid_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "owner--model"
            cache_dir.mkdir()
            (cache_dir / "config.json").write_text(json.dumps({"model": {}, "training": {}}), encoding="utf-8")
            (cache_dir / "vocab.json").write_text("{}", encoding="utf-8")
            (cache_dir / "model.00.safetensors").write_text("", encoding="utf-8")

            cache_checks = iter([False, True])

            with (
                patch("eole.models.hf_resolver.hf_cache_dir", return_value=str(cache_dir)),
                patch(
                    "eole.models.hf_resolver._local_dir_is_loadable_eole_model",
                    side_effect=lambda _: next(cache_checks),
                ),
                patch(
                    "eole.models.hf_resolver.list_repo_files",
                    return_value=["config.json", "vocab.json", "model.00.safetensors"],
                ),
                patch("eole.models.hf_resolver._remote_config_is_eole_model", return_value=True),
                patch("eole.models.hf_resolver.snapshot_download", side_effect=RuntimeError("offline")),
            ):
                resolved = resolve_preconverted_eole_hf_repo("owner/model")

            self.assertEqual(resolved, str(cache_dir))

    def test_remote_eole_download_failure_without_cache_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "owner--model"

            with (
                patch("eole.models.hf_resolver.hf_cache_dir", return_value=str(cache_dir)),
                patch(
                    "eole.models.hf_resolver.list_repo_files",
                    return_value=["config.json", "vocab.json", "model.00.safetensors"],
                ),
                patch("eole.models.hf_resolver._remote_config_is_eole_model", return_value=True),
                patch("eole.models.hf_resolver.snapshot_download", side_effect=RuntimeError("offline")),
            ):
                with self.assertRaisesRegex(RuntimeError, "download failed"):
                    resolve_preconverted_eole_hf_repo("owner/model")

    def test_non_eole_hf_repo_is_not_resolved(self):
        with (
            patch(
                "eole.models.hf_resolver.list_repo_files",
                return_value=["config.json", "tokenizer.json", "model.safetensors.index.json"],
            ),
            patch("eole.models.hf_resolver.snapshot_download") as snapshot_download,
        ):
            resolved = resolve_preconverted_eole_hf_repo("owner/model")

        self.assertIsNone(resolved)
        snapshot_download.assert_not_called()

    def test_build_predictor_loads_preconverted_hf_repo_as_local_eole_model(self):
        class FakeModelClass:
            @staticmethod
            def for_inference(config, device_id):
                return object(), {"src": object()}, SimpleNamespace(architecture="transformer_encoder_scorer")

        class FakePredictor:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        config = SimpleNamespace(
            model_path=["owner/model"],
            hf_token="token",
            model=SimpleNamespace(),
            _update_with_model_config=MagicMock(),
            update=MagicMock(),
        )

        with (
            patch("eole.predict.resolve_preconverted_eole_hf_repo", return_value="/tmp/eole-model") as resolve,
            patch("eole.predict.get_model_class", return_value=FakeModelClass) as get_model,
            patch("eole.predict.get_infer_class", return_value=FakePredictor),
            patch("eole.predict.GNMTGlobalScorer.from_config", return_value=object()),
            patch("eole.models.hf_loader.load_hf_model") as load_hf_model,
        ):
            predictor = build_predictor(config)

        self.assertIsInstance(predictor, FakePredictor)
        self.assertEqual(config.model_path, ["/tmp/eole-model"])
        config._update_with_model_config.assert_called_once()
        resolve.assert_called_once_with("owner/model", token="token")
        get_model.assert_called_once_with(config.model)
        load_hf_model.assert_not_called()

    def test_build_predictor_falls_back_to_generic_hf_loader(self):
        class FakePredictor:
            def __init__(self, *args, **kwargs):
                pass

        config = SimpleNamespace(
            model_path=["owner/model"],
            hf_token=None,
            update=MagicMock(),
        )
        model_config = SimpleNamespace(architecture="transformer_encoder_scorer")

        with (
            patch("eole.predict.resolve_preconverted_eole_hf_repo", return_value=None),
            patch("eole.models.hf_loader.load_hf_model", return_value=(object(), {}, model_config)) as load_hf_model,
            patch("eole.predict.get_infer_class", return_value=FakePredictor),
            patch("eole.predict.GNMTGlobalScorer.from_config", return_value=object()),
        ):
            build_predictor(config)

        load_hf_model.assert_called_once_with(config, 0)

    def test_comet_scorer_resolves_preconverted_hf_repo(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("eole.scorers.eole_comet.resolve_preconverted_eole_hf_repo", return_value="/tmp/comet") as resolve,
        ):
            self.assertEqual(resolve_comet_model_dir("owner/comet"), "/tmp/comet")

        resolve.assert_called_once_with("owner/comet")

    def test_metricx_scorer_resolves_preconverted_hf_repo(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "eole.scorers.eole_metricx.resolve_preconverted_eole_hf_repo", return_value="/tmp/metricx"
            ) as resolve,
        ):
            self.assertEqual(resolve_metricx_model_dir("owner/metricx"), "/tmp/metricx")

        resolve.assert_called_once_with("owner/metricx")

    def test_missing_model_config_errors_are_clear(self):
        with self.assertRaisesRegex(ValueError, "Model config is missing"):
            get_model_class(None)

        with self.assertRaisesRegex(ValueError, "Model config is missing"):
            get_infer_class(None)


if __name__ == "__main__":
    unittest.main()
