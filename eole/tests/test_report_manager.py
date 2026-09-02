"""Tests for the report manager classes."""

import os
import pathlib
import re
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import yaml

import eole

from eole.config.run import TrainConfig
from eole.utils.report_manager import (
    CompositeReportMgr,
    ReportMgrBase,
    StdoutReportMgr,
    TrackioReportMgr,
    build_report_manager,
)


class TestCompositeReportMgr(unittest.TestCase):
    def _make_mock_manager(self):
        mgr = MagicMock(spec=ReportMgrBase)
        mgr.start_time = -1.0
        return mgr

    def test_dispatches_report_training_to_all_managers(self):
        mgr1 = self._make_mock_manager()
        mgr2 = self._make_mock_manager()
        composite = CompositeReportMgr(report_every=5, managers=[mgr1, mgr2])
        stats = MagicMock()

        composite._report_training(10, 100, 0.001, None, stats)

        mgr1._report_training.assert_called_once_with(10, 100, 0.001, None, stats)
        mgr2._report_training.assert_called_once_with(10, 100, 0.001, None, stats)

    def test_dispatches_report_step_to_all_managers(self):
        mgr1 = self._make_mock_manager()
        mgr2 = self._make_mock_manager()
        composite = CompositeReportMgr(report_every=5, managers=[mgr1, mgr2])
        train_stats = MagicMock()
        valid_stats = MagicMock()

        composite._report_step(0.001, None, 10, valid_stats=valid_stats, train_stats=train_stats)

        mgr1._report_step.assert_called_once_with(0.001, None, 10, valid_stats=valid_stats, train_stats=train_stats)
        mgr2._report_step.assert_called_once_with(0.001, None, 10, valid_stats=valid_stats, train_stats=train_stats)

    def test_close_continues_on_failure(self):
        mgr1 = self._make_mock_manager()
        mgr2 = self._make_mock_manager()
        mgr1.close.side_effect = RuntimeError("tensorboard exploded")
        composite = CompositeReportMgr(report_every=5, managers=[mgr1, mgr2])

        composite.close()

        mgr1.close.assert_called_once()
        mgr2.close.assert_called_once()

    def test_start_time_propagates_to_children(self):
        mgr1 = self._make_mock_manager()
        mgr2 = self._make_mock_manager()
        composite = CompositeReportMgr(report_every=5, managers=[mgr1, mgr2])

        composite.start_time = 42.0

        assert composite.start_time == 42.0
        assert mgr1.start_time == 42.0
        assert mgr2.start_time == 42.0

    def test_start_sets_time_on_all(self):
        mgr1 = self._make_mock_manager()
        mgr2 = self._make_mock_manager()
        composite = CompositeReportMgr(report_every=5, managers=[mgr1, mgr2])

        composite.start()

        assert composite.start_time > 0
        assert mgr1.start_time == composite.start_time
        assert mgr2.start_time == composite.start_time


class TestStdoutReportMgr(unittest.TestCase):
    def test_report_training_calls_output(self):
        mgr = StdoutReportMgr(report_every=5)
        mgr.start_time = 1.0
        stats = MagicMock()

        mgr._report_training(10, 100, 0.001, None, stats)

        stats.output.assert_called_once_with(10, 100, 0.001, mgr.start_time)


class TestTrackioReportMgr(unittest.TestCase):
    def test_disables_on_log_failure(self):
        mgr = TrackioReportMgr(report_every=5)
        stats = MagicMock()
        stats.log_trackio.side_effect = RuntimeError("network error")

        mgr._report_training(10, 100, 0.001, None, stats)

        assert mgr._disabled is True

    def test_does_not_call_after_disabled(self):
        mgr = TrackioReportMgr(report_every=5)
        stats = MagicMock()
        stats.log_trackio.side_effect = RuntimeError("network error")

        # First call triggers disable
        mgr._report_training(10, 100, 0.001, None, stats)
        stats.log_trackio.reset_mock()

        # Second call should not attempt logging
        mgr._report_training(15, 100, 0.001, None, stats)
        stats.log_trackio.assert_not_called()

    def test_report_step_disables_on_failure(self):
        mgr = TrackioReportMgr(report_every=5)
        train_stats = MagicMock()
        train_stats.log_trackio.side_effect = RuntimeError("network error")

        mgr._report_step(0.001, None, 10, train_stats=train_stats)

        assert mgr._disabled is True

    @patch("eole.utils.report_manager.trackio", create=True)
    def test_close_calls_finish(self, mock_trackio):
        mgr = TrackioReportMgr(report_every=5)

        with patch.dict("sys.modules", {"trackio": mock_trackio}):
            mgr.close()

        mock_trackio.finish.assert_called_once()

    @patch("eole.utils.report_manager.trackio", create=True)
    def test_close_skipped_when_disabled(self, mock_trackio):
        mgr = TrackioReportMgr(report_every=5)
        mgr._disabled = True

        mgr.close()

        mock_trackio.finish.assert_not_called()


class TestBuildReportManagerTrackio(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch("eole.utils.report_manager._git_short_commit", return_value="abc1234"))

    def _make_config(self, config_file=None, **overrides):
        cfg = TrainConfig(
            src_vocab="src.vocab",
            tgt_vocab="tgt.vocab",
            data={},
            model={"architecture": "rnn"},
            training={"batch_size": 2, "train_steps": 1},
            report_every=5,
            trackio=True,
            trackio_project="proj",
            trackio_group="grp",
            **overrides,
        )
        cfg._config_file = config_file
        return cfg

    def _new_trackio(self):
        state = {"init": None, "artifacts": []}

        def init(
            project=None,
            name=None,
            group=None,
            space_id=None,
            bucket_id=None,
            config=None,
            auto_log_cpu=None,
            auto_log_gpu=None,
            cpu_log_interval=None,
            gpu_log_interval=None,
            **kwargs,
        ):
            state["init"] = dict(
                project=project,
                name=name,
                group=group,
                space_id=space_id,
                bucket_id=bucket_id,
                config=config,
                auto_log_cpu=auto_log_cpu,
                auto_log_gpu=auto_log_gpu,
                cpu_log_interval=cpu_log_interval,
                gpu_log_interval=gpu_log_interval,
                **kwargs,
            )

        def log_artifact(artifact_or_path, name=None, type=None, aliases=None):
            path = pathlib.Path(artifact_or_path)
            # capture content at call time: generated artifacts live in temp dirs
            state["artifacts"].append(
                {"path": str(path), "name": name, "type": type, "content": path.read_text(encoding="utf-8")}
            )

        return SimpleNamespace(init=init, log_artifact=log_artifact), state

    def _run_with_config_file(self, filename):
        trackio, state = self._new_trackio()
        with tempfile.TemporaryDirectory() as tmpdir:
            orig = os.path.join(tmpdir, filename)
            with open(orig, "w", encoding="utf-8") as f:
                f.write("training:\n  batch_size: 2\n")
            config = self._make_config(config_file=orig)
            with patch.dict("sys.modules", {"trackio": trackio}):
                build_report_manager(config, gpu_rank=0)
        return trackio, state, orig

    def test_build_logs_artifacts_and_system_kwargs(self):
        trackio, state, orig = self._run_with_config_file("wmt17-mini-trackio.yml")

        init_kwargs = state["init"]
        self.assertIsNotNone(init_kwargs)
        self.assertEqual(init_kwargs["group"], "grp")
        self.assertEqual(init_kwargs["config"]["eole_version"], eole.__version__)
        self.assertEqual(init_kwargs["config"]["git_commit"], "abc1234")
        self.assertEqual(init_kwargs["config"]["model"]["architecture"], "rnn")
        self.assertEqual(init_kwargs["config"]["training"]["batch_size"], 2)
        self.assertIn("auto_log_cpu", init_kwargs)
        self.assertIn("auto_log_gpu", init_kwargs)
        self.assertEqual(init_kwargs["cpu_log_interval"], 10.0)
        self.assertEqual(init_kwargs["gpu_log_interval"], 10.0)

        self.assertEqual(
            [a["name"] for a in state["artifacts"]],
            ["wmt17-mini-trackio.yml", "wmt17-mini-trackio-effective.yml"],
        )
        self.assertEqual(state["artifacts"][0]["path"], orig)
        self.assertEqual(state["artifacts"][0]["type"], "config")
        self.assertEqual(state["artifacts"][1]["type"], "config")

        self.assertEqual(state["artifacts"][0]["content"], "training:\n  batch_size: 2\n")

        effective = yaml.safe_load(state["artifacts"][1]["content"])
        # full runtime config: explicit values, defaults, and validated derivations
        self.assertEqual(effective["trackio_project"], "proj")
        self.assertEqual(effective["trackio_group"], "grp")
        self.assertFalse(effective["tensorboard"])
        self.assertEqual(effective["training"]["save_format"], "pytorch")
        self.assertEqual(effective["training"]["compute_dtype"], "torch.float32")
        self.assertEqual(effective["model"]["architecture"], "rnn")
        self.assertIn("embeddings", effective["model"])
        # private attrs are excluded from the dump
        self.assertNotIn("_config_file", effective)

    def test_cli_only_run_uploads_only_effective(self):
        trackio, state = self._new_trackio()
        config = self._make_config()
        with patch.dict("sys.modules", {"trackio": trackio}):
            build_report_manager(config, gpu_rank=0)

        self.assertEqual([a["name"] for a in state["artifacts"]], ["config-effective.yaml"])

    def test_artifact_names_derived_from_config_filename(self):
        cases = {
            "train.yaml": ("train.yaml", "train-effective.yaml"),
            "config.yml": ("config.yml", "config-effective.yml"),
            # non-YAML content suffix must not be inherited by the YAML effective artifact
            "config.json": ("config.json", "config-effective.yaml"),
            "my run (v2).yaml": ("my-run--v2-.yaml", "my-run--v2--effective.yaml"),
            "!!!.yaml": ("---.yaml", "----effective.yaml"),
            ".yaml": (".yaml", ".yaml-effective.yaml"),
        }
        for filename, expected_names in cases.items():
            with self.subTest(filename=filename):
                _trackio, state, _orig = self._run_with_config_file(filename)
                names = [a["name"] for a in state["artifacts"]]
                self.assertEqual(names, list(expected_names))
                self.assertTrue(all(re.fullmatch(r"[A-Za-z0-9._-]+", n) for n in names), names)

    def test_build_skips_artifact_when_disabled(self):
        trackio, state = self._new_trackio()
        config = self._make_config(trackio_log_config_artifact=False)
        with patch.dict("sys.modules", {"trackio": trackio}):
            build_report_manager(config, gpu_rank=0)

        self.assertIsNotNone(state["init"])
        self.assertEqual(state["artifacts"], [])

    def test_artifact_failure_keeps_trackio_manager(self):
        trackio, state = self._new_trackio()

        def boom(*args, **kwargs):
            raise RuntimeError("artifact upload failed")

        trackio.log_artifact = boom
        config = self._make_config()
        with patch.dict("sys.modules", {"trackio": trackio}):
            mgr = build_report_manager(config, gpu_rank=0)

        # artifact upload failed, but init succeeded, so metric logging is preserved
        self.assertIsNotNone(state["init"])
        self.assertTrue(any(isinstance(m, TrackioReportMgr) for m in mgr.managers))

    def test_original_artifact_failure_still_uploads_effective(self):
        trackio, state = self._new_trackio()
        record = trackio.log_artifact

        def reject_original(artifact_or_path, name=None, type=None, aliases=None):
            if name == "wmt17-mini-trackio.yml":
                raise RuntimeError("original artifact upload rejected")
            return record(artifact_or_path, name=name, type=type, aliases=aliases)

        trackio.log_artifact = reject_original
        with tempfile.TemporaryDirectory() as tmpdir:
            orig = os.path.join(tmpdir, "wmt17-mini-trackio.yml")
            with open(orig, "w", encoding="utf-8") as f:
                f.write("training:\n  batch_size: 2\n")
            config = self._make_config(config_file=orig)
            with patch.dict("sys.modules", {"trackio": trackio}):
                mgr = build_report_manager(config, gpu_rank=0)

        # original upload failed, but the reproducibility-effective artifact still lands
        self.assertEqual([a["name"] for a in state["artifacts"]], ["wmt17-mini-trackio-effective.yml"])
        self.assertTrue(yaml.safe_load(state["artifacts"][0]["content"])["trackio"])
        self.assertTrue(any(isinstance(m, TrackioReportMgr) for m in mgr.managers))


if __name__ == "__main__":
    unittest.main()
