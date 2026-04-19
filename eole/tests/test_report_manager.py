"""Tests for the report manager classes."""

import unittest
from unittest.mock import MagicMock, patch

from eole.utils.report_manager import (
    CompositeReportMgr,
    ReportMgrBase,
    StdoutReportMgr,
    TrackioReportMgr,
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


if __name__ == "__main__":
    unittest.main()
