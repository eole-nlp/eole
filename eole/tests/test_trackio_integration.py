"""Integration tests for EOLE's real trackio wiring.

The exercised scenario runs in trackio_integration_script.py as a subprocess
so TRACKIO_DIR is set before trackio is imported. Set
EOLE_RUN_TRACKIO_INTEGRATION=1 to enable these tests; they also skip when
the optional eole[trackio] dependencies are absent.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "trackio_integration_script.py"


@unittest.skipUnless(
    os.environ.get("EOLE_RUN_TRACKIO_INTEGRATION") == "1",
    "set EOLE_RUN_TRACKIO_INTEGRATION=1 to run trackio integration tests",
)
class TestTrackioIntegration(unittest.TestCase):
    def test_local_trackio_run_artifacts_metrics_and_system_stats(self):
        repo_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmpdir:
            env = os.environ.copy()
            env["TRACKIO_DIR"] = str(Path(tmpdir) / "trackio")
            env["TRACKIO_IT_WORKDIR"] = tmpdir
            # Run the script against this working tree, not any installed eole.
            env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(repo_root), env.get("PYTHONPATH")]))
            result = subprocess.run(
                [sys.executable, str(SCRIPT)],
                cwd=repo_root,
                env=env,
                text=True,
                capture_output=True,
                timeout=60,
            )

        if result.returncode == 77:
            self.skipTest(result.stdout.strip() or result.stderr.strip())
        self.assertEqual(result.returncode, 0, msg=f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")


if __name__ == "__main__":
    unittest.main()
