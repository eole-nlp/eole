"""Integration tests for EOLE's real trackio wiring.

The exercised scenario runs in trackio_integration_script.py as a subprocess
so TRACKIO_DIR is set before trackio is imported. Set
EOLE_RUN_TRACKIO_INTEGRATION=1 to enable these tests; they also skip when
the optional eole[trackio] dependencies are absent, but fail when an installed
dependency cannot be imported.
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
            env["TRACKIO_STORAGE_MODE"] = "sqlite"
            for key in (
                "SYSTEM",
                "TRACKIO_BUCKET_ID",
                "TRACKIO_DATASET_ID",
                "TRACKIO_SERVER_URL",
                "TRACKIO_SPACE_ID",
                "TRACKIO_WRITE_TOKEN",
            ):
                env.pop(key, None)
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


class TestTrackioIntegrationScriptSkip(unittest.TestCase):
    """The script must skip only for absent dependencies, never for broken installs."""

    def _run_isolated(self, modules):
        # -S drops site-packages so only the stub modules written here are importable.
        with tempfile.TemporaryDirectory() as tmpdir:
            for relpath, source in modules.items():
                path = Path(tmpdir) / relpath
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(source, encoding="utf-8")
            env = os.environ.copy()
            env["PYTHONPATH"] = tmpdir
            return subprocess.run(
                [sys.executable, "-S", str(SCRIPT)],
                env=env,
                text=True,
                capture_output=True,
                timeout=60,
            )

    def test_absent_dependencies_skip(self):
        result = self._run_isolated({})
        self.assertEqual(result.returncode, 77, msg=result.stderr)
        self.assertIn("SKIP", result.stdout)

    def test_installed_trackio_import_error_fails(self):
        cases = {
            "missing_name": "raise ImportError('incompatible huggingface_hub')\n",
            "missing_transitive_module": "import huggingface_hub\n",
        }
        for case, source in cases.items():
            with self.subTest(case=case):
                result = self._run_isolated({"psutil.py": "", "trackio/__init__.py": source})
                self.assertNotIn(result.returncode, (0, 77), msg=result.stdout)
                self.assertIn("Error", result.stderr)


if __name__ == "__main__":
    unittest.main()
