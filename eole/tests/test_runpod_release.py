"""Reject a stale parent before publishing a numbered RunPod release."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class TestRunPodRelease(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        recipe = self.root / "recipes/runpod"
        recipe.mkdir(parents=True)
        self.script = recipe / "check_release.sh"
        shutil.copy2(Path(__file__).resolve().parents[2] / "recipes/runpod/check_release.sh", self.script)
        (self.root / "eole").mkdir()
        (self.root / "eole/__init__.py").write_text('__version__ = "0.6.5"\n')
        self.git("init", "-q")
        self.git("add", ".")
        self.git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "release")
        self.git("tag", "0.6.5")
        revision = self.git("rev-parse", "HEAD").stdout.strip()
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        docker = bin_dir / "docker"
        docker.write_text(
            '#!/bin/sh\nprintf "%s\\n" "$*" >> "$DOCKER_TEST_LOG"\n'
            'case "$1" in\n'
            "pull) exit 0 ;;\n"
            'run) printf "%s\\n" "$PARENT_VERSION" ;;\n'
            'image) printf "%s\\n" "$PARENT_REVISION" ;;\n'
            "*) exit 1 ;;\nesac\n"
        )
        docker.chmod(0o755)
        self.log = self.root / "calls"
        self.env = {
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "DOCKER_TEST_LOG": str(self.log),
            "PARENT_VERSION": "0.6.5",
            "PARENT_REVISION": revision,
        }

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=self.root, text=True, capture_output=True, check=True)

    def check(self, tag="0.6.5"):
        return subprocess.run(
            ["bash", str(self.script), tag, "example/eole:release"],
            cwd="/tmp",
            env=self.env,
            text=True,
            capture_output=True,
        )

    def test_matching_source_and_parent_are_accepted(self):
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("image inspect", self.log.read_text())

    def test_stale_parent_version_or_revision_is_rejected(self):
        for key, value in [("PARENT_VERSION", "0.6.4"), ("PARENT_REVISION", "0" * 40)]:
            with self.subTest(key=key):
                original = self.env[key]
                self.env[key] = value
                result = self.check()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Rebuild the full Eole image", result.stderr)
                self.env[key] = original

    def test_untagged_source_is_rejected_before_pulling_images(self):
        self.git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "--allow-empty", "-qm", "later")
        result = self.check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("source_ref=0.6.5", result.stderr)
        self.assertFalse(self.log.exists())

    def test_wrong_source_package_version_is_rejected(self):
        (self.root / "eole/__init__.py").write_text('__version__ = "0.6.4"\n')
        result = self.check()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.log.exists())

    def test_development_build_does_not_require_a_release_tag_or_pull(self):
        self.git("tag", "-d", "0.6.5")
        for tag in ("", "dev-123456789abc"):
            result = self.check(tag)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.log.exists())


if __name__ == "__main__":
    unittest.main()
