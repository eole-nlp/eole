"""Check image tagging and publishing without a Docker daemon or registry."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class DockerBuildScriptsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copytree(
            Path(__file__).resolve().parents[2] / "docker", self.root / "docker"
        )
        self.run_command("git", "init", "-q")
        self.run_command("git", "add", ".")
        self.run_command(
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "test",
        )
        self.run_command("git", "tag", "0.6.3")
        self.log = self.root / "calls"
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        docker = bin_dir / "docker"
        docker.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$DOCKER_TEST_LOG"\n')
        docker.chmod(0o755)
        self.env = {
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "DOCKER_TEST_LOG": str(self.log),
        }

    def run_command(self, *args, **kwargs):
        return subprocess.run(
            args, cwd=self.root, text=True, capture_output=True, check=True, **kwargs
        )

    def script(self, name, *args):
        return subprocess.run(
            ["bash", str(self.root / "docker" / name), *args],
            cwd="/tmp",
            env=self.env,
            text=True,
            capture_output=True,
        )

    def test_local_build_does_not_push(self):
        result = self.script("build.sh", "0.6.3", "2.14.1", "13.2.1")
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.log.read_text()
        self.assertIn("0.6.3-torch2.14.1-ubuntu24.04-cuda13.2.1", calls)
        self.assertIn("VCS_REF=", calls)
        self.assertNotIn("push ", calls)

    def test_release_tag_must_match_checkout(self):
        result = self.script("build.sh", "0.6.2", "2.14.1", "13.2.1", "--push")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.log.exists())

    def test_explicit_push_and_development_tag(self):
        revision = self.run_command(
            "git", "rev-parse", "--short=12", "HEAD"
        ).stdout.strip()
        result = self.script(
            "build.sh", f"dev-{revision}", "2.14.1", "13.2.1", "--push"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.log.read_text().splitlines()), 2)
        self.assertTrue(self.log.read_text().splitlines()[1].startswith("push "))

    def test_dirty_release_is_not_published(self):
        with (self.root / "docker" / "Dockerfile").open("a") as output:
            output.write("\n# changed\n")
        result = self.script("build.sh", "0.6.3", "2.14.1", "13.2.1", "--push")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.log.exists())

    def test_other_cuda_family_does_not_install_cuda132_compiler(self):
        self.env.pop("CUDA_NVCC_PACKAGE", None)
        result = self.script("build_base.sh", "2.11.0", "13.0.0")
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.log.read_text()
        self.assertNotIn("cuda-nvcc-13-2", calls)
        self.assertNotIn("push ", calls)

    def test_bad_arguments_do_not_start_docker(self):
        for name, args in [
            ("build.sh", []),
            ("build_base.sh", ["2.14.1", "13.2.1", "--typo"]),
            ("build_base.sh", ["2.14.1; echo bad", "13.2.1"]),
        ]:
            with self.subTest(name=name, args=args):
                self.assertNotEqual(self.script(name, *args).returncode, 0)
                self.assertFalse(self.log.exists())


if __name__ == "__main__":
    unittest.main()
