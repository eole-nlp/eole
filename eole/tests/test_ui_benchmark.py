"""Browser benchmark comparisons use fixed workloads and truthful measurement summaries."""

from pathlib import Path
import shutil
import subprocess
import unittest


class BenchmarkTests(unittest.TestCase):
    def test_browser_benchmark_summary(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node is needed for browser helper checks")
        subprocess.run([node, str(Path(__file__).with_name("web") / "test_benchmark.cjs")], check=True, timeout=20)
