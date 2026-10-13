"""The CUDA extension targets every visible GPU without invoking a compiler."""

from pathlib import Path
import runpy
import unittest
from unittest.mock import patch


class TestSetupCudaArchitectures(unittest.TestCase):
    def build_flags(self, capabilities, available=True):
        setup_path = Path(__file__).resolve().parents[2] / "setup.py"
        with (
            patch("torch.cuda.is_available", return_value=available),
            patch("torch.cuda.device_count", return_value=len(capabilities)),
            patch("torch.cuda.get_device_capability", side_effect=capabilities) as capability,
            patch("torch.utils.cpp_extension.CUDAExtension") as extension,
            patch("setuptools.setup"),
        ):
            runpy.run_path(str(setup_path))
        return extension, capability

    def test_mixed_gpus_compile_both_architectures(self):
        extension, capability = self.build_flags([(12, 0), (8, 9)])
        self.assertEqual([call.args for call in capability.call_args_list], [(0,), (1,)])
        flags = extension.call_args.kwargs["extra_compile_args"]["nvcc"]
        self.assertEqual(
            [flag for flag in flags if flag.startswith("-gencode")],
            [
                "-gencode=arch=compute_89,code=sm_89",
                "-gencode=arch=compute_89,code=compute_89",
                "-gencode=arch=compute_120,code=sm_120",
                "-gencode=arch=compute_120,code=compute_120",
            ],
        )

    def test_identical_gpus_do_not_duplicate_targets(self):
        extension, _ = self.build_flags([(8, 9), (8, 9)])
        flags = extension.call_args.kwargs["extra_compile_args"]["nvcc"]
        self.assertEqual(len([flag for flag in flags if flag.startswith("-gencode")]), 2)

    def test_one_visible_gpu_targets_only_that_architecture(self):
        extension, capability = self.build_flags([(8, 9)])
        capability.assert_called_once_with(0)
        flags = extension.call_args.kwargs["extra_compile_args"]["nvcc"]
        self.assertIn("-gencode=arch=compute_89,code=sm_89", flags)
        self.assertFalse(any("compute_120" in flag for flag in flags))

    def test_cpp_standard_is_selected_by_pytorch(self):
        extension, _ = self.build_flags([(12, 0), (8, 9)])
        for flags in extension.call_args.kwargs["extra_compile_args"].values():
            self.assertFalse(any(flag.startswith("-std=") for flag in flags))

    def test_cpu_build_does_not_query_gpu_capabilities(self):
        extension, capability = self.build_flags([], available=False)
        extension.assert_not_called()
        capability.assert_not_called()

    def test_query_failure_does_not_build_for_an_assumed_architecture(self):
        with self.assertRaisesRegex(RuntimeError, "visible GPU 1"):
            self.build_flags([(12, 0), RuntimeError("device query failed")])
