import unittest
from types import SimpleNamespace
from unittest.mock import patch

from eole.modules.gguf_linear import _load_gguf_operation


class TestGGUFBackend(unittest.TestCase):
    def test_plugin_backend(self):
        operation = object()
        with patch("importlib.import_module", return_value=SimpleNamespace(fused_mul_mat_gguf=operation)) as load:
            self.assertEqual(_load_gguf_operation(), (operation, None))
        load.assert_called_once_with("vllm_gguf_plugin.quantization.linear")

    def test_legacy_backend(self):
        operation = object()
        with patch(
            "importlib.import_module",
            side_effect=[ModuleNotFoundError("plugin missing"), SimpleNamespace(fused_mul_mat_gguf=operation)],
        ):
            self.assertEqual(_load_gguf_operation(), (operation, None))

    def test_actual_import_failures_are_preserved(self):
        with patch(
            "importlib.import_module",
            side_effect=[RuntimeError("CUDA symbol mismatch"), ModuleNotFoundError("legacy GGUF missing")],
        ):
            operation, error = _load_gguf_operation()
        self.assertIsNone(operation)
        self.assertIn("CUDA symbol mismatch", error)
        self.assertIn("legacy GGUF missing", error)
