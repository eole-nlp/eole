"""Optional load-test dependencies must not break other Eole commands."""

import subprocess
import sys
import textwrap
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch


class TestLoadTestImports(unittest.TestCase):
    def test_cli_and_converters_import_without_aiohttp(self):
        script = textwrap.dedent(
            """
            import importlib.abc
            import sys

            class WithoutAiohttp(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "aiohttp" or fullname.startswith("aiohttp."):
                        raise ModuleNotFoundError("aiohttp deliberately unavailable", name="aiohttp")

            sys.meta_path.insert(0, WithoutAiohttp())
            from eole.bin.convert.convert_HF import build_config_dict
            from eole.models.hf_loader import load_hf_model
            from eole.bin.tools.load_test import LoadTester
            from eole.bin.main import main

            assert "aiohttp" not in sys.modules
            try:
                LoadTester("http://localhost:5000", "model")
            except ImportError as error:
                assert "python -m pip install aiohttp" in str(error), str(error)
            else:
                raise AssertionError("Missing dependency was not reported")
            sys.argv = ["eole", "tools", "load_test", "--help"]
            try:
                main()
            except SystemExit as error:
                assert error.code == 0, error.code
            """
        )
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class TestLoadTestClient(unittest.IsolatedAsyncioTestCase):
    async def test_request_types_use_lazily_loaded_client(self):
        from eole.bin.tools.load_test import LoadTester

        response = SimpleNamespace(status=200, json=AsyncMock(return_value={}))
        request = MagicMock()
        request.__aenter__ = AsyncMock(return_value=response)
        session = MagicMock()
        session.post.return_value = request
        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=session)
        aiohttp = SimpleNamespace(ClientSession=MagicMock(return_value=client), ClientTimeout=MagicMock())
        with patch.dict(sys.modules, {"aiohttp": aiohttp}):
            tester = LoadTester("http://localhost:5000", "model")
        for kind in ("text", "chat", "openai"):
            with self.subTest(kind=kind):
                results = await tester.run_concurrent_batch(2, request_type=kind)
                self.assertEqual(len(results), 2)
                self.assertTrue(all(result.success for result in results))
        self.assertEqual(session.post.call_count, 6)
        self.assertEqual(aiohttp.ClientTimeout.call_count, 6)
