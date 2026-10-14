import json
import os
import subprocess
import sys
import unittest


class TestTritonCacheInitialization(unittest.TestCase):
    def import_eole(self, settings):
        env = os.environ.copy()
        for key in ("EOLE_TORCH_COMPILE", "TRITON_CACHE_DIR", "TRITON_HOME"):
            env.pop(key, None)
        env.update(settings)
        result = subprocess.check_output(
            [
                sys.executable,
                "-c",
                "import eole, json, os, sys; "
                "print(json.dumps([os.environ.get('TRITON_CACHE_DIR'), "
                "'triton' in sys.modules, 'torch' in sys.modules]))",
            ],
            env=env,
            text=True,
        )
        return json.loads(result)

    def test_compile_pins_default_before_optional_imports(self):
        self.assertEqual(
            self.import_eole({"EOLE_TORCH_COMPILE": "1"}),
            [os.path.expanduser("~/.triton/cache"), False, False],
        )

    def test_custom_triton_home(self):
        self.assertEqual(
            self.import_eole({"EOLE_TORCH_COMPILE": "1", "TRITON_HOME": "/custom/home"})[0],
            "/custom/home/.triton/cache",
        )

    def test_explicit_cache_is_preserved(self):
        for cache in ("/custom/cache", ""):
            with self.subTest(cache=cache):
                self.assertEqual(self.import_eole({"EOLE_TORCH_COMPILE": "1", "TRITON_CACHE_DIR": cache})[0], cache)

    def test_compile_disabled_leaves_cache_unset(self):
        for settings in ({}, {"EOLE_TORCH_COMPILE": "0"}):
            with self.subTest(settings=settings):
                self.assertEqual(self.import_eole(settings), [None, False, False])


if __name__ == "__main__":
    unittest.main()
