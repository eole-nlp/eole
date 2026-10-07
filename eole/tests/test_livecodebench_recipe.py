"""Exercise recipe transport and artifact checks without models or benchmark downloads."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "livecodebench_recipe", Path(__file__).resolve().parents[2] / "recipes/livecodebench/run.py"
)
recipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipe)


class TestLiveCodeBenchRecipe(unittest.TestCase):
    def test_generation_excludes_tests_and_preserves_partial_results(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            tasks = [
                {
                    "id": str(i),
                    "messages": [{"role": "user", "content": "Solve this"}],
                    "evaluation_sample": {"input_output": "SECRET"},
                }
                for i in range(2)
            ]
            recipe.save(output / "tasks.json", {"tasks": tasks})
            config = {
                "base_url": "http://localhost:5010/v1",
                "model_id": "test",
                "max_tokens": 10,
                "request_timeout": 5,
            }
            response = {"choices": [{"message": {"content": "```python\nprint(42)\n```"}, "finish_reason": "stop"}]}
            with patch.object(recipe, "request_json", side_effect=[response, TimeoutError]) as request:
                with self.assertRaises(TimeoutError):
                    recipe.generate(config, output)
                self.assertNotIn("SECRET", json.dumps(request.call_args_list))
                self.assertEqual(request.call_args_list[0].args[0], "http://localhost:5010/v1/chat/completions")
            saved = json.loads((output / "generations.json").read_text())
            self.assertEqual(len(saved["records"]), 1)
            self.assertEqual(saved["records"][0]["id"], "0")
            with self.assertRaisesRegex(ValueError, "already exists"):
                recipe.generate(config, output)

    def test_execution_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.yaml"
            config.write_text(
                (Path(__file__).resolve().parents[2] / "recipes/livecodebench/benchmark.yaml").read_text()
            )
            with patch("sys.argv", ["run.py", "evaluate", "-c", str(config)]):
                with self.assertRaises(SystemExit) as error:
                    recipe.main()
            self.assertEqual(error.exception.code, 2)
