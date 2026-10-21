"""Server readiness waits for preparation, with failed engines released."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from eole.server.model import Model


class TestStartupWarmup(unittest.TestCase):
    def model(self, warmup=True):
        model = Model(model_id="qwen", model_path="local", warmup=warmup)
        model.config = SimpleNamespace(world_size=1, beam_size=1, data_type="image", self_speculative_num_tokens=3)
        model.maybe_retrieve_model = Mock()
        model.get_config = Mock()
        model.apply_chat_template = Mock(return_value="templated warmup")
        return model

    def test_load_drains_worker_warmup_before_marking_model_loaded(self):
        model = self.model()
        engine = Mock()

        def prepare(source, settings, cancel_event):
            self.assertFalse(model.loaded)
            self.assertEqual(source, {"text": "templated warmup", "images": {}})
            self.assertEqual(settings, {"max_length": 32, "max_length_ratio": 0})
            self.assertFalse(cancel_event.is_set())
            yield "ignored"
            self.assertFalse(model.loaded)

        engine.infer_list_stream.side_effect = prepare
        with patch("eole.server.model.InferenceEnginePY", return_value=engine):
            model.load()
        self.assertTrue(model.loaded)
        self.assertIs(model.engine, engine)
        engine.terminate.assert_not_called()

    def test_failed_warmup_closes_engine_and_does_not_mark_ready(self):
        model = self.model()
        engine = Mock()
        engine.infer_list_stream.side_effect = RuntimeError("warmup failed")
        with patch("eole.server.model.InferenceEnginePY", return_value=engine):
            with self.assertRaisesRegex(RuntimeError, "warmup failed"):
                model.load()
        engine.terminate.assert_called_once()
        self.assertFalse(model.loaded)
        self.assertIsNone(model.engine)

    def test_loading_without_warmup_preserves_lazy_behavior(self):
        model = self.model(warmup=False)
        with patch("eole.server.model.InferenceEnginePY") as factory:
            model.load()
        factory.return_value.infer_list_stream.assert_not_called()
        self.assertTrue(model.loaded)


if __name__ == "__main__":
    unittest.main()
