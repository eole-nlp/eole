import unittest
from types import SimpleNamespace

from eole.bin.convert.convert_HF import build_config_dict
from eole.config.models import VisionEncoderConfig


class TestQwenRopeConversion(unittest.TestCase):
    def test_mrope_frequency_layout_keeps_split_half_coordinate_pairing(self):
        for architecture in (
            "Qwen3VLForConditionalGeneration",
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForConditionalGeneration",
        ):
            for interleaved in (True, False, None):
                with self.subTest(architecture=architecture, interleaved=interleaved):
                    rope = {
                        "mrope_section": [11, 11, 10],
                        "rope_theta": 10000000,
                        "partial_rotary_factor": 0.25,
                    }
                    if interleaved is not None:
                        rope["mrope_interleaved"] = interleaved
                    hf = SimpleNamespace(
                        arch=architecture,
                        config={
                            "text_config": {
                                "hidden_size": 6144,
                                "num_hidden_layers": 64,
                                "num_attention_heads": 24,
                                "head_dim": 256,
                                "rope_parameters": rope,
                            }
                        },
                    )
                    config, _, _ = build_config_dict(hf)
                    self.assertIs(config["rope_config"]["rotary_interleave"], False)
                    self.assertEqual(config["rope_config"]["xdrope_section"], [11, 11, 10])
                    self.assertEqual(config["rope_config"]["rotary_theta"], 10000000)
                    self.assertEqual(config["rope_config"]["rotary_dim"], 64)


class TestDeepSeekOCRVisionConversion(unittest.TestCase):
    def test_nested_clip_config_does_not_produce_zero_width_heads(self):
        hf = SimpleNamespace(
            arch="DeepseekOCRForCausalLM",
            config={
                "hidden_size": 1280,
                "num_hidden_layers": 12,
                "num_attention_heads": 10,
                "vision_config": {"width": {"clip-l-14-224": {"heads": 16, "image_size": 224, "patch_size": 14}}},
            },
        )
        config, _, _ = build_config_dict(hf)
        self.assertIsNone(config["encoder"]["head_dim"])
        self.assertEqual(config["encoder"]["heads"], 16)
        self.assertEqual(config["encoder"]["n_positions"], 257)
        encoder = VisionEncoderConfig(**{**config["encoder"], "hidden_size": 1024})
        self.assertEqual(encoder.dim_per_head, 64)


if __name__ == "__main__":
    unittest.main()
