import argparse
import unittest

import torch

from eole.bin.convert.convert_COMET import CometConverter, _build_model_config, _cast_state_dict


def _encoder_config(architecture="XLMRobertaForMaskedLM"):
    return {
        "architectures": [architecture],
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
    }


class TestConvertComet(unittest.TestCase):
    def test_cast_state_dict_casts_only_floating_tensors(self):
        state_dict = {
            "weight": torch.ones(2, dtype=torch.float32),
            "ids": torch.ones(2, dtype=torch.int64),
        }

        converted = _cast_state_dict(state_dict, torch.float16)

        self.assertEqual(converted["weight"].dtype, torch.float16)
        self.assertEqual(converted["ids"].dtype, torch.int64)
        self.assertTrue(converted["weight"].is_contiguous())

    def test_dtype_arg_defaults_to_fp32(self):
        parser = argparse.ArgumentParser()
        CometConverter.add_args(parser)

        args = parser.parse_args(["--model", "Unbabel/wmt22-comet-da"])

        self.assertEqual(args.dtype, "fp32")

    def test_xcomet_conversion_does_not_require_reference(self):
        config = _build_model_config(
            "Unbabel/XCOMET-XL",
            {"class_identifier": "xcomet_metric", "input_segments": ["mt", "src", "ref"]},
            _encoder_config("XLMRobertaXLForMaskedLM"),
        )

        self.assertFalse(config["requires_reference"])
        self.assertEqual(config["input_segments"], ["mt", "src", "ref"])

    def test_regression_metric_conversion_still_requires_reference(self):
        config = _build_model_config(
            "Unbabel/wmt22-comet-da",
            {"class_identifier": "regression_metric", "input_segments": ["mt", "src", "ref"]},
            _encoder_config(),
        )

        self.assertTrue(config["requires_reference"])


if __name__ == "__main__":
    unittest.main()
