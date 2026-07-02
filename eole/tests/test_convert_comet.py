import argparse
import unittest

import torch

from eole.bin.convert.convert_COMET import CometConverter, _cast_state_dict


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


if __name__ == "__main__":
    unittest.main()
