import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from eole.modules.vocabulary_quantization import quantize_rows, quantize_vocabulary_head


def dequantize(packed, scales, bits, group_size):
    pack = 32 // bits
    shifts = torch.arange(pack) * bits
    values = (packed.long()[:, None, :] >> shifts[None, :, None]) & ((1 << bits) - 1)
    values = values.reshape(-1, packed.size(1)).t().float() - (1 << (bits - 1))
    return values * scales.t().repeat_interleave(group_size, dim=1).float()


class TestVocabularyQuantization(unittest.TestCase):
    def test_default_and_inference_wiring(self):
        from eole.config.inference import InferenceConfig
        from eole.predict.inference import Inference

        self.assertEqual(InferenceConfig().vocabulary_quantization, "none")
        model = torch.nn.Module()
        model.generator = torch.nn.Linear(128, 64, bias=False)
        original = model.generator
        inference = SimpleNamespace(model=model)
        with patch("eole.modules.vocabulary_quantization.quantize_vocabulary_head") as quantize:
            Inference._prepare_inference_projections(inference, InferenceConfig())
            quantize.assert_not_called()
            self.assertIs(model.generator, original)
            replacement = torch.nn.Linear(128, 64, bias=False)
            quantize.return_value = replacement
            Inference._prepare_inference_projections(inference, InferenceConfig(vocabulary_quantization="int4"))
            quantize.assert_called_once_with(original, 4, 128)
            self.assertIs(model.generator, replacement)

    def test_packing_matches_independent_rounding(self):
        torch.manual_seed(11)
        weight = torch.randn(64, 256, dtype=torch.bfloat16)
        for bits in (4, 8):
            for group_size in (32, 64, 128):
                packed, scales = quantize_rows(weight, bits, group_size)
                actual = dequantize(packed, scales, bits, group_size)
                groups = weight.float().reshape(64, -1, group_size)
                limit = (1 << (bits - 1)) - 1
                scale = (groups.abs().amax(-1) / limit).to(weight.dtype).float()
                expected = (groups / scale.unsqueeze(-1)).round().clamp(-limit, limit) * scale.unsqueeze(-1)
                torch.testing.assert_close(actual, expected.reshape_as(actual))

    def test_zero_rows_are_finite(self):
        for bits in (4, 8):
            packed, scales = quantize_rows(torch.zeros(64, 128, dtype=torch.float16), bits, 128)
            restored = dequantize(packed, scales, bits, 128)
            self.assertTrue(torch.isfinite(restored).all())
            self.assertTrue(restored.eq(0).all())

    def test_cpu_request_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "CUDA"):
            quantize_vocabulary_head(torch.nn.Linear(128, 64), 8)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_marlin_matches_dequantized_head_and_preserves_source(self):
        torch.manual_seed(17)
        for bits in (4, 8):
            head = torch.nn.Linear(256, 256, bias=False, dtype=torch.bfloat16, device="cuda")
            original = head.weight.detach().clone()
            packed, scales = quantize_rows(head.weight.detach().cpu(), bits, 128)
            reference = dequantize(packed, scales, bits, 128).cuda().to(torch.bfloat16)
            quantized = quantize_vocabulary_head(head, bits)
            x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
            expected = torch.nn.functional.linear(x, reference, head.bias)
            torch.testing.assert_close(quantized(x), expected, atol=0.02, rtol=0.02)
            torch.testing.assert_close(head.weight, original)
