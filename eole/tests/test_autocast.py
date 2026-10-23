import unittest
from contextlib import nullcontext

import torch

from eole.utils.misc import get_autocast


class TestGetAutocast(unittest.TestCase):
    def test_disabled_autocast_ignores_dtype(self):
        ctx = get_autocast(enabled=False, device_type="cuda", dtype=torch.bfloat16)

        self.assertIsInstance(ctx, nullcontext)

    def test_cuda_autocast_defaults_to_fp16(self):
        ctx = get_autocast(enabled=True, device_type="cuda")

        self.assertEqual(ctx.fast_dtype, torch.float16)

    def test_cuda_autocast_keeps_fp16(self):
        ctx = get_autocast(enabled=True, device_type="cuda", dtype=torch.float16)

        self.assertEqual(ctx.fast_dtype, torch.float16)

    def test_cuda_autocast_uses_bf16(self):
        ctx = get_autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16)

        self.assertEqual(ctx.fast_dtype, torch.bfloat16)
