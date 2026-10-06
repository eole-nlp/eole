"""Check graph integration without executing GPU-only FLA kernels."""

import unittest
from unittest.mock import patch

import torch

from eole.modules import gated_delta_net as gdn


@unittest.skipUnless(hasattr(gdn, "_compiled_fla_causal_conv1d_update"), "FLA convolution adapter unavailable")
class TestFLACompile(unittest.TestCase):
    def test_fullgraph_compile_preserves_declared_cache_mutation(self):
        calls = []

        @torch.compiler.disable
        def fake_fla(x, cache, residual=None, weight=None, bias=None, activation=None):
            calls.append(activation)
            # Same in-place cache contract as FLA's convolution update.
            cache.copy_(torch.cat([cache[..., 1:], x.unsqueeze(-1)], dim=-1))
            output = (cache * weight.unsqueeze(0)).sum(-1)
            if bias is not None:
                output = output + bias
            if activation == "silu":
                output = torch.nn.functional.silu(output)
            return output, cache

        x = torch.randn(2, 4, 1)
        weight = torch.randn(4, 3)
        bias = torch.randn(4)
        initial = torch.randn(2, 4, 3)
        eager_cache = initial.clone()
        compiled_cache = initial.clone()
        with patch.object(gdn, "_fla_causal_conv1d_update", fake_fla):
            eager = gdn.causal_conv1d_update(x, eager_cache, weight, bias, "silu")
            compiled = torch.compile(gdn.causal_conv1d_update, backend="aot_eager", fullgraph=True)
            actual = compiled(x, compiled_cache, weight, bias, "silu")
            torch.testing.assert_close(actual, eager)
            torch.testing.assert_close(compiled_cache, eager_cache)
            next_x = torch.randn_like(x)
            expected = gdn.causal_conv1d_update(next_x, eager_cache, weight, None, None)
            actual = compiled(next_x, compiled_cache, weight, None, None)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(compiled_cache, eager_cache)
        self.assertEqual(calls, ["silu", "silu", None, None])


@unittest.skipUnless(hasattr(gdn, "_compiled_fla_rms_norm_gated"), "FLA norm adapter unavailable")
class TestFLANormCompile(unittest.TestCase):
    def test_fullgraph_norm_preserves_fla_inputs(self):
        calls = []

        @torch.compiler.disable
        def fake_norm(x, gate, weight, bias, activation, eps=1e-6):
            calls.append((activation, eps))
            output = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
            output = output * weight
            if bias is not None:
                output = output + bias
            return (output * torch.nn.functional.silu(gate.float())).to(x.dtype)

        norm = gdn.RMSNormGated(4, eps=1e-5)
        with torch.no_grad(), patch.object(gdn, "_fla_rms_norm_gated", fake_norm):
            compiled = torch.compile(norm, backend="aot_eager", fullgraph=True)
            for dtype in (torch.float32, torch.bfloat16):
                x = torch.randn(2, 3, 4, dtype=dtype)
                gate = torch.randn_like(x)
                expected = fake_norm(x, gate, norm.weight, norm.bias, norm.activation, eps=norm.eps)
                actual = compiled(x, gate)
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(compiled(x, gate), expected)
        self.assertEqual(calls, [(norm.activation, norm.eps)] * 6)


if __name__ == "__main__":
    unittest.main()
