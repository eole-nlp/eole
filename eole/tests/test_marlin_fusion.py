import unittest

import torch

from eole.modules.marlin_linear import MarlinQuantLinear


class TestMarlinFusion(unittest.TestCase):
    def projections(self, desc_act=False):
        layers = [MarlinQuantLinear(4, 128, desc_act, True, 128, n, bias=True) for n in (64, 64, 128)]
        for i, layer in enumerate(layers):
            layer.qweight.data.fill_(i)
            layer.qzeros.data.fill_(i)
            layer.scales.data.fill_(i + 1)
            layer.g_idx.data.zero_()
            layer.bias.fill_(i)
        return layers

    def test_output_columns_and_bias_preserve_kvq_order(self):
        layers = self.projections()
        fused = MarlinQuantLinear.fuse(layers)
        self.assertEqual(fused.out_features, 256)
        for name in ("qweight", "qzeros", "scales"):
            torch.testing.assert_close(getattr(fused, name), torch.cat([getattr(p, name) for p in layers], dim=1))
        torch.testing.assert_close(fused.bias, torch.cat([p.bias for p in layers]))
        self.assertIsNone(fused.workspace)

    def test_attention_fusion_is_idempotent(self):
        from eole.modules.multi_headed_attn import MultiHeadedAttention

        attention = MultiHeadedAttention.__new__(MultiHeadedAttention)
        torch.nn.Module.__init__(attention)
        attention.fused_kvq = False
        layers = self.projections()
        attention.linear_keys, attention.linear_values, attention.linear_query = layers
        attention._fuse_KVQ()
        fused = attention.linear_kvq
        attention._fuse_KVQ()
        self.assertIs(attention.linear_kvq, fused)
        self.assertTrue(attention.fused_kvq)
        self.assertFalse(hasattr(attention, "linear_keys"))
        torch.testing.assert_close(fused.qweight, torch.cat([p.qweight for p in layers], dim=1))

    def test_rejects_repacked_weights(self):
        layers = self.projections()
        layers[0].workspace = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "before post_init"):
            MarlinQuantLinear.fuse(layers)

    def test_rejects_different_activation_order(self):
        layers = self.projections(desc_act=True)
        layers[1].g_idx.data.fill_(1)
        with self.assertRaisesRegex(ValueError, "identical activation ordering"):
            MarlinQuantLinear.fuse(layers)

    def test_rejects_incompatible_groups(self):
        layers = self.projections()
        layers[1].group_size = 64
        with self.assertRaisesRegex(ValueError, "compatible projections"):
            MarlinQuantLinear.fuse(layers)
