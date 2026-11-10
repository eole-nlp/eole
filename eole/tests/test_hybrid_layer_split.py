"""Decoder-owned split math and positive graph capability contracts."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from eole.config.models import TransformerDecoderConfig
from eole.decoders.transformer import TransformerDecoderLayer
from eole.modules.hybrid_graph import HybridLayerGraph


def layer_config(layer_type="full_attention", **changes):
    values = dict(
        layers=1,
        hidden_size=32,
        heads=2,
        transformer_ff=64,
        num_experts=4,
        num_experts_per_tok=2,
        moe_transformer_ff=32,
        num_shared_experts=1,
        shared_expert_gate=True,
        mlp_activation_fn="gated-silu",
        layer_types=[layer_type],
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
    )
    values.update(changes)
    return TransformerDecoderConfig(**values)


def make_layer(config, cls=TransformerDecoderLayer):
    running = SimpleNamespace(parallel_gpu=1, dropout=[0.0], use_ckpting=[])
    layer = cls(config, 0, running).eval()
    for parameter in layer.parameters():
        nn.init.normal_(parameter, std=0.1)
    return layer


class TestHybridLayerSplit(unittest.TestCase):
    def test_split_matches_ordinary_layer_for_supported_layouts(self):
        for kind in ("full_attention", "linear_attention"):
            layer = make_layer(layer_config(kind))
            self.assertTrue(layer.supports_hybrid_graph)
            for length in (1, 4):
                with self.subTest(kind=kind, length=length):
                    x = torch.randn(1, length, 32)

                    # Isolate residual/norm/routing/shared-expert math from the
                    # CUDA-only attention implementations in this CPU test.
                    def attention(inputs, **kwargs):
                        return layer.input_layernorm(inputs), inputs.tanh() * 0.5, None

                    with torch.inference_mode(), patch.object(layer, "_self_attention", side_effect=attention), patch(
                        "eole.modules.expert_execution._triton_moe", False
                    ):
                        expected, _ = layer._forward_eager(x)
                        residual, ff, weights, ids = layer._hybrid_prefix(x)
                        routed = layer.mlp._executor.execute(ff.view(-1, 32), weights, ids, training=False)
                        actual = layer._hybrid_suffix(residual, ff, routed)
                    self.assertTrue(torch.equal(actual, expected))

    def test_alternate_residual_layouts_do_not_opt_in(self):
        for changes in (
            {"parallel_residual": True},
            {"ffn_layernorm": True},
            {"with_cross_attn": True},
            {"num_experts": 0},
        ):
            with self.subTest(changes=changes):
                layer = make_layer(layer_config(**changes))
                self.assertFalse(layer.supports_hybrid_graph)
                with self.assertRaisesRegex(ValueError, "does not support hybrid split"):
                    HybridLayerGraph(layer)

    def test_new_layer_and_attention_implementations_require_opt_in(self):
        class NewResidualLayer(TransformerDecoderLayer):
            pass

        with self.subTest(kind="new decoder class"):
            layer = make_layer(layer_config(), NewResidualLayer)
            self.assertFalse(layer.supports_hybrid_graph)
        with self.subTest(kind="unknown sparse attention layout"):
            layer = make_layer(layer_config("sparse_attention"))
            self.assertFalse(layer.supports_hybrid_graph)
            # A sparse implementation without ordinary KV fields must never
            # reach the graph cache-state accessor.
            layer.self_attn = nn.Identity()
            with self.assertRaisesRegex(ValueError, "does not support hybrid split"):
                HybridLayerGraph(layer)
