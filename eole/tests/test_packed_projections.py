"""Inference wiring, fallback and CUDA equivalence for Marlin input packing."""

import builtins
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from eole.config.models import TransformerDecoderConfig
from eole.models.model import BaseModel
from eole.modules.gated_delta_net import GatedDeltaNet
from eole.modules.marlin_linear import MarlinQuantLinear
from eole.modules.transformer_mlp import MLP
from eole.predict.inference import Inference
from eole.ops import _CPP_OPS_AVAILABLE


def config(**kwargs):
    return SimpleNamespace(
        compute_dtype=torch.float16,
        quant_type="autoround",
        parallel_gpu=1,
        batch_size=1,
        beam_size=1,
        **kwargs,
    )


def projection(n, bias=False):
    layer = MarlinQuantLinear(4, 128, False, True, 128, n, bias=bias)
    layer.qweight.data.random_(0, 2**31 - 1)
    layer.scales.data.uniform_(0.01, 0.05)
    layer.qzeros.data.zero_()
    layer.g_idx.data.zero_()
    if bias:
        layer.bias.uniform_(-0.1, 0.1)
    return layer


def block():
    decoder = TransformerDecoderConfig(
        hidden_size=128,
        heads=4,
        transformer_ff=128,
        mlp_activation_fn="gated-silu",
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
    )
    layer = nn.Module()
    layer.self_attn = None
    layer.linear_attn = GatedDeltaNet(decoder, 0).eval()
    layer.linear_attn.in_proj_qkv = projection(192)
    layer.linear_attn.in_proj_z = projection(64)
    layer.mlp = MLP(decoder).eval()
    layer.mlp.gate_up_proj = projection(128, bias=True)
    layer.mlp.up_proj = projection(128, bias=True)
    layer.mlp.down_proj.reset_parameters()
    return layer


def model(layer):
    result = nn.Module()
    result.decoder = nn.Module()
    result.decoder.transformer_layers = nn.ModuleList([layer])
    return result


class TestPackedProjectionSetup(unittest.TestCase):
    def test_flags_defer_and_finalize_only_their_projections(self):
        for kvq in (False, True):
            for gate in (False, True):
                with self.subTest(kvq=kvq, gate=gate):
                    layer = block()
                    root = model(layer)
                    cfg = config(fuse_kvq=kvq, fuse_gate=gate)
                    with patch.object(MarlinQuantLinear, "post_init", autospec=True) as initialize:
                        BaseModel._prepare_for_inference(root, cfg)
                        self.assertEqual(initialize.call_count, 4 - 2 * kvq - 2 * gate)
                        Inference._prepare_inference_projections(SimpleNamespace(model=root), cfg)
                        self.assertEqual(initialize.call_count, 4 - kvq - gate)
                        count = initialize.call_count
                        Inference._prepare_inference_projections(SimpleNamespace(model=root), cfg)
                        self.assertEqual(initialize.call_count, count)
                    self.assertEqual(layer.linear_attn.fused_qkvz, kvq)
                    self.assertEqual(hasattr(layer.mlp, "up_proj"), not gate)
                    self.assertTrue(
                        all(not getattr(m, "_projection_post_init_deferred", False) for m in root.modules())
                    )

    def test_missing_flags_and_non_marlin_gdn_are_unchanged(self):
        layer = block()
        with patch.object(MarlinQuantLinear, "post_init", autospec=True):
            root = model(layer)
            BaseModel._prepare_for_inference(root, config())
            Inference._prepare_inference_projections(SimpleNamespace(model=root), config())
        self.assertFalse(layer.linear_attn.fused_qkvz)
        self.assertTrue(hasattr(layer.mlp, "up_proj"))
        layer.linear_attn.in_proj_qkv = nn.Linear(128, 192)
        layer.linear_attn.in_proj_z = nn.Linear(128, 64)
        layer.linear_attn._fuse_QKVZ()
        self.assertFalse(layer.linear_attn.fused_qkvz)

    def test_mixed_pairs_skip_once_and_finalize_originals(self):
        for name in ("mlp", "linear_attn"):
            layer = block()
            if name == "mlp":
                layer.mlp.up_proj = nn.Linear(128, 128, bias=True)
                original = layer.mlp.gate_up_proj
            else:
                layer.linear_attn.in_proj_z.group_size = 64
                original = layer.linear_attn.in_proj_qkv
            root = model(layer)
            cfg = config(fuse_kvq=name == "linear_attn", fuse_gate=name == "mlp")
            with patch.object(MarlinQuantLinear, "post_init", autospec=True) as initialize, patch(
                "eole.utils.logging.logger.warning"
            ) as warning:
                BaseModel._prepare_for_inference(root, cfg)
                Inference._prepare_inference_projections(SimpleNamespace(model=root), cfg)
                Inference._prepare_inference_projections(SimpleNamespace(model=root), cfg)
                warning.assert_called_once()
                self.assertIn(original, [args.args[0] for args in initialize.call_args_list])
            self.assertFalse(layer.linear_attn.fused_qkvz)
            self.assertTrue(hasattr(layer.mlp, "up_proj"))

    def test_moe_expert_stacking_sees_initialized_weights(self):
        from eole.modules.autoround_linear import post_init_autoround_linear

        class Experts(nn.Module):
            def __init__(self):
                super().__init__()
                self.experts = nn.ModuleList([block().mlp])
                self.initialized = False

            def _maybe_init_quant_weights(self, device, dtype):
                assert all(m.workspace is not None for m in self.modules() if isinstance(m, MarlinQuantLinear))
                self.initialized = True

        root = nn.Module()
        root.moe = Experts()
        with patch.object(MarlinQuantLinear, "post_init", lambda m: setattr(m, "workspace", torch.zeros(1))):
            post_init_autoround_linear(root, defer_kvq=True, defer_gate=True)
        self.assertTrue(root.moe.initialized)

    def test_dense_gate_fusion_needs_no_marlin_import(self):
        layer = block().mlp
        layer.gate_up_proj = nn.Linear(128, 128)
        layer.up_proj = nn.Linear(128, 128)
        layer.model_config.add_ffnbias = True
        x = torch.randn(2, 3, 128)
        expected = layer(x)
        original_import = builtins.__import__

        def import_without_marlin(name, *args, **kwargs):
            if name == "eole.modules.marlin_linear":
                raise ImportError("not installed")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_without_marlin):
            layer._fuse_gate()
            layer._fuse_gate()
        torch.testing.assert_close(layer(x), expected)


@unittest.skipUnless(torch.cuda.is_available() and _CPP_OPS_AVAILABLE, "Eole CUDA extension required")
class TestPackedProjectionCUDA(unittest.TestCase):
    def test_gate_and_qkvz_outputs_compile_and_replay(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype), torch.inference_mode():
                layer = block().cuda().to(dtype)
                mlp, gdn = layer.mlp, layer.linear_attn
                originals = [mlp.gate_up_proj, mlp.up_proj, gdn.in_proj_qkv, gdn.in_proj_z]
                mlp._fuse_gate()
                gdn._fuse_QKVZ()
                for module in originals + [mlp.gate_up_proj, gdn.in_proj_qkvz]:
                    module.post_init()
                for shape in ((1, 1, 128), (1, 4, 128), (4, 1, 128), (2, 17, 128)):
                    x = torch.randn(shape, device="cuda", dtype=dtype)
                    gate, up, qkv, z = [module(x) for module in originals]
                    torch.testing.assert_close(
                        mlp(x), mlp.down_proj(torch.nn.functional.silu(gate) * up), atol=0.02, rtol=0.02
                    )
                    actual_qkv, actual_z = gdn._project_qkvz(x)
                    torch.testing.assert_close(actual_qkv, qkv.transpose(1, 2), atol=0.01, rtol=0.01)
                    torch.testing.assert_close(actual_z, z, atol=0.01, rtol=0.01)
                compiled = torch.compile(gdn._project_qkvz, fullgraph=True, options={"triton.cudagraphs": False})
                x = torch.randn(1, 1, 128, device="cuda", dtype=dtype)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        compiled(x)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = compiled(x)
                for _ in range(3):
                    x.normal_()
                    graph.replay()
                    expected = (originals[2](x).transpose(1, 2), originals[3](x))
                    for a, b in zip(output, expected):
                        torch.testing.assert_close(a, b, atol=0.01, rtol=0.01)
                torch._dynamo.reset()
