"""Selection, inference wiring and numerical checks for opt-in dense INT8."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from eole.config.inference import InferenceConfig
from eole.modules.dense_quantization import quantize_decoder_layer
from eole.modules.vocabulary_quantization import quantize_linear, quantize_rows
from eole.predict.inference import Inference
from eole.tests.test_vocabulary_quantization import dequantize


class TestDenseQuantization(unittest.TestCase):
    def test_roles_preserve_experts_controls_and_existing_quantization(self):
        layer = nn.Module()
        layer.mlp = nn.Module()
        layer.mlp.gate = nn.Linear(128, 64, bias=False)
        layer.mlp.experts = nn.ModuleList([nn.Module()])
        layer.mlp.experts[0].up_proj = nn.Linear(128, 64, bias=False)
        layer.linear_attn = nn.Module()
        layer.linear_attn.in_proj_a = nn.Linear(128, 64, bias=False)
        layer.linear_attn.in_proj_qkv = nn.Linear(128, 64, bias=False)
        layer.mlp.shared_experts = nn.Module()
        layer.mlp.shared_experts.up_proj = nn.Linear(128, 64, bias=False)
        layer.mlp.down_proj = nn.Identity()  # Already quantized/non-Linear stays untouched.
        original = dict(layer.named_modules())
        with patch("eole.modules.dense_quantization.quantize_linear") as quantize:
            self.assertEqual(quantize_decoder_layer(layer), 0)  # CPU is retained.
            quantize.assert_not_called()
        for name, module in original.items():
            self.assertIs(layer.get_submodule(name), module)

    def test_disabled_is_default_and_tensor_parallel_rejected_before_mutation(self):
        self.assertEqual(InferenceConfig().dense_quantization, "none")
        root = nn.Module()
        with patch("eole.modules.hybrid_experts.prepare_hybrid_experts") as prepare:
            with self.assertRaisesRegex(ValueError, "single-GPU"):
                Inference._prepare_inference_projections(
                    SimpleNamespace(model=root),
                    SimpleNamespace(dense_quantization="int8", parallel_gpu=2),
                )
            prepare.assert_not_called()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_selection_fusion_replay_and_prefill(self):
        from eole.modules.autoround_linear import finalize_deferred_autoround_linear
        from eole.modules.marlin_linear import MarlinQuantLinear
        from eole.tests.test_packed_projections import block, model, config

        torch.manual_seed(123)
        layer = block().cuda().bfloat16().eval()
        # Replace the existing quantized GDN inputs with native inputs.
        for name, size in (("in_proj_qkv", 192), ("in_proj_z", 64)):
            setattr(
                layer.linear_attn,
                name,
                nn.Linear(128, size, bias=False, device="cuda", dtype=torch.bfloat16),
            )
        layer.mlp.gate = nn.Linear(128, 64, bias=False, device="cuda", dtype=torch.bfloat16)
        layer.mlp.experts = nn.ModuleList([nn.Module()])
        expert = nn.Linear(128, 64, bias=False, device="cuda", dtype=torch.bfloat16)
        layer.mlp.experts[0].up_proj = expert
        controls = (
            layer.linear_attn.in_proj_a,
            layer.linear_attn.in_proj_b,
            layer.mlp.gate,
        )
        sources = [layer.linear_attn.in_proj_qkv, layer.linear_attn.in_proj_z]
        references = []
        for source in sources:
            packed, scales = quantize_rows(source.weight.cpu(), 8, 32)
            references.append(dequantize(packed, scales, 8, 32).cuda().bfloat16())
        root = model(layer)
        Inference._prepare_inference_projections(
            SimpleNamespace(model=root),
            config(fuse_kvq=True, fuse_gate=True, dense_quantization="int8"),
        )
        self.assertTrue(layer.linear_attn.fused_qkvz)
        self.assertIsInstance(layer.linear_attn.in_proj_qkvz, MarlinQuantLinear)
        self.assertEqual(layer.linear_attn.in_proj_qkvz.bits, 8)
        self.assertEqual(layer.mlp.gate_up_proj.bits, 4)
        self.assertEqual(
            controls,
            (layer.linear_attn.in_proj_a, layer.linear_attn.in_proj_b, layer.mlp.gate),
        )
        self.assertIs(layer.mlp.experts[0].up_proj, expert)
        self.assertFalse(finalize_deferred_autoround_linear(root))
        weight = torch.cat(references)
        projection = layer.linear_attn.in_proj_qkvz
        for rows in (1, 4):
            x = torch.randn(1, rows, 128, device="cuda", dtype=torch.bfloat16)
            torch.testing.assert_close(projection(x), nn.functional.linear(x, weight), atol=0.04, rtol=0.04)
        x = torch.randn(1, 1, 128, device="cuda", dtype=torch.bfloat16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            projection(x)
            projection(x)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = projection(x)
        for _ in range(2):
            x.normal_()
            graph.replay()
            torch.testing.assert_close(output, nn.functional.linear(x, weight), atol=0.04, rtol=0.04)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_deferred_matches_immediate(self):
        from eole.modules.autoround_linear import finalize_deferred_autoround_linear

        source = nn.Linear(256, 128, bias=False, device="cuda", dtype=torch.bfloat16)
        immediate = quantize_linear(source, 8, 32)
        deferred = quantize_linear(source, 8, 32, defer_post_init=True)
        finalize_deferred_autoround_linear(deferred)
        x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
        torch.testing.assert_close(deferred(x), immediate(x), atol=0, rtol=0)
