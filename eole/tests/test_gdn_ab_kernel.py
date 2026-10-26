"""Numerical and capture checks for the four-row GDN A/B inference operator."""

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from eole.modules.gdn_ab import gdn_ab_projections
from eole.tests import test_gdn_control_projection as projection_tests


class TestABPacking(unittest.TestCase):
    def test_pair_order_and_noncontiguous_fallback(self):
        layer = projection_tests.TestGDNControlProjection().new_layer()
        layer.in_proj_a.reset_parameters()
        layer.in_proj_b.reset_parameters()
        x = torch.randn(1, 4, 16)
        layer._ab_pair_weight = torch.cat((layer.in_proj_b.weight, layer.in_proj_a.weight)).detach()
        with patch.object(layer, "_should_pair_ab_projections", side_effect=lambda x: x.is_contiguous()):
            with patch(
                "eole.modules.gated_delta_net.gdn_ab_projections",
                side_effect=torch.nn.functional.linear,
            ) as pair:
                b, a = layer._project_ab(x)
                pair.assert_called_once()
                torch.testing.assert_close(b, layer.in_proj_b(x))
                torch.testing.assert_close(a, layer.in_proj_a(x))
            x = torch.randn(1, 4, 32)[:, :, ::2]
            with patch("eole.modules.gated_delta_net.gdn_ab_projections") as pair:
                b, a = layer._project_ab(x)
                pair.assert_not_called()
                torch.testing.assert_close(b, layer.in_proj_b(x))
                torch.testing.assert_close(a, layer.in_proj_a(x))

    def test_inference_setup_only_packs_single_gpu_mtp(self):
        from eole.predict.inference import Inference

        layer = projection_tests.TestGDNControlProjection().new_layer()
        block = SimpleNamespace(linear_attn=layer, self_attn=None, mlp=SimpleNamespace())
        model = SimpleNamespace(decoder=SimpleNamespace(transformer_layers=[block]))
        config = SimpleNamespace(
            fuse_gate=False,
            self_speculative_decoding=True,
            parallel_gpu=1,
            batch_size=1,
            beam_size=1,
        )
        predictor = SimpleNamespace(model=model)
        for field, disabled in [
            ("self_speculative_decoding", False),
            ("parallel_gpu", 2),
            ("batch_size", 4),
            ("beam_size", 4),
        ]:
            value = getattr(config, field)
            setattr(config, field, disabled)
            with patch.object(layer, "prepare_ab_projections") as prepare:
                Inference._prepare_inference_projections(predictor, config)
                prepare.assert_not_called()
            setattr(config, field, value)
        with patch.object(layer, "prepare_ab_projections") as prepare:
            Inference._prepare_inference_projections(predictor, config)
            prepare.assert_called_once()

    def test_cpu_and_bias_fallback(self):
        layer = projection_tests.TestGDNControlProjection().new_layer()
        layer.prepare_ab_projections()
        self.assertIsNone(layer._ab_pair_weight)
        self.assertNotIn("_ab_pair_weight", layer.state_dict())
        layer.in_proj_a = nn.Identity()
        layer.prepare_ab_projections()
        self.assertIsNone(layer._ab_pair_weight)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestABKernel(unittest.TestCase):
    def test_fp64_reference_compile_and_graph_replay(self):
        torch.manual_seed(109)
        x = torch.randn(1, 4, 5120, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(96, 5120, device="cuda", dtype=torch.bfloat16) / 72

        # FP64 accumulation distinguishes real errors from BF16 rounding boundaries.
        def check(output):
            reference = torch.nn.functional.linear(x.double(), w.double()).float()
            torch.testing.assert_close(output.float(), reference, rtol=0.004, atol=0.0001)

        with torch.inference_mode():
            check(gdn_ab_projections(x, w))
            compiled = torch.compile(gdn_ab_projections, fullgraph=True, options={"triton.cudagraphs": False})
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    compiled(x, w)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = compiled(x, w)
            for _ in range(4):
                x.copy_(torch.randn_like(x))
                graph.replay()
                check(output)
        torch._dynamo.reset()

    @unittest.skipUnless(hasattr(torch.compiler, "load_compiled_function"), "AOT save/load required")
    def test_saved_verifier_rebinds_packed_buffer(self):
        from eole.utils.verifier_cache import CachedVerifier

        class Verifier(nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.register_buffer("weight", weight, persistent=False)
                self._verifier_cache_config = {"paired_ab_projections": True}

            def _forward_verify_graph(self, emb, *, tgt_pad_mask):
                return gdn_ab_projections(emb, self.weight).masked_fill(tgt_pad_mask, 0)

        x = torch.ones(1, 4, 5120, device="cuda", dtype=torch.bfloat16)
        mask = torch.zeros(1, 4, 1, device="cuda", dtype=torch.bool)
        weight = torch.ones(96, 5120, device="cuda", dtype=torch.bfloat16)
        other_weight = torch.zeros_like(weight)
        with tempfile.TemporaryDirectory() as directory, torch.inference_mode():
            first = CachedVerifier(Verifier(weight), directory)
            torch.testing.assert_close(first(x, tgt_pad_mask=mask), x.new_full((1, 4, 96), 5120))
            self.assertEqual(len(list(Path(directory).glob("*/verifier.aot"))), 1)
            second = CachedVerifier(Verifier(other_weight), directory)
            with patch("torch.compile", wraps=torch.compile) as compile_call, patch(
                "torch.compiler.load_compiled_function",
                wraps=torch.compiler.load_compiled_function,
            ) as load:
                torch.testing.assert_close(second(x, tgt_pad_mask=mask), x.new_zeros((1, 4, 96)))
                load.assert_called_once()
                self.assertEqual(compile_call.call_count, 1)  # fallback construction only; no retracing
        torch._dynamo.reset()

    def test_invalid_shape_rejected(self):
        with self.assertRaises(ValueError):
            gdn_ab_projections(
                torch.zeros(1, 3, 5120, device="cuda", dtype=torch.bfloat16),
                torch.zeros(96, 5120, device="cuda", dtype=torch.bfloat16),
            )

    def test_pair_guards_and_pre_ampere_or_missing_triton_fallback(self):
        layer = projection_tests.TestGDNControlProjection().new_layer(hidden_size=5120, heads=48).cuda().bfloat16()
        with patch("torch.cuda.get_device_capability", return_value=(7, 5)):
            layer.prepare_ab_projections()
            self.assertIsNone(layer._ab_pair_weight)
        with patch("importlib.util.find_spec", return_value=None):
            layer.prepare_ab_projections()
            self.assertIsNone(layer._ab_pair_weight)
        for capability in [(8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (10, 3), (12, 0)]:
            with self.subTest(capability=capability), patch(
                "torch.cuda.get_device_capability", return_value=capability
            ):
                layer.prepare_ab_projections()
                self.assertIsNotNone(layer._ab_pair_weight)
        layer.prepare_ab_projections()
        if torch.cuda.get_device_capability()[0] < 8:
            return
        layer._speculating = True
        x = torch.zeros(1, 4, 5120, device="cuda", dtype=torch.bfloat16)
        self.assertTrue(layer._should_pair_ab_projections(x))
        # Pairing is independent of the legacy SM120-only GEMV fallback.
        layer._control_gemv_enabled = False
        self.assertTrue(layer._should_pair_ab_projections(x))
        for shape in [(1, 1, 5120), (2, 4, 5120), (1, 3, 5120), (1, 4, 2560)]:
            self.assertFalse(layer._should_pair_ab_projections(torch.zeros(shape, device="cuda", dtype=x.dtype)))
        self.assertFalse(layer._should_pair_ab_projections(x.float()))
        self.assertFalse(
            layer._should_pair_ab_projections(torch.zeros(1, 4, 10240, device="cuda", dtype=x.dtype)[:, :, ::2])
        )
        layer.train()
        self.assertFalse(layer._should_pair_ab_projections(x))
        layer.eval()
        layer._speculating = False
        self.assertFalse(layer._should_pair_ab_projections(x))

    def test_pack_order_and_device_policy(self):
        layer = projection_tests.TestGDNControlProjection().new_layer(hidden_size=5120, heads=48).cuda().bfloat16()
        with torch.no_grad():
            layer.in_proj_a.weight.fill_(2)
            layer.in_proj_b.weight.fill_(1)
        layer.prepare_ab_projections()
        if torch.cuda.get_device_capability()[0] >= 8:
            self.assertIsNotNone(layer._ab_pair_weight)
            torch.testing.assert_close(layer._ab_pair_weight[:48], layer.in_proj_b.weight)
            torch.testing.assert_close(layer._ab_pair_weight[48:], layer.in_proj_a.weight)
            self.assertFalse(layer._ab_pair_weight.requires_grad)
            self.assertNotIn("_ab_pair_weight", layer.state_dict())
            layer.in_proj_a = nn.Linear(5120, 48, bias=True, device="cuda", dtype=torch.bfloat16)
            layer.prepare_ab_projections()
            self.assertIsNone(layer._ab_pair_weight)
        else:
            self.assertIsNone(layer._ab_pair_weight)


if __name__ == "__main__":
    unittest.main()
