"""Numerical and capture checks for the four-row GDN inference operator."""

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from eole.modules.gdn_control import gdn_control_pair
from eole.tests import test_gdn_control_projection as control_tests


class TestControlPacking(unittest.TestCase):
    def test_pair_order_and_noncontiguous_fallback(self):
        layer = control_tests.TestGDNControlProjection().new_layer()
        layer.in_proj_a.reset_parameters()
        layer.in_proj_b.reset_parameters()
        x = torch.randn(1, 4, 16)
        layer._control_pair_weight = torch.cat((layer.in_proj_b.weight, layer.in_proj_a.weight)).detach()
        with patch.object(layer, "_should_split_control_rows", return_value=True):
            with patch("eole.modules.gated_delta_net.gdn_control_pair", side_effect=torch.nn.functional.linear) as pair:
                b, a = layer._project_controls(x)
                pair.assert_called_once()
                torch.testing.assert_close(b, layer.in_proj_b(x))
                torch.testing.assert_close(a, layer.in_proj_a(x))
            x = torch.randn(1, 4, 32)[:, :, ::2]
            with patch("eole.modules.gated_delta_net.gdn_control_pair") as pair:
                b, a = layer._project_controls(x)
                pair.assert_not_called()
                torch.testing.assert_close(b, layer.in_proj_b(x))
                torch.testing.assert_close(a, layer.in_proj_a(x))

    def test_inference_setup_only_packs_single_gpu_mtp(self):
        from eole.predict.inference import Inference

        layer = control_tests.TestGDNControlProjection().new_layer()
        block = SimpleNamespace(linear_attn=layer, self_attn=None, mlp=SimpleNamespace())
        model = SimpleNamespace(decoder=SimpleNamespace(transformer_layers=[block]))
        config = SimpleNamespace(
            fuse_gate=False, self_speculative_decoding=True, parallel_gpu=1, batch_size=1, beam_size=1
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
            with patch.object(layer, "prepare_control_projections") as prepare:
                Inference._prepare_inference_projections(predictor, config)
                prepare.assert_not_called()
            setattr(config, field, value)
        with patch.object(layer, "prepare_control_projections") as prepare:
            Inference._prepare_inference_projections(predictor, config)
            prepare.assert_called_once()

    def test_cpu_and_bias_fallback(self):
        layer = control_tests.TestGDNControlProjection().new_layer()
        layer.prepare_control_projections()
        self.assertIsNone(layer._control_pair_weight)
        self.assertNotIn("_control_pair_weight", layer.state_dict())
        layer.in_proj_a = nn.Identity()
        layer.prepare_control_projections()
        self.assertIsNone(layer._control_pair_weight)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestControlKernel(unittest.TestCase):
    def test_fp64_reference_compile_and_graph_replay(self):
        torch.manual_seed(109)
        x = torch.randn(1, 4, 5120, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(96, 5120, device="cuda", dtype=torch.bfloat16) / 72

        # FP64 accumulation distinguishes real errors from BF16 rounding boundaries.
        def check(output):
            reference = torch.nn.functional.linear(x.double(), w.double()).float()
            torch.testing.assert_close(output.float(), reference, rtol=0.004, atol=0.0001)

        with torch.inference_mode():
            check(gdn_control_pair(x, w))
            compiled = torch.compile(gdn_control_pair, fullgraph=True, options={"triton.cudagraphs": False})
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
                self._verifier_cache_config = {"paired_controls": True}

            def _forward_verify_graph(self, emb, *, tgt_pad_mask):
                return gdn_control_pair(emb, self.weight).masked_fill(tgt_pad_mask, 0)

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
                "torch.compiler.load_compiled_function", wraps=torch.compiler.load_compiled_function
            ) as load:
                torch.testing.assert_close(second(x, tgt_pad_mask=mask), x.new_zeros((1, 4, 96)))
                load.assert_called_once()
                self.assertEqual(compile_call.call_count, 1)  # fallback construction only; no retracing
        torch._dynamo.reset()

    def test_invalid_shape_rejected(self):
        with self.assertRaises(ValueError):
            gdn_control_pair(
                torch.zeros(1, 3, 5120, device="cuda", dtype=torch.bfloat16),
                torch.zeros(96, 5120, device="cuda", dtype=torch.bfloat16),
            )

    def test_pack_order_and_device_policy(self):
        layer = control_tests.TestGDNControlProjection().new_layer(hidden_size=5120, heads=48).cuda().bfloat16()
        with torch.no_grad():
            layer.in_proj_a.weight.fill_(2)
            layer.in_proj_b.weight.fill_(1)
        layer.prepare_control_projections()
        if torch.cuda.get_device_capability() == (12, 0):
            self.assertIsNotNone(layer._control_pair_weight)
            torch.testing.assert_close(layer._control_pair_weight[:48], layer.in_proj_b.weight)
            torch.testing.assert_close(layer._control_pair_weight[48:], layer.in_proj_a.weight)
            self.assertFalse(layer._control_pair_weight.requires_grad)
            self.assertNotIn("_control_pair_weight", layer.state_dict())
            layer.in_proj_a = nn.Linear(5120, 48, bias=True, device="cuda", dtype=torch.bfloat16)
            layer.prepare_control_projections()
            self.assertIsNone(layer._control_pair_weight)
        else:
            self.assertIsNone(layer._control_pair_weight)


if __name__ == "__main__":
    unittest.main()
