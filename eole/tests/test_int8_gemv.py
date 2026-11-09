"""Integer reference, storage, batching and capture checks for W8A8 projections."""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from eole.modules.int8_gemv import W8A8Linear, int8_projection
from eole.modules.vocabulary_quantization import quantize_linear


def reference(x, module):
    k, n = module.in_features, module.out_features
    groups = x.float().reshape(-1, k // 32, 32)
    scales = (groups.abs().amax(-1).double() / 127).float().clamp_min(1e-30)
    qx = (groups.double() / scales[..., None].double()).float().round().clamp(-127, 127)
    shifts = torch.arange(4, device=x.device) * 8
    qw = ((module.packed[..., None].long() >> shifts) & 255).reshape(n, k // 32, 32)
    qw = torch.where(qw >= 128, qw - 256, qw)
    if x.numel() // k >= 64:
        from eole.modules.marlin_linear import MarlinQuantLinear
        from eole.modules.int8_gemv import round_int8_inputs

        restored_x = (qx * scales[..., None]).reshape_as(x).to(x.dtype)
        torch.testing.assert_close(round_int8_inputs(x.contiguous()), restored_x, rtol=0, atol=0)
        # Independently pack the same codes with the normal setup path. The
        # temporary-layout prefill must reproduce it exactly, not just be close
        # to a floating-point dequantization with different rounding.
        canonical = MarlinQuantLinear(8, 32, False, True, k, n, False).to(device=x.device, dtype=x.dtype)
        canonical.qweight.data.copy_((module.packed ^ -2139062144).t())
        canonical.scales.data.copy_(module.scales.t())
        canonical.qzeros.data.zero_()
        canonical.post_init()
        return canonical(restored_x)
    dots = (qx[:, None] * qw).sum(-1)
    return (dots * scales[:, None] * module.scales.float()).sum(-1).reshape(*x.shape[:-1], n).to(x.dtype)


class TestW8A8Policy(unittest.TestCase):
    def test_missing_triton_rejected_before_preparation(self):
        from eole.predict.inference import Inference

        with patch("importlib.util.find_spec", return_value=None):
            with patch("eole.modules.hybrid_experts.prepare_hybrid_experts") as prepare:
                with self.assertRaisesRegex(ValueError, "Triton"):
                    Inference._prepare_inference_projections(
                        SimpleNamespace(model=nn.Module()), SimpleNamespace(dense_quantization="w8a8")
                    )
                prepare.assert_not_called()

    def test_invalid_cpu_input_and_fake_shape(self):
        with self.assertRaisesRegex(ValueError, "CUDA"):
            int8_projection(torch.zeros(2, 128), torch.zeros(64, 32, dtype=torch.int32), torch.ones(64, 4))
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            x = torch.empty(2, 3, 2048, device="cuda", dtype=torch.bfloat16)
            y = int8_projection(
                x,
                torch.empty(192, 512, device="cuda", dtype=torch.int32),
                torch.empty(192, 64, device="cuda", dtype=torch.bfloat16),
            )
            self.assertEqual(y.shape, (2, 3, 192))
            self.assertEqual(y.dtype, x.dtype)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestW8A8CUDA(unittest.TestCase):
    def make_module(self, k=2048, dtype=torch.bfloat16):
        source = nn.Linear(k, 128, bias=False, device="cuda", dtype=dtype)
        packed = quantize_linear(source, 8, 32, defer_post_init=True)
        return W8A8Linear(packed).eval()

    def test_integer_reference_shapes_and_single_row_equivalence(self):
        torch.manual_seed(71)
        for k, dtype in ((2048, torch.bfloat16), (5120, torch.bfloat16), (384, torch.float16)):
            module = self.make_module(k, dtype)
            # Exactly one weight layout: no Marlin fallback or native weight copy.
            self.assertEqual(set(module.state_dict()), {"packed", "scales", "workspace", "empty_index"})
            self.assertFalse(hasattr(module, "fallback"))
            for rows in (0, 1, 4, 9, 32, 64):
                x = torch.randn(1, rows, k, device="cuda", dtype=dtype)
                actual = module(x)
                if rows:
                    torch.testing.assert_close(actual, reference(x, module), rtol=0.008, atol=0.0005)
                    if rows < 64:
                        singles = torch.cat([module(x[:, i : i + 1]) for i in range(rows)], 1)
                        torch.testing.assert_close(actual, singles, rtol=0, atol=0)
                else:
                    self.assertEqual(actual.shape, (1, 0, 128))
            x = torch.zeros(1, 4, k, device="cuda", dtype=dtype)
            self.assertTrue(module(x).eq(0).all())
            noncontiguous = torch.randn(1, 4, k * 2, device="cuda", dtype=dtype)[..., ::2]
            torch.testing.assert_close(module(noncontiguous), module(noncontiguous.contiguous()), rtol=0, atol=0)

    def test_compile_capture_replay(self):
        module = self.make_module()
        compiled = torch.compile(module, fullgraph=True, options={"triton.cudagraphs": False})
        for rows in (1, 4, 64):
            x = torch.randn(1, rows, 2048, device="cuda", dtype=torch.bfloat16)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                compiled(x)
                compiled(x)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = compiled(x)
            for _ in range(3):
                x.normal_()
                graph.replay()
                torch.testing.assert_close(output, reference(x, module), rtol=0.008, atol=0.0005)
        torch._dynamo.reset()

    def test_inference_packing_without_requantizing_checkpoint_projections(self):
        from eole.predict.inference import Inference
        from eole.tests.test_packed_projections import block, model, config
        from eole.modules.marlin_linear import MarlinQuantLinear

        layer = block().cuda().bfloat16().eval()
        for name, n in (("in_proj_qkv", 192), ("in_proj_z", 64)):
            setattr(layer.linear_attn, name, nn.Linear(128, n, bias=False, device="cuda", dtype=torch.bfloat16))
        root = model(layer)
        Inference._prepare_inference_projections(
            SimpleNamespace(model=root), config(dense_quantization="w8a8", fuse_kvq=True, fuse_gate=True)
        )
        self.assertIsInstance(layer.linear_attn.in_proj_qkvz, W8A8Linear)
        self.assertIsInstance(layer.mlp.gate_up_proj, MarlinQuantLinear)
        self.assertEqual(layer.mlp.gate_up_proj.bits, 4)
        self.assertFalse(any(getattr(m, "_projection_post_init_deferred", False) for m in root.modules()))

    def test_speculative_target_matches_ordinary_decoding(self):
        from eole.predict.inference import Inference
        from eole.tests.test_packed_projections import config
        from eole.tests.test_speculative_decoding import _build_model, _generator

        for gdn in (False, True):
            with self.subTest(gdn=gdn):
                target, vocab = _build_model(123, gated_delta=gdn, qwen_mtp=True, hidden=128)
                target = target.cuda().bfloat16()
                Inference._prepare_inference_projections(
                    SimpleNamespace(model=target), config(dense_quantization="w8a8", fuse_kvq=True, fuse_gate=True)
                )
                self.assertTrue(any(isinstance(m, W8A8Linear) for m in target.decoder.modules()))
                plain = copy.deepcopy(target)
                batch = {
                    "srclen": torch.tensor([4], device="cuda"),
                    "src": torch.tensor([[4, 5, 6, 7]], device="cuda"),
                    "left_pad": True,
                }
                with torch.inference_mode(), patch("eole.decoders.transformer.EOLE_TORCH_COMPILE", False), patch(
                    "eole.predict.generator.EOLE_TORCH_COMPILE", False
                ):
                    expected = _generator(plain, vocab, False).predict_batch(batch, False)
                    gen = _generator(target, vocab, True)
                    actual = gen.predict_batch(batch, False)
                self.assertGreater(gen._speculative_drafted_tokens, 0)
                torch.testing.assert_close(actual["predictions"][0][0], expected["predictions"][0][0], rtol=0, atol=0)
