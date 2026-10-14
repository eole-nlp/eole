import unittest
from unittest.mock import patch

import torch
from torch import nn

from eole.config.models import TransformerDecoderConfig
from eole.modules.gated_delta_net import GatedDeltaNet


class TestGDNControlProjection(unittest.TestCase):
    def new_layer(self, hidden_size=16, heads=2):
        config = TransformerDecoderConfig(
            hidden_size=hidden_size,
            heads=40 if hidden_size == 5120 else heads,
            transformer_ff=hidden_size * 2,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_num_key_heads=heads,
            linear_num_value_heads=heads,
        )
        return GatedDeltaNet(config, layer_idx=0).eval()

    def test_split_preserves_outputs_and_token_order(self):
        layer = self.new_layer()
        torch.manual_seed(17)
        projection = nn.Linear(16, 2, bias=True)
        inputs = torch.randn(1, 4, 16)
        with patch.object(projection, "forward", wraps=projection.forward) as calls:
            actual = layer._project_control(projection, inputs, split_rows=True)
        self.assertEqual(calls.call_count, 4)
        self.assertEqual([tuple(c.args[0].shape) for c in calls.call_args_list], [(1, 16)] * 4)
        torch.testing.assert_close(actual, projection(inputs))

    def test_fallback_uses_one_projection(self):
        layer = self.new_layer()
        projection = nn.Linear(16, 2)
        for shape in [(1, 1, 16), (1, 3, 16), (1, 4, 16), (2, 4, 16), (1, 17, 16)]:
            with self.subTest(shape=shape):
                inputs = torch.randn(shape)
                with patch.object(projection, "forward", wraps=projection.forward) as calls:
                    actual = layer._project_control(projection, inputs)
                self.assertEqual(calls.call_count, 1)
                torch.testing.assert_close(actual, projection(inputs))

    def test_cpu_speculation_does_not_query_cuda_or_enable_dispatch(self):
        layer = self.new_layer()
        layer.conv_state = torch.zeros(1, layer.conv_dim, layer.conv_kernel_size)
        layer.recurrent_state = torch.zeros(1, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim)
        with patch("torch.cuda.get_device_capability") as capability:
            layer.begin_speculation(4)
        capability.assert_not_called()
        self.assertFalse(layer._control_gemv_enabled)
        layer._control_gemv_enabled = True
        self.assertFalse(layer._should_split_control_rows(torch.randn(1, 4, 16)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_dispatch_guards(self):
        layer = self.new_layer().cuda().bfloat16()
        layer._speculating = True
        layer._control_gemv_enabled = True
        inputs = torch.randn(1, 4, 16, device="cuda", dtype=torch.bfloat16)
        self.assertTrue(layer._should_split_control_rows(inputs))
        for shape in [(1, 1, 16), (1, 3, 16), (2, 4, 16), (1, 17, 16)]:
            self.assertFalse(layer._should_split_control_rows(torch.randn(shape, device="cuda", dtype=inputs.dtype)))
        for dtype in [torch.float16, torch.float32]:
            self.assertFalse(layer._should_split_control_rows(inputs.to(dtype)))
        layer.train()
        self.assertFalse(layer._should_split_control_rows(inputs))
        layer.eval()
        layer._speculating = False
        self.assertFalse(layer._should_split_control_rows(inputs))
        layer._speculating = True
        layer.in_proj_a = nn.Identity()
        self.assertFalse(layer._should_split_control_rows(inputs))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_real_shape_compiles_and_replays_in_cuda_graph(self):
        layer = self.new_layer(hidden_size=5120, heads=48).cuda().bfloat16()
        layer.conv_state = torch.zeros(1, layer.conv_dim, layer.conv_kernel_size, device="cuda")
        layer.recurrent_state = torch.zeros(1, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim, device="cuda")
        layer.begin_speculation(4)
        self.assertEqual(layer._control_gemv_enabled, torch.cuda.get_device_capability() == (12, 0))
        with patch("torch.cuda.get_device_capability", return_value=(8, 9)):
            layer.begin_speculation(4)
        self.assertFalse(layer._control_gemv_enabled)
        layer.begin_speculation(3)
        self.assertFalse(layer._control_gemv_enabled)
        original_a = layer.in_proj_a
        layer.in_proj_a = nn.Linear(5120, 24, bias=False).cuda().bfloat16()
        layer.begin_speculation(4)
        self.assertFalse(layer._control_gemv_enabled)
        layer.in_proj_a = original_a
        layer.begin_speculation(4)
        # Test the arithmetic/capture even on CUDA devices outside the dispatch policy.
        layer.in_proj_a.reset_parameters()
        inputs = torch.randn(1, 4, 5120, device="cuda", dtype=torch.bfloat16)
        compiled = torch.compile(lambda x: layer._project_control(layer.in_proj_a, x, True), fullgraph=True)
        with torch.inference_mode():
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    compiled(inputs)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = compiled(inputs)
            for _ in range(2):
                inputs.copy_(torch.randn_like(inputs))
                graph.replay()
                torch.testing.assert_close(output, layer.in_proj_a(inputs), atol=0.02, rtol=0.02)
        torch._dynamo.reset()


if __name__ == "__main__":
    unittest.main()
