import unittest
from unittest.mock import patch

import torch

from eole.config.models import TransformerDecoderConfig
from eole.modules.gated_delta_net import GatedDeltaNet, _torch_speculative_conv1d


class TestGatedDeltaNetSpeculation(unittest.TestCase):
    def _new_layer(self):
        config = TransformerDecoderConfig(
            decoder_type="transformer",
            layers=2,
            hidden_size=16,
            heads=2,
            transformer_ff=32,
            linear_conv_kernel_dim=3,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
        )
        layer = GatedDeltaNet(config, layer_idx=0)
        for module in (layer.in_proj_qkv, layer.in_proj_z, layer.in_proj_b, layer.in_proj_a, layer.out_proj):
            module.reset_parameters()
        return layer

    def _init_state(self, layer, batch=1):
        layer.conv_state = torch.randn(batch, layer.conv_dim, layer.conv_kernel_size)
        layer.recurrent_state = torch.randn(
            batch,
            layer.num_v_heads,
            layer.head_k_dim,
            layer.head_v_dim,
        )

    def test_commit_full_chunk_matches_sequential_decode(self):
        torch.manual_seed(41)
        sequential = self._new_layer()
        speculative = self._new_layer()
        speculative.load_state_dict(sequential.state_dict())
        self._init_state(sequential)
        speculative.conv_state = sequential.conv_state.clone()
        speculative.recurrent_state = sequential.recurrent_state.clone()
        inputs = torch.randn(1, 3, sequential.hidden_size)

        expected = torch.cat([sequential(inputs[:, i : i + 1]) for i in range(3)], dim=1)
        speculative.begin_speculation(inputs.size(1))
        actual = speculative(inputs)
        speculative.end_speculation()
        speculative.commit_speculation(3)

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(speculative.conv_state, sequential.conv_state, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(speculative.recurrent_state, sequential.recurrent_state, atol=1e-5, rtol=1e-5)

    def test_partial_commit_matches_sequential_prefix(self):
        torch.manual_seed(43)
        sequential = self._new_layer()
        speculative = self._new_layer()
        speculative.load_state_dict(sequential.state_dict())
        self._init_state(sequential)
        speculative.conv_state = sequential.conv_state.clone()
        speculative.recurrent_state = sequential.recurrent_state.clone()
        inputs = torch.randn(1, 3, sequential.hidden_size)

        expected = torch.cat([sequential(inputs[:, i : i + 1]) for i in range(2)], dim=1)
        speculative.begin_speculation(inputs.size(1))
        actual = speculative(inputs)
        speculative.end_speculation()
        speculative.commit_speculation(2)

        torch.testing.assert_close(actual[:, :2], expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(speculative.conv_state, sequential.conv_state, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(speculative.recurrent_state, sequential.recurrent_state, atol=1e-5, rtol=1e-5)

    def test_grouped_partial_commit_matches_independent_commits(self):
        torch.manual_seed(47)
        sequential_layers = [self._new_layer(), self._new_layer()]
        speculative_layers = [self._new_layer(), self._new_layer()]
        inputs = [torch.randn(1, 3, layer.hidden_size) for layer in sequential_layers]

        for sequential, speculative in zip(sequential_layers, speculative_layers):
            speculative.load_state_dict(sequential.state_dict())
            self._init_state(sequential)
            speculative.conv_state = sequential.conv_state.clone()
            speculative.recurrent_state = sequential.recurrent_state.clone()

        for layer, layer_inputs in zip(sequential_layers, inputs):
            for index in range(2):
                layer(layer_inputs[:, index : index + 1])

        for layer, layer_inputs in zip(speculative_layers, inputs):
            layer.begin_speculation(layer_inputs.size(1))
            layer(layer_inputs)
            layer.end_speculation()

        GatedDeltaNet.commit_speculation_group(speculative_layers, 2)

        for sequential, speculative in zip(sequential_layers, speculative_layers):
            torch.testing.assert_close(speculative.conv_state, sequential.conv_state, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(speculative.recurrent_state, sequential.recurrent_state, atol=1e-5, rtol=1e-5)

    def test_speculative_buffers_follow_state_dtype(self):
        layer = self._new_layer()
        self._init_state(layer)
        layer.begin_speculation(3)
        layer.end_speculation()
        layer.conv_state = layer.conv_state.bfloat16()
        layer.recurrent_state = layer.recurrent_state.bfloat16()
        layer.begin_speculation(3)
        self.assertEqual(layer._spec_buffers["query"].dtype, torch.bfloat16)
        self.assertEqual(layer._spec_buffers["conv_extended"].dtype, torch.bfloat16)
        self.assertEqual(layer._spec_buffers["final_recurrent_state"].dtype, torch.bfloat16)
        layer.end_speculation()

    def test_short_convolution_matches_grouped_convolution(self):
        torch.manual_seed(61)
        for dtype in (torch.float32, torch.bfloat16):
            for kernel in (2, 3, 4):
                for tokens in (2, 3, 4):
                    inputs = torch.randn(2, 16, kernel + tokens, dtype=dtype)
                    weight = torch.randn(16, 1, kernel, dtype=dtype)
                    for bias in (None, torch.randn(16, dtype=dtype)):
                        expected = torch.nn.functional.silu(
                            torch.nn.functional.conv1d(inputs, weight, bias, groups=16)[:, :, 1:]
                        )
                        actual = _torch_speculative_conv1d(inputs, weight, bias)
                        tolerance = 1e-6 if dtype == torch.float32 else 0.008
                        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)

    def test_grouped_commit_preserves_bfloat16_states(self):
        torch.manual_seed(59)
        for count in (1, 2, 3):
            layers = [self._new_layer().bfloat16(), self._new_layer().bfloat16()]
            references = [self._new_layer().bfloat16(), self._new_layer().bfloat16()]
            for layer, reference in zip(layers, references):
                reference.load_state_dict(layer.state_dict())
                self._init_state(layer)
                layer.conv_state = layer.conv_state.bfloat16()
                layer.recurrent_state = layer.recurrent_state.bfloat16()
                reference.conv_state = layer.conv_state.clone()
                reference.recurrent_state = layer.recurrent_state.clone()
                inputs = torch.randn(1, 3, layer.hidden_size, dtype=torch.bfloat16)
                for candidate in (layer, reference):
                    candidate.begin_speculation(3)
                    candidate(inputs)
                    candidate.end_speculation()
                reference.commit_speculation(count)
            GatedDeltaNet.commit_speculation_group(layers, count)
            for layer, reference in zip(layers, references):
                torch.testing.assert_close(layer.conv_state, reference.conv_state, atol=0, rtol=0)
                torch.testing.assert_close(layer.recurrent_state, reference.recurrent_state, atol=0, rtol=0)

    def test_grouped_full_commit_does_not_pack_states(self):
        torch.manual_seed(51)
        layers = [self._new_layer(), self._new_layer()]
        expected = []
        for layer in layers:
            self._init_state(layer)
            layer.begin_speculation(3)
            layer(torch.randn(1, 3, layer.hidden_size))
            layer.end_speculation()
            expected.append(layer._spec_buffers["final_recurrent_state"].clone())
        with patch("torch.cat", side_effect=AssertionError("full commit must not pack")):
            GatedDeltaNet.commit_speculation_group(layers, 3)
        for layer, state in zip(layers, expected):
            torch.testing.assert_close(layer.recurrent_state, state, atol=0, rtol=0)

    def test_replay_buffers_are_reused_for_different_prefix_lengths(self):
        torch.manual_seed(53)
        layers = [self._new_layer(), self._new_layer()]
        references = [self._new_layer(), self._new_layer()]
        for layer, reference in zip(layers, references):
            reference.load_state_dict(layer.state_dict())
            self._init_state(layer)
            reference.conv_state = layer.conv_state.clone()
            reference.recurrent_state = layer.recurrent_state.clone()
        pointers = None
        for count in (1, 2, 1):
            for layer, reference in zip(layers, references):
                inputs = torch.randn(1, 3, layer.hidden_size)
                for index in range(count):
                    reference(inputs[:, index : index + 1])
                layer.begin_speculation(3)
                layer(inputs)
                layer.end_speculation()
            GatedDeltaNet.commit_speculation_group(layers, count)
            current = {name: buffer.data_ptr() for name, buffer in layers[0]._spec_replay_buffers.items()}
            if pointers is not None:
                self.assertEqual(current, pointers)
            pointers = current
            for layer, reference in zip(layers, references):
                torch.testing.assert_close(layer.conv_state, reference.conv_state)
                torch.testing.assert_close(layer.recurrent_state, reference.recurrent_state, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
