"""State-only GDN replay correctness, eligibility and pointer-cache lifetime."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from eole.modules.gated_delta_net import GatedDeltaNet, fused_recurrent_gated_delta_rule
from eole.modules.gdn_state import commit_speculation_states


class TestStateCommitFallback(unittest.TestCase):
    @torch.no_grad()
    def test_empty_cpu_and_full_chunk_do_not_launch(self):
        self.assertFalse(commit_speculation_states([], 1))
        layer = SimpleNamespace(recurrent_state=torch.zeros(1, 4, 128, 128, dtype=torch.bfloat16))
        for n in (0, 1, 4, 5):
            self.assertFalse(commit_speculation_states([layer], n))
        self.assertEqual(layer.recurrent_state.count_nonzero(), 0)


@unittest.skipUnless(
    torch.cuda.is_available() and fused_recurrent_gated_delta_rule is not None, "CUDA and FLA required"
)
class TestDirectStateCommit(unittest.TestCase):
    def layers(self, heads=4, count=3, kernel=4):
        result = []
        for _ in range(count):
            spec = {
                name: torch.randn(1, 4, heads, 128, device="cuda", dtype=torch.bfloat16)
                for name in ("query", "key", "value")
            }
            spec["g"] = -torch.rand(1, 4, heads, device="cuda")
            spec["beta"] = torch.rand(1, 4, heads, device="cuda").bfloat16()
            spec["conv_extended"] = torch.randn(1, 12, kernel + 4, device="cuda", dtype=torch.bfloat16)
            state = torch.randn(1, heads, 128, 128, device="cuda", dtype=torch.bfloat16)
            spec["final_recurrent_state"] = torch.randn_like(state)
            result.append(
                SimpleNamespace(
                    _spec_buffers=spec,
                    recurrent_state=state,
                    conv_state=torch.randn(1, 12, kernel, device="cuda", dtype=torch.bfloat16),
                    conv_dim=12,
                    conv_kernel_size=kernel,
                    _recurrent_gated_delta_rule=fused_recurrent_gated_delta_rule,
                )
            )
        return result

    def compare(self, layers, n):
        expected = []
        for layer in layers:
            s = layer._spec_buffers
            _, state = fused_recurrent_gated_delta_rule(
                s["query"][:, :n],
                s["key"][:, :n],
                s["value"][:, :n],
                s["g"][:, :n],
                s["beta"][:, :n],
                layer.recurrent_state,
                True,
                True,
            )
            expected.append((state.bfloat16(), s["conv_extended"][:, :, n : n + layer.conv_kernel_size].clone()))
        self.assertTrue(commit_speculation_states(layers, n))
        for layer, (state, conv) in zip(layers, expected):
            torch.testing.assert_close(layer.recurrent_state, state, rtol=0, atol=0)
            torch.testing.assert_close(layer.conv_state, conv, rtol=0, atol=0)

    @torch.inference_mode()
    def test_prefixes_and_repeated_commits_match_fla(self):
        torch.manual_seed(101)
        for heads, kernel in ((4, 3), (4, 4), (24, 4), (48, 4)):
            layers = self.layers(heads, kernel=kernel)
            for n in (1, 2, 3, 1, 3):
                with self.subTest(heads=heads, n=n):
                    self.compare(layers, n)

    @torch.inference_mode()
    def test_group_wiring_skips_packing_and_preserves_full_chunk_path(self):
        layers = self.layers()
        with patch("torch.cat", side_effect=AssertionError("direct commit must not pack")):
            GatedDeltaNet.commit_speculation_group(layers, 2)
        expected = [layer._spec_buffers["final_recurrent_state"].clone() for layer in layers]
        with patch("eole.modules.gated_delta_net.commit_speculation_states") as direct:
            GatedDeltaNet.commit_speculation_group(layers, 4)
            direct.assert_not_called()
        for layer, state in zip(layers, expected):
            torch.testing.assert_close(layer.recurrent_state, state, rtol=0, atol=0)

    @torch.inference_mode()
    def test_cache_reuses_table_and_rebinds_buffers_layer_order_and_stream(self):
        layers = self.layers()
        self.compare(layers, 1)
        cache = layers[0]._spec_commit_cache
        self.compare(layers, 2)
        self.assertIs(layers[0]._spec_commit_cache, cache)
        retired = layers[1].recurrent_state
        retired_value = retired.clone()
        self.assertTrue(any(tensor is retired for tensor in cache[2]))
        layers[1].recurrent_state = torch.randn_like(retired)
        layers[0].conv_state = torch.randn_like(layers[0].conv_state)
        layers[2]._spec_buffers["key"] = torch.randn_like(layers[2]._spec_buffers["key"])
        self.compare(layers, 3)
        torch.testing.assert_close(retired, retired_value, rtol=0, atol=0)
        self.assertIsNot(layers[0]._spec_commit_cache, cache)
        cache = layers[0]._spec_commit_cache
        layers[1:] = reversed(layers[1:])
        self.compare(layers, 1)
        self.assertIsNot(layers[0]._spec_commit_cache, cache)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.compare(layers, 2)
        torch.cuda.current_stream().wait_stream(stream)

    @torch.inference_mode()
    def test_unsupported_layouts_do_not_mutate_state(self):
        for invalid in ("dtype", "stride", "shape", "backend", "duplicate", "overlap", "unaligned", "triton", "arch"):
            layers = self.layers()
            if invalid == "dtype":
                layers[1]._spec_buffers["beta"] = layers[1]._spec_buffers["beta"].float()
            elif invalid == "stride":
                layers[1].recurrent_state = layers[1].recurrent_state.transpose(-1, -2)
            elif invalid == "shape":
                layers[1]._spec_buffers["key"] = layers[1]._spec_buffers["key"][:, :3]
            elif invalid == "backend":
                layers[1]._recurrent_gated_delta_rule = lambda *a, **kw: None
            elif invalid == "duplicate":
                layers[1] = layers[0]
            elif invalid == "overlap":
                state = layers[0].recurrent_state
                count = state.numel()
                shared = torch.randn(count * 2, device=state.device, dtype=state.dtype)
                layers[0].recurrent_state = shared[:count].view_as(state)
                layers[1].recurrent_state = shared[count // 2 : count // 2 + count].view_as(state)
            elif invalid == "unaligned":
                shape = layers[1]._spec_buffers["key"].shape
                count = layers[1]._spec_buffers["key"].numel()
                shared = torch.randn(count + 1, device="cuda", dtype=torch.bfloat16)
                layers[1]._spec_buffers["key"] = shared[1:].view(shape)
            before = [layer.recurrent_state.clone() for layer in layers]
            context = (
                patch("importlib.util.find_spec", return_value=None)
                if invalid == "triton"
                else (
                    patch("torch.cuda.get_device_capability", return_value=(7, 5))
                    if invalid == "arch"
                    else patch("torch.cuda.get_device_capability", wraps=torch.cuda.get_device_capability)
                )
            )
            with self.subTest(invalid=invalid), context:
                self.assertFalse(commit_speculation_states(layers, 1))
            for layer, state in zip(layers, before):
                torch.testing.assert_close(layer.recurrent_state, state, rtol=0, atol=0)

    @torch.inference_mode()
    def test_cold_capture_falls_back_before_mutation(self):
        layers = self.layers()
        before = [layer.recurrent_state.clone() for layer in layers]
        with patch("torch.cuda.is_current_stream_capturing", return_value=True):
            self.assertFalse(commit_speculation_states(layers, 1))
        for layer, state in zip(layers, before):
            torch.testing.assert_close(layer.recurrent_state, state, rtol=0, atol=0)

    @torch.inference_mode()
    def test_kernel_failure_propagates_without_replaying(self):
        layers = self.layers()
        with patch("eole.triton.gdn_state.commit_states", side_effect=RuntimeError("launch failed")):
            with self.assertRaisesRegex(RuntimeError, "launch failed"):
                GatedDeltaNet.commit_speculation_group(layers, 1)


if __name__ == "__main__":
    unittest.main()
