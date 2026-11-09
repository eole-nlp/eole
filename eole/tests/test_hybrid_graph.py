"""Hybrid GPU stages retain state and fall back outside single-token decode."""

import unittest
from unittest.mock import Mock, patch

import torch

from eole.modules.hybrid_graph import GPUStage, HybridCacheStorage, HybridLayerGraph


class TestHybridStorage(unittest.TestCase):
    def test_reuse_reset_and_invalidate_before_replacement(self):
        graph = Mock()
        storage = HybridCacheStorage([graph])
        signature = (1, torch.float32, torch.device("cpu"))
        storage.begin(signature)
        tensor = storage.zeros("state", (1, 4), dtype=torch.float32, device=torch.device("cpu"))
        tensor.fill_(3)
        graph.close.reset_mock()
        storage.begin(signature)
        reused = storage.zeros("state", (1, 4), dtype=torch.float32, device=torch.device("cpu"))
        self.assertIs(reused, tensor)
        self.assertEqual(reused.count_nonzero(), 0)
        graph.close.assert_not_called()
        with self.assertRaises(ValueError):
            storage.zeros("state", (2, 4), dtype=torch.float32, device=torch.device("cpu"))
        storage.begin((2, torch.float32, torch.device("cpu")))
        graph.close.assert_called_once_with()
        replacement = storage.zeros("state", (2, 4), dtype=torch.float32, device=torch.device("cpu"))
        self.assertIsNot(replacement, tensor)
        storage.close()
        storage.close()
        self.assertEqual(storage.tensors, {})


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestHybridGPUStage(unittest.TestCase):
    def test_replay_changes_inputs_and_invalidates_replaced_state(self):
        state = [torch.zeros(4, device="cuda")]

        def function(x, *, nested):
            state[0].add_(x)
            return state[0] + nested[0]

        stage = GPUStage(function, lambda: state)
        with torch.inference_mode():
            for iteration in range(1, 6):
                x = torch.full((4,), float(iteration), device="cuda")
                output = stage(x, nested=(x * 2,)).clone()
                expected = iteration * (iteration + 1) / 2 + iteration * 2
                torch.testing.assert_close(output, torch.full_like(x, expected))
            self.assertEqual(stage.captures, 1)
            state[0] = torch.zeros_like(state[0])
            output = stage(torch.ones(4, device="cuda"), nested=(torch.zeros(4, device="cuda"),))
            torch.testing.assert_close(output, torch.ones_like(output))
            self.assertIsNone(stage.graph)
            stage.reset()

    def test_failed_capture_warns_once_and_falls_back(self):
        stage = GPUStage(lambda x: x + 1)
        x = torch.ones(4, device="cuda")
        with torch.inference_mode(), patch("torch.cuda.graph", side_effect=RuntimeError("unsupported capture")) as cap:
            for _ in range(5):
                torch.testing.assert_close(stage(x), x + 1)
            self.assertEqual(cap.call_count, 1)
            self.assertIsNone(stage.graph)

    def test_request_guards(self):
        layer = Mock()
        layer.training = False
        layer.layer_type = "full_attention"
        layer.self_attn.kcache = torch.zeros(4, device="cuda")
        layer.self_attn.vcache = torch.zeros(4, device="cuda")
        layer.self_attn.cache_leftpad = torch.zeros(1, device="cuda")
        graph = HybridLayerGraph(layer)
        x = torch.ones(1, 1, 32, device="cuda")
        self.assertFalse(graph.allowed(x, {}))
        with torch.inference_mode():
            self.assertTrue(graph.allowed(x, {}))
            for kwargs in ({"return_attn": True}, {"with_align": True}, {"cache_slice": 16}):
                self.assertFalse(graph.allowed(x, kwargs))
            self.assertFalse(graph.allowed(x.expand(2, 1, 32), {}))
            self.assertFalse(graph.allowed(x.expand(1, 2, 32), {}))
            layer.training = True
            self.assertFalse(graph.allowed(x, {}))
            layer.training = False
            layer.self_attn.kcache = None
            self.assertFalse(graph.allowed(x, {}))
