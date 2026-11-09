"""Hybrid GPU stages retain state and fall back outside single-token decode."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from eole.modules.hybrid_graph import GraphPool, GPUStage, HybridCacheStorage, HybridLayerGraph


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


class TestGraphPoolOrder(unittest.TestCase):
    def test_waits_for_all_warmups_and_restarts_after_partial_order(self):
        pool = GraphPool(torch.device("cuda"))
        a, b = GPUStage(lambda x: x, pool=pool), GPUStage(lambda x: x, pool=pool)
        with patch("torch.cuda.graph_pool_handle", return_value=(1, 2)) as handle, patch("torch.cuda.synchronize"):
            for _ in range(2):
                for stage in (a, b):
                    self.assertFalse(pool.enter(stage))
                    stage.warm_calls += 1
            self.assertTrue(pool.enter(a))
            self.assertTrue(pool.enter(b))
            handle.assert_called_once()
            # A missing prefix may not silently replay the rest out of order.
            self.assertFalse(pool.enter(b))
            self.assertTrue(pool.disabled)
            self.assertIsNone(pool.handle)
            self.assertFalse(pool.enter(a))
            pool.begin()
            self.assertFalse(pool.disabled)
            self.assertEqual((a.warm_calls, b.warm_calls), (0, 0))
            self.assertFalse(pool.enter(a))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestSharedHybridPool(unittest.TestCase):
    def stages(self):
        pool = GraphPool(torch.device("cuda"))
        state = [torch.zeros(128, device="cuda")]

        def prefix(x):
            state[0].add_(x)
            return state[0] * 2

        a = GPUStage(prefix, lambda: state, pool=pool)
        b = GPUStage(lambda x, y: x + y * 3, pool=pool)
        return pool, state, a, b

    def test_outputs_reuse_and_new_state_epoch(self):
        pool, state, a, b = self.stages()
        x = torch.ones(128, device="cuda")
        with torch.inference_mode():
            for i in range(1, 7):
                torch.testing.assert_close(b(a(x), x), torch.full_like(x, i * 2 + 3))
            self.assertEqual(a.graph.pool(), b.graph.pool())
            self.assertIs(a.stream, b.stream)
            handle = pool.handle
            pool.begin()
            self.assertEqual(a.captures, 1)
            torch.testing.assert_close(b(a(x), x), torch.full_like(x, 17))
            # Unexpected state replacement abandons the entire pool epoch.
            state[0] = torch.zeros_like(x)
            torch.testing.assert_close(b(a(x), x), torch.full_like(x, 5))
            self.assertTrue(pool.disabled)
            self.assertIsNone(a.graph)
            self.assertIsNone(b.graph)
            pool.begin()
            for i in range(2, 5):
                torch.testing.assert_close(b(a(x), x), torch.full_like(x, i * 2 + 3))
            self.assertNotEqual(handle, pool.handle)
            self.assertEqual(a.graph.pool(), b.graph.pool())
            pool.reset()
            pool.reset()

    def test_singleton_prefill_slice_preserves_capture(self):
        pool = GraphPool(torch.device("cuda"))
        a = GPUStage(lambda x: x * 2, pool=pool)
        b = GPUStage(lambda x: x + 3, pool=pool)
        source = torch.ones(1, 65, 128, device="cuda")
        sliced = source[:, -1:]
        contiguous = torch.ones(1, 1, 128, device="cuda")
        self.assertNotEqual(sliced.stride(), contiguous.stride())
        with torch.inference_mode():
            for x in (contiguous,) * 4 + (sliced, contiguous):
                torch.testing.assert_close(b(a(x)), x * 2 + 3)
            self.assertFalse(pool.disabled)
            self.assertEqual((a.captures, b.captures), (1, 1))
            pool.reset()

    def test_failed_suffix_falls_back_once_without_double_state_update(self):
        pool, state, a, b = self.stages()
        x = torch.ones(128, device="cuda")
        graph = torch.cuda.graph
        calls = []

        def fail_suffix(*args, **kwargs):
            calls.append(True)
            if len(calls) == 2:
                raise RuntimeError("unsupported suffix")
            return graph(*args, **kwargs)

        with torch.inference_mode(), patch("torch.cuda.graph", side_effect=fail_suffix):
            for i in range(1, 5):
                torch.testing.assert_close(b(a(x), x), torch.full_like(x, i * 2 + 3))
            self.assertTrue(pool.disabled)
            pool.begin()
            for i in range(5, 9):
                torch.testing.assert_close(b(a(x), x), torch.full_like(x, i * 2 + 3))
            self.assertEqual(len(calls), 3)  # Failed suffix is not retried.
            pool.reset()

    def test_expert_failure_abandons_partial_cycle(self):
        pool = GraphPool(torch.device("cuda"))
        executor = SimpleNamespace(execute=Mock(side_effect=RuntimeError("CPU expert failed")))
        mlp = SimpleNamespace(
            _executor=executor,
            route=lambda x: (x.new_ones((1, 1)), torch.zeros((1, 1), device=x.device, dtype=torch.long)),
            combine=lambda x, routed: x + routed,
        )
        layer = SimpleNamespace(
            layer_type="full_attention",
            mlp=mlp,
            self_attn=SimpleNamespace(
                kcache=torch.zeros(1, device="cuda"),
                vcache=torch.zeros(1, device="cuda"),
                cache_leftpad=torch.zeros(1, device="cuda"),
            ),
            _self_attention=lambda x, **kwargs: (None, x * 2, None),
            post_attention_layernorm=lambda x: x,
        )
        graph = HybridLayerGraph(layer, pool=pool)
        x = torch.ones(1, 1, 128, device="cuda")
        with torch.inference_mode():
            with self.assertRaisesRegex(RuntimeError, "CPU expert failed"):
                graph(x)
            self.assertTrue(pool.disabled)
            pool.begin()
            executor.execute = lambda x, *args, **kwargs: x
            for _ in range(4):
                output, _ = graph(x)
                torch.testing.assert_close(output, x * 9)
            self.assertEqual(graph.before.graph.pool(), graph.after.graph.pool())
            graph.close()
            self.assertTrue(pool.disabled)
