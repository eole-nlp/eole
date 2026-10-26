"""Keep compiled recurrent FLA launches on the native FP32 scalar path."""

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from eole.modules import gated_delta_net as gdn
from eole.ops import fla as fla_ops


@unittest.skipUnless(gdn.fused_recurrent_gated_delta_rule is not None, "FLA is optional")
class TestRecurrentBoundary(unittest.TestCase):
    def test_compiled_inference_dispatch_and_fallback(self):
        q = torch.randn(1, 4, 2, 8)
        g = torch.randn(1, 4, 2)
        state = torch.randn(1, 2, 8, 8)
        with patch.object(torch.compiler, "is_compiling", return_value=True):
            with patch.object(gdn, "_compiled_fla_recurrent_gated_delta_rule") as opaque:
                with patch.object(gdn, "_fla_fused_recurrent_gated_delta_rule") as native:
                    with torch.no_grad():
                        gdn.fused_recurrent_gated_delta_rule(q, q, q, g, g, state, True, True)
                        opaque.assert_called_once()
                        native.assert_not_called()
                        gdn.fused_recurrent_gated_delta_rule(q, q, q, g, g, state, False, True)
                        native.assert_called_once()
                    native.reset_mock()
                    gdn.fused_recurrent_gated_delta_rule(q, q, q, g, g, state, True, True)
                    native.assert_called_once()  # Preserve the gradient-enabled path.

    def test_compile_graph_keeps_native_launch_opaque(self):
        graphs = []

        def backend(graph, example_inputs):
            graphs.append(graph)
            return graph.forward

        def native(q, k, v, **kwargs):
            return v.clone(), kwargs["initial_state"].float().clone()

        def run(q, g, state):
            return gdn.fused_recurrent_gated_delta_rule(q, q, q, g, g, state, True, True)

        with torch.no_grad(), patch.object(fla_ops, "_fla_fused_recurrent_gated_delta_rule", side_effect=native):
            compiled = torch.compile(run, backend=backend, fullgraph=True)
            q = torch.randn(1, 4, 2, 8)
            state = torch.randn(1, 2, 8, 8)
            result = compiled(q, torch.randn(1, 4, 2), state)
            torch.testing.assert_close(result[0], q)
            torch.testing.assert_close(result[1], state)
        self.assertEqual(len(graphs), 1)
        targets = [node.target for node in graphs[0].graph.nodes if node.op == "call_function"]
        self.assertIn(torch.ops.eole._fla_recurrent_gated_delta_rule.default, targets)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_compiled_matches_native_and_replays_changed_inputs(self):
        torch.manual_seed(123)
        for dtype in (torch.float16, torch.bfloat16):
            for length in (1, 4):
                with self.subTest(dtype=dtype, length=length), torch.inference_mode():
                    q, k, v = [torch.randn(1, length, 4, 256, device="cuda", dtype=dtype)[..., ::2] for _ in range(3)]
                    g = -torch.rand(1, length, 4, device="cuda")
                    beta = torch.rand_like(g).to(dtype)
                    state = torch.randn(1, 4, 128, 128, device="cuda", dtype=dtype)

                    def run(q, k, v, g, beta, state):
                        return gdn.fused_recurrent_gated_delta_rule(q, k, v, g, beta, state, True, True)

                    compiled = torch.compile(run, fullgraph=True)
                    args = (q, k, v, g, beta, state)
                    expected = run(*args)
                    actual = compiled(*args)
                    for result, reference in zip(actual, expected):
                        torch.testing.assert_close(result, reference, rtol=0, atol=0)
                    compiled(*args)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = compiled(*args)
                    q.mul_(0.5)
                    state.mul_(0.5)
                    expected = run(*args)
                    graph.replay()
                    for result, reference in zip(captured, expected):
                        torch.testing.assert_close(result, reference, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    @unittest.skipUnless(hasattr(torch.compiler, "load_compiled_function"), "AOT save/load unavailable")
    def test_saved_verifier_rebinds_recurrent_state(self):
        from eole.utils.verifier_cache import CachedVerifier

        class Verifier(torch.nn.Module):
            def __init__(self, state):
                super().__init__()
                self.register_buffer("state", state)
                self._verifier_cache_config = {"fp32_recurrent_launch": True}

            def _forward_verify_graph(self, emb, *, tgt_pad_mask):
                beta = torch.ones_like(emb[..., 0])
                g = torch.zeros_like(beta, dtype=torch.float32)
                output, _ = gdn.fused_recurrent_gated_delta_rule(emb, emb, emb, g, beta, self.state, True, True)
                return output.masked_fill(tgt_pad_mask, 0)

        with tempfile.TemporaryDirectory() as directory, torch.inference_mode():
            x = torch.randn(1, 4, 4, 128, device="cuda", dtype=torch.bfloat16)
            mask = torch.zeros(1, 4, 4, 1, device="cuda", dtype=torch.bool)
            state = torch.zeros(1, 4, 128, 128, device="cuda", dtype=torch.bfloat16)
            first = CachedVerifier(Verifier(state), directory)
            first(x, tgt_pad_mask=mask)
            self.assertEqual(len(list(Path(directory).glob("*/verifier.aot"))), 1)
            other = Verifier(torch.randn_like(state))
            expected = other._forward_verify_graph(x, tgt_pad_mask=mask)
            with patch("torch.compiler.load_compiled_function", wraps=torch.compiler.load_compiled_function) as load:
                second = CachedVerifier(other, directory)
                torch.testing.assert_close(second(x, tgt_pad_mask=mask), expected, rtol=0, atol=0)
                load.assert_called_once()


if __name__ == "__main__":
    unittest.main()
