"""Small CPU regressions for the experimental external drafter path."""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from eole.modules.dflash2 import (
    Attention,
    DFlash2Adapter,
    Norm,
    Selector,
    apply_rotary,
    convolve,
    normalize_rotary,
    prepare_attention,
    rotate,
    rotary_tables,
)
from eole.tests.test_speculative_decoding import _build_model, _generator


def config():
    return SimpleNamespace(
        hidden_size=16,
        head_dim=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_bias=False,
        rms_norm_eps=1e-6,
        sliding_window=8,
        rope_parameters={"rope_theta": 10000000.0},
        vocab_size=12,
        dflash_config={"selector_rank": 4, "selector_top_k": 3},
    )


class TestDFlash2(unittest.TestCase):
    def test_split_half_rope(self):
        x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
        # At position zero RoPE must be the identity.
        torch.testing.assert_close(rotate(x, torch.tensor([0]), 10000.0), x)
        angle = torch.tensor([1.0, 0.01, 1.0, 0.01])
        expected = x * angle.cos() + torch.tensor([[[[-3.0, -4.0, 1.0, 2.0]]]]) * angle.sin()
        torch.testing.assert_close(rotate(x, torch.tensor([1]), 10000.0), expected)

    @unittest.skipUnless(torch.cuda.is_available(), "Fused Q/K normalization requires CUDA")
    def test_fused_normalization_rope_preserves_fp32_math_and_rounding(self):
        for dtype in (torch.bfloat16, torch.float16):
            for length in (1, 8, 131, 2048):
                with self.subTest(dtype=dtype, length=length):
                    torch.manual_seed(length)
                    x = torch.randn(1, length, 8, 128, device="cuda", dtype=dtype).transpose(1, 2)
                    norm = Norm(128, 1e-6).to(device="cuda", dtype=dtype)
                    norm.weight.data.normal_(1, 0.15)
                    tables = rotary_tables(torch.arange(123, 123 + length, device="cuda"), 128, 1e7, dtype)
                    with torch.no_grad():
                        actual = normalize_rotary(x, norm, tables)
                        expected = apply_rotary(norm(x), tables)
                    # FP32 reduction association may move a value across a
                    # BF16/FP16 rounding boundary; no reduction in BF16/FP16.
                    torch.testing.assert_close(
                        actual, expected, atol=2 * torch.finfo(dtype).eps, rtol=torch.finfo(dtype).eps
                    )

    def test_normalization_rope_gradient_path_stays_eager(self):
        x = torch.randn(1, 2, 3, 128, requires_grad=True)
        norm = Norm(128, 1e-6)
        tables = rotary_tables(torch.arange(3), 128, 1e7, torch.float32)
        actual = normalize_rotary(x, norm, tables)
        torch.testing.assert_close(actual, apply_rotary(norm(x), tables), rtol=0, atol=0)
        actual.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(norm.weight.grad)

    def test_shared_mask_uses_retained_absolute_positions(self):
        cfg = config()
        previous = torch.arange(6)
        context = torch.arange(6, 15)
        positions = torch.arange(15, 19)
        _, _, retained, mask = prepare_attention(cfg, context, positions, previous, torch.float32)
        torch.testing.assert_close(retained, torch.arange(8, 15))
        expected = torch.tensor([[abs(q - k) < 8 for k in range(8, 19)] for q in range(15, 19)])[None, None]
        torch.testing.assert_close(mask, expected)

    def test_shared_attention_matches_independent_cache_updates(self):
        torch.manual_seed(19)
        independent = Attention(config()).eval()
        shared = copy.deepcopy(independent)
        with torch.no_grad():
            for start, length in [(5, 5), (8, 3), (10, 2), (14, 4)]:
                context = torch.randn(1, length, 16)
                hidden = torch.randn(1, 4, 16)
                cp, qp = torch.arange(start - length, start), torch.arange(start, start + 4)
                previous = shared.cache[2] if shared.cache is not None else None
                state = prepare_attention(shared.cfg, cp, qp, previous, hidden.dtype)
                actual = shared(hidden, context, cp, qp, state)
                expected = independent(hidden, context, cp, qp)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for a, b in zip(shared.cache, independent.cache):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_drafter_builds_rotary_tables_once_per_block(self):
        cfg = config()
        cfg.num_hidden_layers, cfg.intermediate_size = 3, 32
        cfg.dflash_config.update(
            target_layer_ids=[0, 1], block_size=8, conv_group_size=4, conv_kernel_size=2, mask_token_id=11
        )
        model = DFlash2Adapter(cfg).eval()
        for name, p in model.named_parameters():
            if "base_kernel" in name:
                torch.nn.init.normal_(p, std=0.05)
        embedding = torch.nn.Embedding(12, 16)
        target = SimpleNamespace(
            tgt_emb=lambda ids, step: embedding(ids), generator=torch.nn.Linear(16, 12, bias=False)
        )
        with torch.no_grad(), patch("eole.modules.dflash2.rotary_tables", wraps=rotary_tables) as tables:
            model.set_context(torch.randn(1, 5, 32), 0)
            self.assertEqual(len(model.draft(target, torch.tensor([[4]]), 5, 7)), 7)
            self.assertEqual(tables.call_count, 1)
            model.set_context(torch.randn(1, 3, 32), 5)
            model.draft(target, torch.tensor([[4]]), 8, 7)
            self.assertEqual(tables.call_count, 2)

    def _tiny_drafter(self):
        cfg = config()
        cfg.num_hidden_layers, cfg.intermediate_size = 2, 32
        cfg.dflash_config.update(
            target_layer_ids=[0, 1], block_size=8, conv_group_size=4, conv_kernel_size=2, mask_token_id=11
        )
        model = DFlash2Adapter(cfg).eval()
        for name, parameter in model.named_parameters():
            if "base_kernel" in name:
                torch.nn.init.normal_(parameter, std=0.05)
        return model

    def test_mlp_graph_replays_full_block_and_leaves_tail_eager(self):
        layer = self._tiny_drafter().layers[0]
        reference = copy.deepcopy(layer)
        # Isolate branch dispatch from attention's persistent cache.
        layer.self_attn.forward = lambda hidden, *args: torch.zeros_like(hidden)
        reference.self_attn.forward = layer.self_attn.forward
        layer._mlp_graph_mask = torch.zeros(1, 1, 8, dtype=torch.bool)
        layer._mlp_graph = Mock(side_effect=layer._forward_mlp_branch)
        with torch.no_grad():
            for length in (8, 3, 8):
                hidden = torch.randn(1, length, 16)
                expected = reference(hidden, None, None, None)
                actual = layer(hidden, None, None, None)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(layer._mlp_graph.call_count, 2)
        with torch.enable_grad():
            layer(torch.randn(1, 8, 16), None, None, None)
        layer.train()
        with torch.no_grad():
            layer(torch.randn(1, 8, 16), None, None, None)
        self.assertEqual(layer._mlp_graph.call_count, 2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA graph replay requires a GPU")
    def test_mlp_cuda_graph_matches_eager_without_attention_state(self):
        torch.manual_seed(91)
        model = self._tiny_drafter().to(device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            self.assertEqual(model.prepare_mlp_graphs(), 2)
            runners = [layer._mlp_graph for layer in model.layers]
            self.assertEqual(model.prepare_mlp_graphs(), 2)
            self.assertEqual(runners, [layer._mlp_graph for layer in model.layers])
            for _ in range(2):
                hidden = torch.randn(1, 8, 16, device="cuda", dtype=torch.bfloat16)
                for layer in model.layers:
                    actual = layer._mlp_graph(hidden, tgt_pad_mask=layer._mlp_graph_mask).clone()
                    expected = layer._forward_mlp_branch(hidden)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertIsNone(layer.self_attn.cache)
            self.assertIsNone(model.pending)

    @unittest.skipUnless(torch.cuda.is_available(), "Fixed-buffer replay requires CUDA")
    def test_draft_graph_wraparound_partial_append_and_request_reset(self):
        import weakref

        from eole.modules.dflash2_graph import DFlash2GraphAdapter

        torch.manual_seed(8)
        reference = self._tiny_drafter().to(device="cuda", dtype=torch.bfloat16)
        model = DFlash2GraphAdapter(reference.cfg).to(device="cuda", dtype=torch.bfloat16).eval()
        model.load_state_dict(reference.state_dict())
        target = torch.nn.Module()
        target.tgt_emb = torch.nn.Embedding(12, 16, device="cuda", dtype=torch.bfloat16)
        target.generator = torch.nn.Linear(16, 12, bias=False, device="cuda", dtype=torch.bfloat16)
        object.__setattr__(model, "_target_ref", weakref.ref(target))
        with torch.no_grad():
            self.assertTrue(model.prepare_draft_graph())
            graph = model._graph.graph
            pointers = [buffer.data_ptr() for pair in model._fixed for buffer in pair]
            self.assertTrue(model.prepare_draft_graph())
            first_request = []
            for request in range(2):
                torch.manual_seed(123)
                reference.reset()
                model.reset()
                self.assertTrue((model._positions == -1).all())
                with self.assertRaisesRegex(RuntimeError, "not initialized"):
                    model.draft(target, torch.tensor([[4]], device="cuda"), 5, 7)
                for start, length in ((5, 5), (8, 3), (10, 2), (14, 4), (15, 1)):
                    features = torch.randn(1, length, 32, device="cuda", dtype=torch.bfloat16)
                    anchor = torch.tensor([[4]], device="cuda")
                    reference.set_context(features, start - length)
                    model.set_context(features, start - length)
                    # The reference needs the usual target embedding step argument.
                    wrapper = SimpleNamespace(tgt_emb=lambda ids, step: target.tgt_emb(ids), generator=target.generator)
                    reference.draft(wrapper, anchor, start, 7)
                    proposals = model.draft(target, anchor, start, 2 if start == 15 else 7)
                    self.assertEqual(len(proposals), 2 if start == 15 else 7)
                    self.assertIsNone(model.pending)
                    result = torch.cat(proposals, dim=1).clone()
                    if request == 0:
                        first_request.append(result)
                    else:
                        torch.testing.assert_close(result, first_request.pop(0), atol=0, rtol=0)
                    for layer, (key, value) in zip(reference.layers, model._fixed):
                        expected_key, expected_value, positions = layer.self_attn.cache
                        indices = positions.remainder(reference.cfg.sliding_window)
                        torch.testing.assert_close(model._positions[indices], positions)
                        torch.testing.assert_close(key.index_select(2, indices), expected_key, atol=0.03, rtol=0.02)
                        torch.testing.assert_close(value.index_select(2, indices), expected_value, atol=0.03, rtol=0.02)
                self.assertIs(model._graph.graph, graph)
                self.assertEqual(pointers, [buffer.data_ptr() for pair in model._fixed for buffer in pair])

    def test_drafter_quantization_rejects_biases_and_prepared_graphs(self):
        model = self._tiny_drafter()
        model.layers[0]._mlp_graph = Mock()
        with self.assertRaisesRegex(ValueError, "before preparing"):
            model.quantize_projections()
        model.layers[0]._mlp_graph = None
        model.layers[0].self_attn.q_proj.bias = torch.nn.Parameter(torch.zeros(16))
        with self.assertRaisesRegex(ValueError, "bias-free"):
            model.quantize_projections(fuse_kvq=True, fuse_gate=True)
        self.assertTrue(all(isinstance(layer.self_attn.k_proj, torch.nn.Linear) for layer in model.layers))

    def test_packed_drafter_projections_match_separate_projections(self):
        model = self._tiny_drafter()
        attention = copy.deepcopy(model.layers[0].self_attn)
        mlp = copy.deepcopy(model.layers[0].mlp)

        def joined(*modules):
            result = torch.nn.Linear(modules[0].in_features, sum(m.out_features for m in modules), bias=False)
            result.weight = torch.nn.Parameter(torch.cat([m.weight for m in modules]))
            return result

        attention.noise_qkv = joined(attention.q_proj, attention.k_proj, attention.v_proj)
        attention.context_kv = joined(attention.k_proj, attention.v_proj)
        mlp.gate_up = joined(mlp.gate_proj, mlp.up_proj)
        del attention.q_proj, attention.k_proj, attention.v_proj
        del mlp.gate_proj, mlp.up_proj
        hidden = torch.randn(1, 8, 16)
        reference = model.layers[0]
        for actual, expected in zip(attention.project_noise(hidden), reference.self_attn.project_noise(hidden)):
            torch.testing.assert_close(actual, expected)
        for actual, expected in zip(attention.project_context(hidden), reference.self_attn.project_context(hidden)):
            torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(mlp(hidden), reference.mlp(hidden))

    def test_dynamic_convolution_does_not_read_future(self):
        torch.manual_seed(8)
        x = torch.randn(1, 5, 8)
        dynamic = torch.randn(1, 5, 2, 2)
        base = torch.randn(2, 8)
        expected = torch.zeros_like(x)
        for t in range(5):
            for d in range(8):
                for offset in range(min(t + 1, 2)):
                    expected[0, t, d] += (base[offset, d] + dynamic[0, t, offset, d // 4]) * x[0, t - offset, d]
        torch.testing.assert_close(convolve(x, dynamic, base, 4), expected)

    def test_context_cache_equals_fresh_context_and_stays_bounded(self):
        torch.manual_seed(9)
        cached = Attention(config()).eval()
        fresh = copy.deepcopy(cached)
        context = torch.randn(1, 15, 16)
        with torch.no_grad():
            cached(
                torch.randn(1, 4, 16),
                context[:, :6],
                torch.arange(6),
                torch.arange(6, 10),
            )
            noise = torch.randn(1, 4, 16)
            actual = cached(noise, context[:, 6:], torch.arange(6, 15), torch.arange(15, 19))
            expected = fresh(noise, context, torch.arange(15), torch.arange(15, 19))
        torch.testing.assert_close(actual, expected)
        self.assertLessEqual(cached.cache[0].size(2), 8)
        self.assertLess(int(cached.cache[2][-1]), 15)  # no noise tokens persisted

    def test_selector_transition_can_override_unary_argmax(self):
        selector = Selector(config())
        with torch.no_grad():
            for p in selector.parameters():
                p.zero_()
            selector.hidden_projection.weight[0, 0] = 1
            selector.predecessor_codebook.weight[2, 0] = 10
            selector.successor_codebook.weight[4, 0] = 1
        hidden = torch.zeros(1, 1, 16)
        hidden[..., 0] = 1
        logits = torch.full((1, 1, 12), -100.0)
        logits[..., 3], logits[..., 4] = 2.0, 1.0
        self.assertEqual(selector(hidden, logits, torch.tensor([[2]]))[0].item(), 4)

    def test_chunked_prefill_preserves_bounded_feature_taps(self):
        target, _ = _build_model(126, gated_delta=False)
        target.decoder._dflash2_layer_ids = (0, 1)
        target.decoder._dflash2_window = 3
        chunked = copy.deepcopy(target)
        chunked.decoder.prefill_chunk_size = 2
        prompt = torch.tensor([[4, 5, 6, 7, 8]])
        mask = torch.zeros(1, 1, 5, dtype=torch.bool)
        with torch.no_grad():
            emb = target.tgt_emb(prompt)
            target.decoder._init_cache(emb, mask)
            full, full_attn = target.decoder(emb, step=0, tgt_pad_mask=mask)
            chunked.decoder._init_cache(emb, mask)
            parts, part_attn = chunked.decoder(emb, step=0, tgt_pad_mask=mask)
        torch.testing.assert_close(parts, full)
        self.assertEqual(part_attn["dflash2_features"].shape, (1, 3, 32))
        torch.testing.assert_close(part_attn["dflash2_features"], full_attn["dflash2_features"])

    def test_external_drafter_uses_feature_taps_and_target_verification(self):
        self._check_external_drafter(False)

    def test_external_drafter_with_gated_delta_target(self):
        self._check_external_drafter(True)

    def _check_external_drafter(self, gated_delta):
        target, vocab = _build_model(123, gated_delta=gated_delta)
        plain_target = copy.deepcopy(target)
        target.decoder._dflash2_layer_ids = (0, 1)
        gen = _generator(target, vocab, True)
        contexts = []

        class WrongDrafter:
            block_size = 5

            def reset(self):
                pass

            def set_context(self, features, start):
                contexts.append((features.clone(), start))

            def draft(self, model, anchor, start, count):
                return [anchor.new_full((1, 1), 4) for _ in range(count)]

        gen.dflash2 = WrongDrafter()

        def native_mtp_must_not_run(*args, **kwargs):
            raise AssertionError("Native MTP called for external drafter")

        target.draft_mtp_tokens = native_mtp_must_not_run
        target.set_mtp_context = native_mtp_must_not_run
        target.commit_mtp_draft = native_mtp_must_not_run
        prompt = torch.tensor([[4, 5, 6, 7]])
        batch = {"srclen": torch.tensor([4]), "src": prompt, "left_pad": True}
        with torch.no_grad():
            plain = _generator(plain_target, vocab, False).predict_batch(batch, False)
            actual = gen.predict_batch(batch, False)
        torch.testing.assert_close(actual["predictions"][0][0], plain["predictions"][0][0])
        self.assertGreater(gen._speculative_drafted_tokens, 0)
        self.assertEqual(contexts[0][0].shape, (1, 4, 32))
        self.assertEqual(contexts[0][1], 0)
        self.assertIsNone(gen._dflash2_features)
