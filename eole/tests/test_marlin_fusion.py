import unittest

import torch

from eole.modules.marlin_linear import MarlinQuantLinear


class TestMarlinFusion(unittest.TestCase):
    def projections(self, desc_act=False):
        layers = [MarlinQuantLinear(4, 128, desc_act, True, 128, n, bias=True) for n in (64, 64, 128)]
        for i, layer in enumerate(layers):
            layer.qweight.data.fill_(i)
            layer.qzeros.data.fill_(i)
            layer.scales.data.fill_(i + 1)
            layer.g_idx.data.zero_()
            layer.bias.fill_(i)
        return layers

    def test_output_columns_and_bias_preserve_kvq_order(self):
        layers = self.projections()
        fused = MarlinQuantLinear.fuse(layers)
        self.assertEqual(fused.out_features, 256)
        for name in ("qweight", "qzeros", "scales"):
            torch.testing.assert_close(getattr(fused, name), torch.cat([getattr(p, name) for p in layers], dim=1))
        torch.testing.assert_close(fused.bias, torch.cat([p.bias for p in layers]))
        self.assertIsNone(fused.workspace)

    def test_attention_fusion_is_idempotent(self):
        from eole.modules.multi_headed_attn import MultiHeadedAttention

        attention = MultiHeadedAttention.__new__(MultiHeadedAttention)
        torch.nn.Module.__init__(attention)
        attention.fused_kvq = False
        layers = self.projections()
        attention.linear_keys, attention.linear_values, attention.linear_query = layers
        attention._fuse_KVQ()
        fused = attention.linear_kvq
        attention._fuse_KVQ()
        self.assertIs(attention.linear_kvq, fused)
        self.assertTrue(attention.fused_kvq)
        self.assertFalse(hasattr(attention, "linear_keys"))
        torch.testing.assert_close(fused.qweight, torch.cat([p.qweight for p in layers], dim=1))

    def test_rejects_repacked_weights(self):
        layers = self.projections()
        layers[0].workspace = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "before post_init"):
            MarlinQuantLinear.fuse(layers)

    def test_rejects_different_activation_order(self):
        layers = self.projections(desc_act=True)
        layers[1].g_idx.data.fill_(1)
        with self.assertRaisesRegex(ValueError, "identical activation ordering"):
            MarlinQuantLinear.fuse(layers)

    def test_rejects_incompatible_groups(self):
        layers = self.projections()
        layers[1].group_size = 64
        with self.assertRaisesRegex(ValueError, "compatible projections"):
            MarlinQuantLinear.fuse(layers)


class TestMarlinFusionWiring(unittest.TestCase):
    projections = TestMarlinFusion.projections

    def attention(self, cls):
        attention = cls.__new__(cls)
        torch.nn.Module.__init__(attention)
        attention.fused_kvq = False
        attention.linear_keys, attention.linear_values, attention.linear_query = self.projections()
        attention.dim_per_head = 64
        return attention

    def test_model_fuses_self_attention_only(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from eole.models.model import BaseModel
        from eole.modules.multi_headed_attn import SelfMHA, ContextMHA

        from eole.predict.inference import Inference

        model = torch.nn.Module()
        model.decoder = torch.nn.Module()
        layer = torch.nn.Module()
        layer.self_attn = self.attention(SelfMHA)
        layer.context_attn = self.attention(ContextMHA)
        model.decoder.transformer_layers = torch.nn.ModuleList([layer])
        config = SimpleNamespace(compute_dtype=torch.float16, quant_type="autoround", fuse_kvq=True, fuse_gate=False)
        with patch.object(MarlinQuantLinear, "post_init", autospec=True) as post_init:
            BaseModel._prepare_for_inference(model, config)
            self.assertFalse(layer.self_attn.fused_kvq)
            self.assertEqual(post_init.call_count, 3)  # Cross-attention is initialized immediately.
            Inference._prepare_inference_projections(SimpleNamespace(model=model), config)
            self.assertEqual(post_init.call_count, 4)  # One packed self-attention projection.
        model.self_attn = layer.self_attn
        model.context_attn = layer.context_attn
        self.assertTrue(model.self_attn.fused_kvq)
        self.assertFalse(model.context_attn.fused_kvq)
        # Exercise cache prefill as well as the selection predicate.
        with patch.object(model.context_attn.linear_keys, "forward", return_value=torch.zeros(1, 2, 64)), patch.object(
            model.context_attn.linear_values, "forward", return_value=torch.ones(1, 2, 64)
        ):
            model.context_attn._prefill_cache(torch.zeros(1, 2, 128))
        self.assertEqual(model.context_attn.kcache.shape, (1, 2, 1, 64))
        self.assertTrue(model.context_attn.vcache.eq(1).all())

    def test_missing_fusion_option(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from eole.models.model import BaseModel
        from eole.modules.multi_headed_attn import SelfMHA

        model = torch.nn.Module()
        model.self_attn = self.attention(SelfMHA)
        with patch("eole.modules.autoround_linear.post_init_autoround_linear"):
            BaseModel._prepare_for_inference(
                model, SimpleNamespace(compute_dtype=torch.float16, quant_type="autoround")
            )
        self.assertFalse(model.self_attn.fused_kvq)

    def test_incompatible_projections_preserved_and_retry_skipped(self):
        from eole.modules.multi_headed_attn import SelfMHA
        from unittest.mock import patch

        for mismatch in ("keys", "values", "query", "group", "bits", "desc_act"):
            attention = self.attention(SelfMHA)
            if mismatch in ("keys", "values", "query"):
                setattr(attention, "linear_" + mismatch, torch.nn.Linear(128, 64))
            elif mismatch == "group":
                attention.linear_values.group_size = 64
            elif mismatch == "bits":
                attention.linear_query.bits = 8
            else:
                attention.linear_query.desc_act = True
            projections = [attention.linear_keys, attention.linear_values, attention.linear_query]
            with patch("eole.utils.logging.logger.warning") as warning:
                attention._fuse_KVQ()
                attention._fuse_KVQ()
                warning.assert_called_once()
            self.assertFalse(attention.fused_kvq)
            self.assertEqual(projections, [attention.linear_keys, attention.linear_values, attention.linear_query])
            self.assertFalse(hasattr(attention, "linear_kvq"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for Marlin kernels")
    def test_cuda_projection_and_query_gate_equivalence(self):
        from eole.modules.multi_headed_attn import SelfMHA

        torch.manual_seed(451)
        for gated in (False, True):
            attention = self.attention(SelfMHA)
            if gated:
                attention.linear_query = MarlinQuantLinear(4, 128, False, True, 128, 256, bias=True)
            layers = [attention.linear_keys, attention.linear_values, attention.linear_query]
            for layer in layers:
                layer.qweight.data.random_(0, 2**31 - 1)
                layer.scales.data.uniform_(0.01, 0.1)
                layer.bias.uniform_(-0.1, 0.1)
                layer.g_idx.data.zero_()
                layer.qzeros.data.zero_()
                layer.cuda()
            fused = MarlinQuantLinear.fuse(layers)
            for layer in layers + [fused]:
                layer.post_init()
            inputs = torch.randn(2, 3, 128, device="cuda", dtype=torch.float16)
            expected = torch.cat([layer(inputs) for layer in layers], dim=-1)
            torch.testing.assert_close(fused(inputs), expected, atol=0.01, rtol=0.01)
            attention.q_gating = gated
            attention.heads = 2
            attention.heads_kv = 1
            attention.parallel_gpu = 1
            attention.qk_norm_post_rope = False
            attention.position_encoding_type = None
            attention.maybe_ckpt = lambda module, value: module(value)
            reference = attention._prepare_inputs(inputs, inputs, inputs)
            gate = attention._attn_gate.clone() if gated else None
            attention.linear_kvq = fused
            attention.fused_kvq = True
            actual = attention._prepare_inputs(inputs, inputs, inputs)
            for ref, result in zip(reference, actual):
                torch.testing.assert_close(result, ref, atol=0.01, rtol=0.01)
            if gated:
                torch.testing.assert_close(attention._attn_gate, gate, atol=0.001, rtol=0.01)

    def test_optional_marlin_import_not_needed_for_other_backends(self):
        import builtins
        from unittest.mock import patch
        from eole.modules.autoround_linear import fuse_autoround_projections

        original_import = builtins.__import__

        def without_marlin(name, *args, **kwargs):
            if name == "eole.modules.marlin_linear":
                raise ImportError("Marlin not installed")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=without_marlin):
            self.assertEqual(fuse_autoround_projections([torch.nn.Linear(128, 64)] * 3), (False, None))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for Marlin kernels")
    def test_inference_finalizes_deferred_weights(self):
        from types import SimpleNamespace
        from eole.models.model import BaseModel
        from eole.predict.inference import Inference
        from eole.modules.multi_headed_attn import SelfMHA

        for incompatible in (False, True):
            model = torch.nn.Module()
            model.decoder = torch.nn.Module()
            layer = torch.nn.Module()
            layer.self_attn = self.attention(SelfMHA)
            if incompatible:
                layer.self_attn.linear_values = torch.nn.Linear(128, 64).half().cuda()
            projections = [layer.self_attn.linear_keys, layer.self_attn.linear_values, layer.self_attn.linear_query]
            for projection in projections:
                if isinstance(projection, MarlinQuantLinear):
                    projection.qweight.data.random_(0, 2**31 - 1)
                    projection.scales.data.fill_(0.05)
                    projection.qzeros.data.zero_()
                    projection.g_idx.data.zero_()
                    projection.cuda()
            model.decoder.transformer_layers = torch.nn.ModuleList([layer])
            config = SimpleNamespace(
                compute_dtype=torch.float16, quant_type="autoround", fuse_kvq=True, fuse_gate=False
            )
            BaseModel._prepare_for_inference(model, config)
            self.assertIsNone(projections[0].workspace)
            predictor = SimpleNamespace(model=model)
            Inference._prepare_inference_projections(predictor, config)
            inputs = torch.randn(1, 2, 128, device="cuda", dtype=torch.float16)
            if incompatible:
                self.assertFalse(layer.self_attn.fused_kvq)
                self.assertIsNotNone(projections[0].workspace)
                expected = torch.cat([projection(inputs) for projection in projections], dim=-1)
                actual = torch.cat([projection(inputs) for projection in projections], dim=-1)
            else:
                self.assertTrue(layer.self_attn.fused_kvq)
                self.assertIsNotNone(layer.self_attn.linear_kvq.workspace)
                for projection in projections:
                    projection.post_init()
                expected = torch.cat([projection(inputs) for projection in projections], dim=-1)
                actual = layer.self_attn.linear_kvq(inputs)
            torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)
            Inference._prepare_inference_projections(predictor, config)
            if not incompatible:
                torch.testing.assert_close(layer.self_attn.linear_kvq(inputs), actual, atol=0.01, rtol=0.01)

    def test_hybrid_layers_skip_missing_self_attention(self):
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        from eole.predict.inference import Inference

        model = torch.nn.Module()
        attentions = []
        for side in ("encoder", "decoder"):
            stack = torch.nn.Module()
            stack.transformer_layers = torch.nn.ModuleList()
            for has_attention in (False, True, False):
                layer = torch.nn.Module()
                layer.self_attn = torch.nn.Module() if has_attention else None
                if has_attention:
                    layer.self_attn._fuse_KVQ = Mock()
                    attentions.append(layer.self_attn)
                stack.transformer_layers.append(layer)
            setattr(model, side, stack)
        config = SimpleNamespace(fuse_kvq=True, fuse_gate=False)
        with patch("eole.modules.autoround_linear.finalize_deferred_autoround_linear") as finalize:
            Inference._prepare_inference_projections(SimpleNamespace(model=model), config)
            finalize.assert_called_once_with(model)
        for attention in attentions:
            attention._fuse_KVQ.assert_called_once()

    def test_non_module_scoring_model_skips_quantized_initialization(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from eole.predict.inference import Inference

        predictor = SimpleNamespace(model=SimpleNamespace())
        config = SimpleNamespace(fuse_kvq=False, fuse_gate=False)
        with patch("eole.modules.autoround_linear.finalize_deferred_autoround_linear") as finalize:
            Inference._prepare_inference_projections(predictor, config)
            finalize.assert_not_called()
