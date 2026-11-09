"""Routing/execution separation and experimental host expert placement."""

import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from torch import nn
from eole.modules.moe import MoE
from eole.modules.expert_execution import vectorized_moe
from eole.modules.expert_placement import ExpertWeightPlacement, validate_expert_placement
from eole.modules.hybrid_experts import (
    CPUInt8Linear,
    HybridExpertExecutor,
    HybridExpertRuntime,
    prepare_hybrid_experts,
)


def tiny_moe(experts=4):
    config = SimpleNamespace(
        hidden_size=32,
        transformer_ff=32,
        moe_transformer_ff=32,
        num_experts=experts,
        num_experts_per_tok=2,
        num_shared_experts=0,
        shared_expert_gate=False,
        moe_softmax_after=False,
        moe_renormalize=True,
        mlp_activation_fn="gated-silu",
        add_ffnbias=False,
    )
    running = SimpleNamespace(parallel_gpu=1, dropout=[0.0], use_ckpting=[])
    model = MoE(config, running).eval()
    for p in model.parameters():
        nn.init.normal_(p, std=0.1)
    return model


class TestExpertExecution(unittest.TestCase):
    def test_resident_state_dict_and_training_gradients(self):
        model = tiny_moe()
        keys = set(model.state_dict())
        self.assertFalse(any("executor" in key for key in keys))
        model.train()
        x = torch.randn(1, 3, 32, requires_grad=True)
        # CPU reference isolates executor gradients from the existing CUDA-only gate fusion.
        with patch.object(model, "_maybe_fuse_gates"):
            model(x).sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(model.gate.weight.grad)

    def test_resident_routing_matches_reference(self):
        model = tiny_moe()
        x = torch.randn(1, 5, 32)
        with torch.no_grad(), patch("eole.modules.expert_execution._triton_moe", False):
            scores = model.gate(x.view(-1, 32)).softmax(-1)
            weights, ids = scores.topk(2, sorted=False)
            weights = weights / weights.sum(-1, keepdim=True)
            expected = vectorized_moe(x.view(-1, 32).repeat_interleave(2, 0), weights, ids, 2, model.experts)
            torch.testing.assert_close(model(x).view(-1, 32), expected)

    def test_placement_is_independent_of_checkpoint_keys(self):
        model = nn.Module()
        model.mlp = tiny_moe()
        config = SimpleNamespace(moe_expert_backend="hybrid_int8", moe_gpu_experts=1)
        placement = ExpertWeightPlacement(model, config, torch.device("cuda"))
        self.assertEqual(placement.device_for("mlp.experts.0.gate_up_proj"), torch.device("cuda"))
        self.assertEqual(placement.device_for("mlp.experts.1.gate_up_proj"), torch.device("cpu"))
        self.assertEqual(placement.device_for("mlp.shared_experts.gate_up_proj"), torch.device("cuda"))
        self.assertEqual(placement.device_for("mlp.gate"), torch.device("cuda"))
        self.assertIsNone(placement.dtype_for("mlp.experts.1.gate_up_proj", torch.bfloat16))
        self.assertEqual(placement.dtype_for("mlp.gate", torch.bfloat16), torch.bfloat16)
        resident = ExpertWeightPlacement(model, SimpleNamespace(), torch.device("cuda"))
        self.assertEqual(resident.device_for("mlp.experts.1.gate_up_proj"), torch.device("cuda"))

    def test_unsupported_hybrid_modes_fail_before_loading(self):
        with patch("eole.EOLE_TORCH_COMPILE", False):
            for options, message in (
                ({"parallel_gpu": 2}, "one GPU"),
                ({"quant_type": "bnb_NF4"}, "unquantized source"),
                ({"self_speculative_decoding": True}, "ordinary decoding"),
            ):
                with self.subTest(options=options):
                    config = SimpleNamespace(moe_expert_backend="hybrid_int8", **options)
                    with self.assertRaisesRegex(ValueError, message):
                        validate_expert_placement(config)
            config = SimpleNamespace(moe_expert_backend="hybrid_int8", moe_gpu_experts=4)
            with self.assertRaisesRegex(ValueError, "at least one CPU expert"):
                ExpertWeightPlacement(tiny_moe(), config, torch.device("cuda"))
        with patch("eole.EOLE_TORCH_COMPILE", True):
            with self.assertRaisesRegex(ValueError, "EOLE_TORCH_COMPILE=0"):
                validate_expert_placement(SimpleNamespace(moe_expert_backend="hybrid_int8"))

    def test_preparation_is_idempotent_and_engine_termination_closes_runtime(self):
        from eole.inference_engine import InferenceEnginePY

        root = nn.Module()
        root.mlp = tiny_moe()
        config = SimpleNamespace(moe_expert_backend="hybrid_int8", moe_gpu_experts=0, world_size=1)
        with patch("eole.EOLE_TORCH_COMPILE", False):
            prepare_hybrid_experts(root, config)
            executor = root.mlp._executor
            prepare_hybrid_experts(root, config)
            self.assertIs(root.mlp._executor, executor)
        engine = InferenceEnginePY.__new__(InferenceEnginePY)
        engine.config = config
        engine.predictor = SimpleNamespace(model=root)
        engine.terminate()
        with self.assertRaises(RuntimeError):
            root._expert_runtime.pool.submit(lambda: None)

    def test_graph_opt_out_and_unsupported_decoders_do_not_create_cuda_streams(self):
        for enabled, dynamic, backend, context in (
            (False, False, "flash", False),
            (True, True, "flash", False),
            (True, False, "torch", False),
            (True, False, "flash", True),
        ):
            with self.subTest(enabled=enabled, dynamic=dynamic, backend=backend, context=context):
                root = nn.Module()
                root.decoder = nn.Module()
                root.decoder.dynamic_shapes = dynamic
                root.decoder.self_attn_backend = backend
                layer = nn.Module()
                layer.mlp = tiny_moe()
                layer.context_attn = context
                layer.ffn_layernorm = False
                layer.parallel_residual = False
                layer.dropout_p = 0
                root.decoder.transformer_layers = nn.ModuleList([layer])
                config = SimpleNamespace(
                    moe_expert_backend="hybrid_int8", moe_gpu_experts=0, moe_expert_cuda_graph=enabled
                )
                with patch("eole.EOLE_TORCH_COMPILE", False), patch("torch.cuda.Stream") as stream:
                    prepare_hybrid_experts(root, config)
                    try:
                        stream.assert_not_called()
                        self.assertFalse(hasattr(layer, "_hybrid_graph"))
                    finally:
                        root._expert_runtime.close()

    def test_ensemble_decoder_without_graph_capability_keeps_eager_execution(self):
        from eole.decoders.ensemble import EnsembleDecoder

        decoders = []
        for _ in range(2):
            decoder = nn.Module()
            decoder.attentional = False
            decoder.mlp = tiny_moe()
            decoders.append(decoder)
        root = nn.Module()
        root.decoder = EnsembleDecoder(decoders)
        self.assertFalse(hasattr(root.decoder, "dynamic_shapes"))
        # Omit the graph option to exercise its enabled-by-default behavior.
        config = SimpleNamespace(moe_expert_backend="hybrid_int8", moe_gpu_experts=0)
        with patch("eole.EOLE_TORCH_COMPILE", False), patch("torch.cuda.Stream") as stream:
            prepare_hybrid_experts(root, config)
            try:
                stream.assert_not_called()
                self.assertFalse(hasattr(root.decoder, "_hybrid_cache_storage"))
                for decoder in root.decoder.model_decoders:
                    self.assertIsInstance(decoder.mlp._executor, HybridExpertExecutor)
                    with torch.inference_mode():
                        output = decoder.mlp(torch.randn(1, 1, 32))
                    self.assertEqual(output.shape, (1, 1, 32))
                    self.assertTrue(output.isfinite().all())
            finally:
                root._expert_runtime.close()

    def test_cpu_int8_error_is_bounded_and_zero_weights_are_finite(self):
        torch.manual_seed(42)
        linear = nn.Linear(128, 32)
        x = torch.randn(8, 128)
        actual = CPUInt8Linear(linear)(x)
        expected = linear(x)
        self.assertLess((actual - expected).square().mean().sqrt().item(), 0.025)
        nn.init.zeros_(linear.weight)
        self.assertTrue(CPUInt8Linear(linear)(x).isfinite().all())

    def test_single_token_worker_preserves_values_and_route_positions(self):
        model = tiny_moe()
        runtime = HybridExpertRuntime()
        executor = HybridExpertExecutor(model.experts, 1, runtime)
        try:
            for dtype in (torch.float32, torch.bfloat16):
                for selected in ([0], [1], [0, 1], [3, 2, 1], [3, 0, 2, 1]):
                    with self.subTest(dtype=dtype, selected=selected), torch.inference_mode():
                        x = torch.randn(1, 32, dtype=dtype)
                        original_x = x.clone()
                        ids = torch.tensor([selected])
                        expected = torch.zeros(1, len(selected), 32)
                        for expert_id in torch.unique(ids).tolist():
                            if expert_id >= executor.resident:
                                token, position = torch.where(ids == expert_id)
                                expected[token, position] = executor.cpu_experts[expert_id](x[token]).float()
                        actual = executor._run_cpu(x, ids)
                        self.assertEqual(actual.dtype, torch.bfloat16)
                        torch.testing.assert_close(actual.float(), expected, rtol=0, atol=0)
                        torch.testing.assert_close(x, original_x, rtol=0, atol=0)
        finally:
            runtime.close()

    def test_cpu_only_executor_preserves_all_routes_and_drains_worker(self):
        model = tiny_moe()
        reference = copy.deepcopy(model)
        runtime = HybridExpertRuntime()
        executor = HybridExpertExecutor(model.experts, 0, runtime)
        try:
            with torch.inference_mode():
                x = torch.randn(3, 32)
                ids = torch.tensor([[0, 1], [2, 3], [0, 3]])
                weights = torch.tensor([[0.3, 0.7], [0.6, 0.4], [0.5, 0.5]])
                actual = executor.execute(x, weights, ids)
                expected = vectorized_moe(x.repeat_interleave(2, 0), weights, ids, 2, reference.experts)
                self.assertLess((actual - expected).square().mean().sqrt().item(), 0.02)
                self.assertTrue(all(isinstance(e, nn.Identity) for e in model.experts))
                with self.assertRaisesRegex(ValueError, "inference only"):
                    executor.execute(x, weights, ids, training=True)
        finally:
            runtime.close()

    @unittest.skipUnless(torch.cuda.is_available(), "Hybrid overlap requires CUDA")
    def test_gpu_cpu_partition_matches_independent_expert_execution(self):
        model = tiny_moe()
        runtime = HybridExpertRuntime()
        model.experts[0].cuda()
        executor = HybridExpertExecutor(model.experts, 1, runtime)
        try:
            with torch.inference_mode():
                x = torch.randn(3, 32, device="cuda")
                ids = torch.tensor([[0, 1], [2, 0], [1, 3]], device="cuda")
                weights = torch.tensor([[0.3, 0.7], [0.6, 0.4], [0.5, 0.5]], device="cuda")
                for rows in (1, 3):
                    with self.subTest(rows=rows):
                        batch_x, batch_ids, batch_weights = x[:rows], ids[:rows], weights[:rows]
                        parts = torch.zeros(rows, 2, 32, device="cuda")
                        for index in range(4):
                            token, position = torch.where(batch_ids == index)
                            if token.numel() == 0:
                                continue
                            values = (
                                model.experts[0](batch_x[token])
                                if index == 0
                                else executor.cpu_experts[index](batch_x[token].cpu()).cuda()
                            )
                            parts[token, position] = values.float()
                        expected = (parts * batch_weights.unsqueeze(-1)).sum(1)
                        torch.testing.assert_close(
                            executor.execute(batch_x, batch_weights, batch_ids), expected, atol=1e-6, rtol=1e-5
                        )
        finally:
            runtime.close()
