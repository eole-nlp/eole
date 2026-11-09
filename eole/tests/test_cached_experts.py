"""Hybrid GPTQ residency and preserved checkpoint codes."""

from pathlib import Path
from threading import Lock
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn
import torch.nn.functional as F

from eole.modules.expert_cache import FrequencyExpertCache
from eole.modules.host_experts import HostGPTQLinear, HostInt8Linear
from eole.modules.hybrid_experts import HybridExpertRuntime


def projection(k=256, n=128):
    from eole.modules.marlin_linear import MarlinQuantLinear

    linear = MarlinQuantLinear(4, 128, False, True, k, n).bfloat16()
    linear.qweight.data.random_(torch.iinfo(torch.int32).min, torch.iinfo(torch.int32).max)
    linear.qzeros.data.fill_(-2004318072)
    linear.scales.data.uniform_(0.001, 0.01)
    linear.g_idx.data.copy_(torch.arange(k, dtype=torch.int32) // 128)
    return linear


def experts(count=4):
    result = nn.ModuleList()
    for _ in range(count):
        expert = nn.Module()
        expert.activation = F.silu
        expert.gate_up_proj = projection(2048, 512)
        expert.up_proj = projection(2048, 512)
        expert.down_proj = projection(512, 2048)
        result.append(expert)
    return result


class TestExpertCache(unittest.TestCase):
    def test_frequency_admission_is_bounded_and_explicit(self):
        policy = FrequencyExpertCache(6, 2, interval=2, swaps=1)
        policy.observe(torch.tensor([[3, 4]]))
        self.assertEqual(policy.replacements(), [])
        policy.observe(torch.tensor([[3, 4]]))
        self.assertEqual(policy.replacements(), [(0, 3)])
        self.assertEqual(policy.resident, {0, 1})
        policy.commit(0, 3)
        self.assertEqual(policy.resident, {1, 3})
        with self.assertRaisesRegex(ValueError, "Stale"):
            policy.commit(0, 4)
        # Equal demand cannot evict a trained resident.
        policy.scores.fill_(1)
        self.assertEqual(policy.replacements(), [])

    def test_cpu_layout_preserves_original_gptq_values(self):
        linear = projection()
        host = HostGPTQLinear(linear)
        shifts = torch.arange(8, dtype=torch.int64) * 4
        codes = ((linear.qweight.long().unsqueeze(1) >> shifts[None, :, None]) & 15).reshape(256, 128)
        weight = (codes.double() - 8) * linear.scales.double().repeat_interleave(128, 0)
        x = torch.randn(3, 256, dtype=torch.bfloat16)
        expected = x.double() @ weight
        error = (host(x).double() - expected).square().mean().sqrt() / expected.square().mean().sqrt()
        self.assertLess(error.item(), 0.004)
        torch.testing.assert_close(host.scales_zeros[..., 0], linear.scales, rtol=0, atol=0)
        linear.qzeros.data.zero_()
        with self.assertRaisesRegex(ValueError, "zero point 8"):
            HostGPTQLinear(linear)

    def test_unsupported_checkpoint_formats_are_rejected_before_packing(self):
        linear = projection()
        for attribute, value in (
            ("checkpoint_weight_layout", "awq"),
            ("bits", 8),
            ("group_size", 64),
            ("desc_act", True),
            ("bias", nn.Parameter(torch.zeros(128), requires_grad=False)),
            ("scales", nn.Parameter(linear.scales.half(), requires_grad=False)),
        ):
            original = getattr(linear, attribute)
            with self.subTest(attribute=attribute):
                setattr(linear, attribute, value)
                try:
                    with self.assertRaisesRegex(ValueError, "Hybrid GPTQ requires"):
                        HostGPTQLinear(linear)
                finally:
                    setattr(linear, attribute, original)

    def test_repacked_marlin_weights_cannot_be_read_as_gptq(self):
        linear = projection()
        linear.qweight = nn.Parameter(torch.zeros(16, 256, dtype=torch.int32), requires_grad=False)
        with self.assertRaisesRegex(ValueError, "unrepacked CPU GPTQ"):
            HostGPTQLinear(linear)

    def test_int8_cpu_codes_scales_and_zero_weights(self):
        torch.manual_seed(87)
        source = nn.Linear(256, 128, bias=False, dtype=torch.bfloat16)
        host = HostInt8Linear(source)
        x = torch.randn(3, 256, dtype=torch.bfloat16)
        reference = x.double() @ (host.weight.double() * host.scales.double()[:, None]).t()
        torch.testing.assert_close(host(x).double(), reference, rtol=0.004, atol=0.0002)
        reconstructed = host.weight.float() * host.scales.float()[:, None]
        error = (
            reconstructed - source.weight.float()
        ).square().mean().sqrt() / source.weight.float().square().mean().sqrt()
        self.assertLess(error.item(), 0.012)
        with torch.no_grad():
            source.weight.zero_()
        self.assertEqual(HostInt8Linear(source)(x).count_nonzero().item(), 0)

    def test_adaptive_float_placement_keeps_all_experts_on_host(self):
        from eole.modules.expert_placement import ExpertWeightPlacement
        from eole.tests.test_expert_execution import tiny_moe

        model = nn.Module()
        model.mlp = tiny_moe()
        config = SimpleNamespace(moe_expert_backend="adaptive", moe_gpu_experts=2)
        with patch("eole.EOLE_TORCH_COMPILE", False):
            placement = ExpertWeightPlacement(model, config, torch.device("cuda"))
        for i in range(4):
            self.assertEqual(placement.device_for(f"mlp.experts.{i}.gate_up_proj"), torch.device("cpu"))
        self.assertEqual(placement.device_for("mlp.gate"), torch.device("cuda"))

    @unittest.skipUnless(torch.cuda.is_available(), "Marlin execution requires CUDA")
    def test_int8_gpu_and_cpu_represent_identical_quantized_weights(self):
        torch.manual_seed(88)
        host = HostInt8Linear(nn.Linear(2048, 512, bias=False, dtype=torch.bfloat16))
        with torch.inference_mode():
            projection = host.marlin_projection(torch.device("cuda"))
            projection.post_init()
            x = torch.randn(4, 2048, dtype=torch.bfloat16)
            expected = x.double() @ (host.weight.double() * host.scales.double()[:, None]).t()
            for output in (host(x).double(), projection(x.cuda()).cpu().double()):
                error = (output - expected).square().mean().sqrt() / expected.square().mean().sqrt()
                self.assertLess(error.item(), 0.004)

    def test_quantized_placement_keeps_all_source_experts_on_host(self):
        from eole.modules.expert_placement import ExpertWeightPlacement
        from eole.tests.test_expert_execution import tiny_moe

        model = nn.Module()
        model.mlp = tiny_moe()
        config = SimpleNamespace(moe_expert_backend="hybrid_gptq", moe_gpu_experts=4, quant_type="autoround")
        with patch("eole.EOLE_TORCH_COMPILE", False):
            placement = ExpertWeightPlacement(model, config, torch.device("cuda"))
        for i in range(4):
            self.assertEqual(placement.device_for(f"mlp.experts.{i}.gate_up_proj"), torch.device("cpu"))
        self.assertEqual(placement.device_for("mlp.gate"), torch.device("cuda"))
        self.assertTrue(model.mlp._defer_expert_post_init)

    def test_loader_defers_only_routed_experts(self):
        from eole.modules.autoround_linear import post_init_autoround_linear

        class Projection(nn.Module):
            def __init__(self):
                super().__init__()
                self.initialized = False

            def post_init(self):
                self.initialized = True

        root = nn.Module()
        root.moe = nn.Module()
        root.moe._defer_expert_post_init = True
        root.moe.experts = nn.ModuleList([Projection()])
        root.moe.shared_experts = Projection()
        root.moe._maybe_init_quant_weights = lambda *args: self.fail("Stacked host experts")
        post_init_autoround_linear(root)
        self.assertFalse(root.moe.experts[0].initialized)
        self.assertTrue(root.moe.shared_experts.initialized)

    def test_fully_resident_executor_does_not_read_ids_back_to_cpu(self):
        from eole.modules.cached_experts import CachedExpertExecutor

        executor = CachedExpertExecutor.__new__(CachedExpertExecutor)
        executor.closed, executor.fully_resident, executor.capture = False, True, True
        executor.lock = Lock()
        executor.routes = executor.lookup_hits = executor.gpu_hits = 0
        parts = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        executor.gpu = Mock(return_value=parts)
        ids = SimpleNamespace(numel=lambda: 2, cpu=lambda: self.fail("Unnecessary routing readback"))
        with torch.inference_mode():
            result = executor.execute(torch.zeros(1, 2), torch.tensor([[0.25, 0.75]]), ids)
        torch.testing.assert_close(result, torch.tensor([[2.5, 3.5]]))
        self.assertEqual(executor.lookup_hits, 2)

    def test_failed_admission_never_publishes_partial_upload(self):
        from eole.modules.cached_experts import CachedExpertExecutor

        executor = CachedExpertExecutor.__new__(CachedExpertExecutor)
        executor.adaptive = True
        executor.policy = FrequencyExpertCache(4, 1, interval=1)
        executor.mapping, executor.spare, executor.migrations = {0: 0}, 1, 0
        with patch.object(executor, "_upload", side_effect=RuntimeError("upload failed")):
            with patch.object(executor, "_publish") as publish:
                with self.assertRaisesRegex(RuntimeError, "upload failed"):
                    executor._admit(torch.tensor([[2, 3]]))
                publish.assert_not_called()
        self.assertEqual(executor.mapping, {0: 0})
        self.assertEqual(executor.policy.resident, {0})
        self.assertEqual(executor.spare, 1)

    @unittest.skipUnless(torch.cuda.is_available(), "Marlin packing requires CUDA")
    def test_failed_disk_preallocation_cleans_temporary_store(self):
        from eole.modules.host_experts import MarlinExpertStore

        with tempfile.TemporaryDirectory() as directory:
            with patch("eole.modules.host_experts.os.posix_fallocate", side_effect=OSError("No disk space")):
                with self.assertRaisesRegex(OSError, "No disk space"):
                    MarlinExpertStore(experts(), torch.device("cuda"), directory)
            self.assertEqual(list(Path(directory).iterdir()), [])

    @unittest.skipUnless(torch.cuda.is_available(), "Marlin execution requires CUDA")
    def test_decode_replay_prefill_and_cache_replacement(self):
        for floating in (False, True):
            with self.subTest(floating=floating):
                self.check_decode_replay_prefill_and_cache_replacement(floating)

    def check_decode_replay_prefill_and_cache_replacement(self, floating):
        from eole.modules.cached_experts import CachedExpertExecutor

        runtime = HybridExpertRuntime()
        with tempfile.TemporaryDirectory() as directory:
            config = SimpleNamespace(
                moe_expert_backend="adaptive" if floating else "hybrid_gptq",
                moe_gpu_experts=2,
                moe_expert_cache_interval=1,
                moe_expert_cache_swaps=1,
                moe_expert_cache_policy="fixed",
                moe_expert_cuda_graph=True,
                moe_expert_cache_dir=directory,
            )
            executor = None
            try:
                with torch.inference_mode():
                    source = experts()
                    if floating:
                        for expert in source:
                            expert.gate_up_proj = nn.Linear(2048, 512, bias=False, dtype=torch.bfloat16)
                            expert.up_proj = nn.Linear(2048, 512, bias=False, dtype=torch.bfloat16)
                            expert.down_proj = nn.Linear(512, 2048, bias=False, dtype=torch.bfloat16)
                    executor = CachedExpertExecutor(source, runtime, config, torch.device("cuda"))
                    self.assertEqual(executor.store.bits, 8 if floating else 4)
                    runtime.executors.append(executor)
                    x = torch.randn(1, 2048, device="cuda", dtype=torch.bfloat16)
                    ids = torch.tensor([[0, 2]], device="cuda")
                    weights = torch.tensor([[0.4, 0.6]], device="cuda")
                    executor.capture = False
                    expected = executor.execute(x, weights, ids).clone()
                    executor.capture = True
                    actual = executor.execute(x, weights, ids).clone()
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    saved_scales = executor.gpu.tensors[1][0].clone()
                    executor.gpu.tensors[1][0].fill_(float("nan"))
                    misses = torch.tensor([[2, 3]], device="cuda")
                    self.assertTrue(executor.execute(x, weights, misses).isfinite().all())
                    executor.gpu.tensors[1][0].copy_(saved_scales)
                    graph = executor.gpu.graph
                    pointer = executor.gpu.decode_scratch[0].data_ptr()
                    # Larger prefill scratch must not invalidate decode capture.
                    executor.execute(x.repeat(7, 1), weights.repeat(7, 1), ids.repeat(7, 1))
                    # Swap the same logical expert into a previously missing GPU slot.
                    executor.adaptive = True
                    executor.execute(x, weights, ids)
                    self.assertEqual(executor.gpu.decode_scratch[0].data_ptr(), pointer)
                    self.assertIs(executor.gpu.graph, graph)
                    self.assertGreater(executor.migrations, 0)
                    executor.capture = False
                    eager = executor.execute(x, weights, ids).clone()
                    executor.adaptive = False
                    executor.capture = True
                    replay = executor.execute(x, weights, ids).clone()
                    torch.testing.assert_close(replay, eager, rtol=0, atol=0)
                    # Compare all routes to independent original-code CPU execution.
                    host_parts = torch.stack([e(x.cpu())[0] for e in executor.store.cpu_experts])
                    reference = (host_parts[ids.cpu()[0]].cuda() * weights[0, :, None]).sum(0)
                    error = (replay[0] - reference).square().mean().sqrt() / reference.square().mean().sqrt()
                    self.assertLess(error.item(), 0.012)
            finally:
                runtime.close()
            self.assertEqual(list(Path(directory).iterdir()), [])
            self.assertTrue(executor.closed)
            with self.assertRaisesRegex(RuntimeError, "closed"):
                executor.execute(x, weights, ids)


if __name__ == "__main__":
    unittest.main()
