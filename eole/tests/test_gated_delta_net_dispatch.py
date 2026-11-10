"""Optional GPU libraries must not change execution on CPU/non-CUDA inputs."""

from contextlib import ExitStack
import copy
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from eole.config.models import TransformerDecoderConfig
from eole.modules import gated_delta_net as gdn


FAST_KERNELS = (
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "chunk_gated_delta_rule",
    "fused_recurrent_gated_delta_rule",
)


def new_layer():
    config = TransformerDecoderConfig(
        decoder_type="transformer",
        layers=2,
        hidden_size=32,
        heads=2,
        transformer_ff=64,
        linear_conv_kernel_dim=3,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
    )
    layer = gdn.GatedDeltaNet(config, layer_idx=0)
    for module in (layer.in_proj_qkv, layer.in_proj_z, layer.in_proj_b, layer.in_proj_a, layer.out_proj):
        module.reset_parameters()
    return layer


def non_cuda_devices():
    return ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


class TestGatedDeltaNetDispatch(unittest.TestCase):
    def test_convolution_activation_normalization(self):
        self.assertIsNone(gdn._causal_conv1d_activation(None))
        self.assertEqual(gdn._causal_conv1d_activation("silu"), "silu")
        self.assertEqual(gdn._causal_conv1d_activation("swish"), gdn._causal_conv1d_activation("silu"))
        with self.assertRaisesRegex(ValueError, "Unsupported causal convolution activation"):
            gdn._causal_conv1d_activation("gelu")

    def test_convolution_activation_contract(self):
        for device in non_cuda_devices() + (["cuda"] if torch.cuda.is_available() else []):
            for activation in (None, "silu", "swish"):
                with self.subTest(device=device, activation=activation):
                    x = torch.randn(1, 16, 7, device=device)
                    weight = torch.randn(16, 3, device=device)
                    bias = torch.randn(16, device=device)
                    # Force the fallback on GPU too, independently of installed packages.
                    with patch.object(gdn, "causal_conv1d_fn", None):
                        actual = gdn._causal_conv1d_dispatch(x, weight, bias, activation)
                    expected = F.conv1d(x, weight[:, None], bias, padding=2, groups=16)[..., :7]
                    if activation is not None:
                        expected = F.silu(expected)
                    torch.testing.assert_close(actual, expected)
                    initial = torch.randn(1, 16, 3, device=device)
                    unchanged_state = initial.clone()
                    expected_state, actual_state = initial.clone(), initial.clone()
                    with patch.object(gdn, "causal_conv1d_update", None):
                        updated = gdn._causal_conv1d_update_dispatch(x[..., :1], actual_state, weight, bias, activation)
                    reference = gdn._torch_causal_conv1d_update(
                        x[..., :1], expected_state, weight, bias, "silu" if activation else None
                    )
                    torch.testing.assert_close(updated, reference)
                    torch.testing.assert_close(actual_state, expected_state)
                    # Unsupported activations must fail before launching either backend
                    # or mutating the decode cache.
                    with (
                        patch.object(gdn, "causal_conv1d_fn") as prefill,
                        patch.object(gdn, "causal_conv1d_update") as decode,
                    ):
                        with self.assertRaisesRegex(ValueError, "Unsupported causal convolution activation"):
                            gdn._causal_conv1d_dispatch(x, weight, bias, "gelu")
                        with self.assertRaisesRegex(ValueError, "Unsupported causal convolution activation"):
                            gdn._causal_conv1d_update_dispatch(x[..., :1], initial, weight, bias, "gelu")
                        prefill.assert_not_called()
                        decode.assert_not_called()
                        torch.testing.assert_close(initial, unchanged_state)

    def test_non_cuda_forward_backward_with_optional_kernels_present(self):
        for device in non_cuda_devices():
            with self.subTest(device=device), ExitStack() as stack:
                kernels = [
                    stack.enter_context(patch.object(gdn, name, side_effect=AssertionError("GPU kernel on CPU/MPS")))
                    for name in FAST_KERNELS
                ]
                if gdn._fla_rmsnorm_gated:
                    kernels.append(
                        stack.enter_context(
                            patch.object(gdn._FLARMSNormGated, "forward", side_effect=AssertionError("GPU norm"))
                        )
                    )
                # Construct with accelerators available, then move to the requested device.
                layer = new_layer().to(device)
                x = torch.randn(2, 7, 32, device=device, requires_grad=True)
                output = layer(x)
                output.square().mean().backward()
                self.assertEqual(output.device.type, device)
                self.assertTrue(torch.isfinite(output).all())
                self.assertTrue(torch.isfinite(x.grad).all())
                for name, parameter in layer.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                for kernel in kernels:
                    kernel.assert_not_called()

    def test_cpu_prefill_and_cached_decode_match_full_sequence(self):
        torch.manual_seed(77)
        with ExitStack() as stack:
            for name in FAST_KERNELS:
                stack.enter_context(patch.object(gdn, name, side_effect=AssertionError("GPU kernel on CPU")))
            full = new_layer().eval()
            cached = copy.deepcopy(full)
            cached.conv_state = torch.zeros(1, cached.conv_dim, cached.conv_kernel_size)
            cached.recurrent_state = torch.zeros(1, cached.num_v_heads, cached.head_k_dim, cached.head_v_dim)
            x = torch.randn(1, 7, 32)
            with torch.no_grad():
                expected = full(x)
                prefix = cached(x[:, :3])
                actual = torch.cat([prefix] + [cached(x[:, i : i + 1]) for i in range(3, 7)], dim=1)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

    def test_cpu_compile_does_not_launch_fla_norm(self):
        if not gdn._fla_rmsnorm_gated:
            self.skipTest("FLA norm unavailable")
        norm = gdn.RMSNormGated(16)
        x = torch.randn(2, 3, 16, requires_grad=True)
        gate = torch.randn_like(x, requires_grad=True)
        with patch.object(gdn, "_compiled_fla_rms_norm_gated", side_effect=AssertionError("GPU adapter")) as kernel:
            compiled = torch.compile(norm, backend="aot_eager", fullgraph=True)
            actual = compiled(x, gate)
            expected = norm(x, gate)
            torch.testing.assert_close(actual, expected)
            actual.sum().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            kernel.assert_not_called()

    def test_cpu_fla_norm_preserves_residual_and_prenorm_contract(self):
        if not gdn._fla_rmsnorm_gated:
            self.skipTest("FLA norm unavailable")
        for dtype in (torch.float32, torch.bfloat16):
            for activation in ("silu", "swish", "sigmoid"):
                for has_residual, residual_in_fp32 in ((False, False), (False, True), (True, False)):
                    with self.subTest(dtype=dtype, activation=activation, residual=has_residual, fp32=residual_in_fp32):
                        norm = gdn.RMSNormGated(16, activation=activation).to(dtype)
                        x = torch.randn(2, 3, 16, dtype=dtype, requires_grad=True)
                        gate = torch.randn_like(x, requires_grad=True)
                        residual = torch.randn_like(x, dtype=torch.float32) if has_residual else None
                        values = x.float() if residual is None else x.float() + residual
                        expected = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + norm.eps)
                        expected = expected * norm.weight.float()
                        expected *= torch.sigmoid(gate.float()) if activation == "sigmoid" else F.silu(gate.float())
                        actual, residual_out = norm(
                            x, gate, residual=residual, prenorm=True, residual_in_fp32=residual_in_fp32
                        )
                        torch.testing.assert_close(actual, expected.to(dtype))
                        expected_dtype = torch.float32 if has_residual or residual_in_fp32 else dtype
                        self.assertEqual(residual_out.dtype, expected_dtype)
                        torch.testing.assert_close(residual_out, values.to(expected_dtype))
                        (actual.float().sum() + residual_out.float().sum()).backward()
                        self.assertTrue(torch.isfinite(x.grad).all())
                        self.assertTrue(torch.isfinite(gate.grad).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_dispatch_follows_inputs_after_cpu_cuda_cpu_moves(self):
        with ExitStack() as stack:
            # Wrap reference implementations so the test verifies GPU selection independently
            # of which optional packages are installed, including cache mutation.
            kernels = [
                stack.enter_context(patch.object(gdn, name, wraps=reference))
                for name, reference in zip(
                    FAST_KERNELS,
                    (
                        lambda x, w, b, a: F.silu(
                            F.conv1d(x, w[:, None], b, padding=w.shape[-1] - 1, groups=x.shape[1])[..., : x.shape[-1]]
                        ),
                        gdn._torch_causal_conv1d_update,
                        gdn._torch_chunk_gated_delta_rule,
                        gdn._torch_recurrent_gated_delta_rule,
                    ),
                )
            ]
            layer = new_layer().eval()
            keys = set(layer.state_dict())
            for device in ("cpu", "cuda", "cpu"):
                for kernel in kernels:
                    kernel.reset_mock()
                layer.to(device)
                layer.conv_state = torch.zeros(1, layer.conv_dim, layer.conv_kernel_size, device=device)
                layer.recurrent_state = torch.zeros(
                    1, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim, device=device
                )
                with torch.no_grad():
                    layer(torch.randn(1, 4, 32, device=device))
                    layer(torch.randn(1, 1, 32, device=device))
                for kernel in kernels:
                    self.assertEqual(kernel.call_count, int(device == "cuda"))
                self.assertEqual(set(layer.state_dict()), keys)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_real_cuda_fla_fullgraph_decode_preserves_caches(self):
        if not gdn._fla_rmsnorm_gated or gdn.chunk_gated_delta_rule is None:
            self.skipTest("FLA unavailable")
        layer = new_layer().cuda().eval()
        layer.conv_state = torch.zeros(1, layer.conv_dim, layer.conv_kernel_size, device="cuda")
        layer.recurrent_state = torch.zeros(1, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim, device="cuda")
        x = torch.randn(1, 1, 32, device="cuda")
        conv, state = layer.conv_state.clone(), layer.recurrent_state.clone()
        with torch.no_grad():
            expected = layer(x)
            expected_conv, expected_state = layer.conv_state.clone(), layer.recurrent_state.clone()
            layer.conv_state.copy_(conv)
            layer.recurrent_state.copy_(state)
            compiled = torch.compile(layer, backend="aot_eager", fullgraph=True)
            actual = compiled(x)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(layer.conv_state, expected_conv)
        torch.testing.assert_close(layer.recurrent_state, expected_state, atol=1e-5, rtol=1e-4)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_real_cuda_fla_forward_backward_matches_cpu(self):
        if not gdn._fla_rmsnorm_gated or gdn.chunk_gated_delta_rule is None:
            self.skipTest("FLA unavailable")
        torch.manual_seed(91)
        cpu = new_layer()
        # Keep decay away from underflow so CPU/GPU derivative comparisons are
        # well-conditioned (the randomly initialized A can otherwise be large).
        with torch.no_grad():
            cpu.A_log.fill_(-2.0)
            cpu.dt_bias.zero_()
        cuda = copy.deepcopy(cpu).cuda()
        x = torch.randn(1, 9, 32)
        cpu_input = x.clone().requires_grad_()
        cuda_input = x.cuda().requires_grad_()
        expected = cpu(cpu_input)
        actual = cuda(cuda_input)
        torch.testing.assert_close(actual.cpu(), expected, atol=5e-4, rtol=5e-3)
        expected.square().mean().backward()
        actual.square().mean().backward()
        # FLA uses reduced-precision tensor-core intermediates even for FP32
        # inputs; absolute tolerances account for cancellation near zero.
        torch.testing.assert_close(cuda_input.grad.cpu(), cpu_input.grad, atol=1e-5, rtol=1e-2)
        for (name, p), (cuda_name, cp) in zip(cpu.named_parameters(), cuda.named_parameters()):
            self.assertEqual(name, cuda_name)
            torch.testing.assert_close(cp.grad.cpu(), p.grad, atol=5e-5, rtol=1e-2, msg=name)


if __name__ == "__main__":
    unittest.main()
