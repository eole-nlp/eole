from contextlib import ExitStack
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from eole.config.models import TransformerDecoderConfig
from eole.modules import gated_delta_net as gdn
from eole.utils.inference_backends import inference_backend_summary


class TestInferenceBackends(unittest.TestCase):
    def _model(self):
        model = nn.Module()
        model.decoder = nn.Module()
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
        model.decoder.transformer_layers = nn.ModuleList([gdn.GatedDeltaNet(config, layer_idx=0)])
        return model

    def _assert_pytorch_dispatch(self, output):
        for phase in ("conv prefill", "conv decode", "GDN prefill", "GDN decode/verify/replay"):
            self.assertIn(f"Inference {phase}: PyTorch", output)
        self.assertNotIn("_dispatch", output)

    def test_constructed_cpu_layer_resolves_dispatchers(self):
        model = self._model()
        with ExitStack() as stack:
            for name in (
                "causal_conv1d_fn",
                "causal_conv1d_update",
                "chunk_gated_delta_rule",
                "fused_recurrent_gated_delta_rule",
            ):
                stack.enter_context(patch.object(gdn, name, side_effect=AssertionError("must not execute")))
            output = "\n".join(inference_backend_summary(model, False, "0"))
        self._assert_pytorch_dispatch(output)
        if gdn._fla_rmsnorm_gated:
            self.assertIn("(PyTorch fallback)", output)
            self.assertNotIn("(FLA; inference compile adapter)", output)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_constructed_layer_summary_follows_device_moves_and_availability(self):
        model = self._model()
        with ExitStack() as stack:
            # Real Python functions retain provider metadata for diagnostic naming.
            def conv(x, weight, bias=None, *, activation=None):
                raise AssertionError("must not execute")

            conv.__module__ = "causal_conv1d.causal_conv1d_interface"

            def delta(*args, **kwargs):
                raise AssertionError("must not execute")

            delta.__module__ = "fla.ops.gated_delta_rule"

            def recurrent(*args, **kwargs):
                raise AssertionError("must not execute")

            for name, fn in (
                ("causal_conv1d_fn", conv),
                ("causal_conv1d_update", conv),
                ("chunk_gated_delta_rule", delta),
                ("fused_recurrent_gated_delta_rule", recurrent),
            ):
                stack.enter_context(patch.object(gdn, name, fn))
            for device in ("cpu", "cuda", "cpu"):
                with self.subTest(device=device):
                    model.to(device)
                    output = "\n".join(inference_backend_summary(model, False, "0"))
                    if device == "cpu":
                        self._assert_pytorch_dispatch(output)
                    else:
                        self.assertIn("conv prefill: causal-conv1d", output)
                        self.assertIn("conv decode: causal-conv1d", output)
                        self.assertIn("GDN prefill: fla.ops.gated_delta_rule", output)
                        self.assertIn("GDN decode/verify/replay: FLA", output)
                        self.assertNotIn("_dispatch", output)
                        if gdn._fla_rmsnorm_gated:
                            self.assertIn("(FLA; inference compile adapter)", output)
            model.cuda()
            with ExitStack() as missing:
                for name in (
                    "causal_conv1d_fn",
                    "causal_conv1d_update",
                    "chunk_gated_delta_rule",
                    "fused_recurrent_gated_delta_rule",
                ):
                    missing.enter_context(patch.object(gdn, name, None))
                self._assert_pytorch_dispatch("\n".join(inference_backend_summary(model, False, "0")))

    def test_explicit_kernel_overrides_are_preserved(self):
        model = self._model()
        layer = model.decoder.transformer_layers[0]
        layer._causal_conv1d_fn = None
        layer._causal_conv1d_update = gdn._torch_causal_conv1d_update
        layer._chunk_gated_delta_rule = gdn._torch_chunk_gated_delta_rule
        layer._recurrent_gated_delta_rule = gdn._torch_recurrent_gated_delta_rule
        self._assert_pytorch_dispatch("\n".join(inference_backend_summary(model, False, "0")))

    @patch("eole.utils.inference_backends._version", return_value="test-version")
    def test_installed_packages_do_not_imply_active_kernels(self, version):
        output = "\n".join(inference_backend_summary(self._model(), False, "0"))
        self.assertIn("FlashAttention KV-cache selected=False", output)
        self.assertIn("fla-core=test-version", output)
        self.assertIn("GDN decode/verify/replay: PyTorch", output)
        self.assertIn("decoder path=eager", output)

    def test_compile_configuration_and_speculative_convolution(self):
        model = self._model()
        model.decoder._forward_compile = lambda *args: None
        output = "\n".join(inference_backend_summary(model, True, "0", True, 3))
        self.assertIn("decoder path=compiled", output)
        self.assertIn("CUDA graphs requested=True; capture not verified", output)
        self.assertIn("drafts=3, verifier=compiled", output)
        self.assertIn("conv verify: compiled PyTorch stencil + SiLU", output)
        self.assertIn("recurrent head=eager", output)

    def test_layer_compile_leaves_verification_eager(self):
        model = self._model()
        model.decoder.transformer_layers[0]._forward_compile = lambda *args: None
        output = "\n".join(inference_backend_summary(model, True, "2", True, 3))
        self.assertIn("layers path=compiled", output)
        self.assertIn("verifier=eager", output)
        self.assertIn("conv verify: PyTorch grouped conv1d", output)


if __name__ == "__main__":
    unittest.main()
