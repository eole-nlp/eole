import unittest
from unittest.mock import patch

import torch.nn as nn

from eole.modules import gated_delta_net as gdn
from eole.utils.inference_backends import inference_backend_summary


class TestInferenceBackends(unittest.TestCase):
    def _model(self):
        model = nn.Module()
        model.decoder = nn.Module()
        model.decoder.projection = nn.Linear(4, 4)
        layer = gdn.GatedDeltaNet.__new__(gdn.GatedDeltaNet)
        nn.Module.__init__(layer)
        layer._causal_conv1d_fn = None
        layer._causal_conv1d_update = gdn._torch_causal_conv1d_update
        layer._chunk_gated_delta_rule = gdn._torch_chunk_gated_delta_rule
        layer._recurrent_gated_delta_rule = gdn._torch_recurrent_gated_delta_rule
        model.decoder.transformer_layers = nn.ModuleList([layer])
        model.mtp_heads = nn.ModuleList()
        return model

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
