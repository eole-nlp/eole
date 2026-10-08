import unittest

from eole.modules.quantization_selection import is_selected


class TestQuantizationSelection(unittest.TestCase):
    def test_legacy_leaf_names(self):
        for root in ("decoder", "mtp_heads.0"):
            self.assertTrue(is_selected(f"{root}.layers.0.mlp.down_proj", ["down_proj"]))
        self.assertFalse(is_selected("decoder.down_proj.child", ["down_proj"]))

    def test_exact_and_glob_paths(self):
        path = "decoder.layers.3.mlp.down_proj"
        self.assertTrue(is_selected(path, [path]))
        self.assertTrue(is_selected(path, ["decoder.layers.*.mlp.down_proj"]))
        self.assertFalse(is_selected("mtp_heads.0.layers.3.mlp.down_proj", ["decoder.*.down_proj"]))
        self.assertFalse(is_selected("decoder.layers.4.mlp.down_proj", [path]))

    def test_exclusions_win_and_skip_subtrees(self):
        path = "decoder.layers.3.mlp.down_proj"
        for exclusion in ("mlp", "decoder.layers.3", "decoder.layers.*.mlp", path):
            self.assertFalse(is_selected(path, ["down_proj", path], [exclusion]))
        self.assertTrue(is_selected(path, ["down_proj"], ["mtp_heads"]))
        self.assertFalse(is_selected("mtp_heads.0.down_proj", ["down_proj"], ["mtp_heads"]))

    def test_empty_includes_select_nothing(self):
        self.assertFalse(is_selected("decoder.down_proj", []))

    def test_packed_checkpoint_selection_mismatch(self):
        import torch.nn as nn
        from eole.models.model import BaseModel

        class Model(nn.Module):
            _checkpoint_key_for_param = staticmethod(lambda key: key)

        model = Model()
        model.down_proj = nn.Linear(8, 8)
        with self.assertRaisesRegex(ValueError, "quant_layers"):
            BaseModel._validate_packed_quantization_selection(model, {"down_proj.qweight": 0})
        BaseModel._validate_packed_quantization_selection(model, {"down_proj.weight": 0})
        model.down_proj = nn.Module()
        model.down_proj.register_buffer("qweight", __import__("torch").zeros(1))
        with self.assertRaisesRegex(ValueError, "floating-point"):
            BaseModel._validate_packed_quantization_selection(model, {"down_proj.weight": 0})
        BaseModel._validate_packed_quantization_selection(model, {"down_proj.qweight": 0})
