"""Packed INT4 conversion and mixed-precision module selection regressions."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from safetensors.torch import save_file, load_file

from eole.bin.convert.compressed_tensors import repack_int4, validate_config
from eole.bin.convert.convert_HF import (
    HuggingfaceFiles,
    build_config_dict,
    build_shards,
    check_conversion_equality,
)
from eole.modules.autoround_linear import replace_autoround_linear
from eole.models.hf_loader import HFLoader, _InMemoryTensorStore
from eole.models.model import BaseModel
from eole.config.run import PredictConfig


def quant_config():
    return {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "group",
                    "group_size": 128,
                    "actorder": "static",
                },
                "input_activations": None,
                "output_activations": None,
            }
        },
    }


def packed_fixture(n=64, k=256):
    # Independent source packing: offset signed [-8,7] values by 8 and store
    # eight consecutive inputs in each little-endian int32 word.
    signed = (torch.arange(n * k).reshape(n, k) * 7 % 16 - 8).to(torch.int8)
    values = (signed.to(torch.int64) + 8).reshape(n, k // 8, 8)
    shifts = torch.arange(8, dtype=torch.int64) * 4
    packed = (values << shifts).sum(-1).to(torch.int32)
    scales = (torch.arange(n * (k // 128)).reshape(n, k // 128) + 1).float() / 256
    return signed, packed, scales, torch.tensor([n, k])


class TestCompressedTensors(unittest.TestCase):
    def test_repacking_preserves_dequantized_weights_and_linear_output(self):
        signed, packed, scales, shape = packed_fixture()
        converted = repack_int4(packed, scales, shape, 128)
        shifts = torch.arange(8, dtype=torch.int32) * 4
        # Independently decode the destination GPTQ layout along its input axis.
        codes = ((converted["qweight"].unsqueeze(1) >> shifts.view(1, 8, 1)) & 15).reshape(256, 64)
        zeros = ((converted["qzeros"].unsqueeze(-1) >> shifts) & 15).reshape(2, 64) + 1
        weight = (codes.float() - zeros[converted["g_idx"]]) * converted["scales"][converted["g_idx"]]
        reference = signed.float() * scales.repeat_interleave(128, dim=1)
        self.assertTrue(torch.equal(weight.t(), reference))
        x = torch.randn(3, 256, generator=torch.Generator().manual_seed(42))
        torch.testing.assert_close(x @ weight, x @ reference.t().contiguous(), rtol=0, atol=0)
        self.assertEqual(converted["qweight"].dtype, torch.int32)

    def test_supported_group_sizes_preserve_group_assignment(self):
        signed, packed, _, shape = packed_fixture()
        for group_size in (32, 64, 128):
            with self.subTest(group_size=group_size):
                scales = torch.arange(64 * (256 // group_size)).reshape(64, 256 // group_size).float() / 256 + 1
                converted = repack_int4(packed, scales, shape, group_size)
                self.assertTrue(torch.equal(converted["g_idx"], torch.arange(256) // group_size))
                reference = signed.float() * scales.repeat_interleave(group_size, dim=1)
                actual = signed.float() * converted["scales"][converted["g_idx"]].t()
                self.assertTrue(torch.equal(reference, actual))

    def test_rejects_unsupported_schemes(self):
        for field, value in (
            ("symmetric", False),
            ("num_bits", 8),
            ("strategy", "channel"),
            ("actorder", "group"),
            ("group_size", 16),
        ):
            with self.subTest(field=field):
                config = quant_config()
                config["config_groups"]["group_0"]["weights"][field] = value
                with self.assertRaises(ValueError):
                    validate_config(config)
        config = quant_config()
        config["config_groups"]["group_0"]["input_activations"] = {"num_bits": 8}
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_rejects_malformed_tensors(self):
        _, packed, scales, shape = packed_fixture()
        for p, s, sh in (
            (packed.float(), scales, shape),
            (packed, scales[:, :1], shape),
            (packed, scales, torch.tensor([64, 255])),
        ):
            with self.assertRaises(ValueError):
                repack_int4(p, s, sh, 128)

    def test_exact_selection_keeps_gdn_controls_mtp_and_other_layers_float(self):
        model = nn.Module()
        model.decoder = nn.Module()
        model.decoder.layers = nn.ModuleList([nn.Module(), nn.Module()])
        for layer in model.decoder.layers:
            layer.in_proj_qkv = nn.Linear(256, 64)
            layer.in_proj_a = nn.Linear(256, 64)
        model.mtp_heads = nn.ModuleList([nn.Module()])
        model.mtp_heads[0].in_proj_qkv = nn.Linear(256, 64)

        class QuantLinear(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()

        with patch("eole.modules.autoround_linear._get_autoround_quant_linear_cls", return_value=(QuantLinear, False)):
            replace_autoround_linear(
                model, ["in_proj_qkv", "in_proj_a"], quantized_modules=["decoder.layers.0.in_proj_qkv"]
            )
        self.assertIsInstance(model.decoder.layers[0].in_proj_qkv, QuantLinear)
        self.assertIsInstance(model.decoder.layers[0].in_proj_a, nn.Linear)
        self.assertIsInstance(model.decoder.layers[1].in_proj_qkv, nn.Linear)
        self.assertIsInstance(model.mtp_heads[0].in_proj_qkv, nn.Linear)

    def test_vision_model_uses_exact_paths_when_replacing_subtrees(self):
        class Vision(nn.Module):
            pass

        model = nn.Module()
        model.encoder = Vision()
        model.decoder = nn.Module()
        model.decoder.in_proj_qkv = nn.Linear(128, 64)
        model.mtp_heads = nn.ModuleList([nn.Module()])
        model.mtp_heads[0].in_proj_qkv = nn.Linear(128, 64)

        class QuantLinear(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()

        running = SimpleNamespace(
            quant_type="autoround",
            quant_layers=["in_proj_qkv"],
            w_bit=4,
            group_size=128,
            quantized_modules=["decoder.in_proj_qkv"],
        )
        with (
            patch("eole.models.model.VisionEncoder", Vision),
            patch("eole.modules.autoround_linear._get_autoround_quant_linear_cls", return_value=(QuantLinear, False)),
        ):
            BaseModel.maybe_quantize(model, running)
        self.assertIsInstance(model.decoder.in_proj_qkv, QuantLinear)
        self.assertIsInstance(model.mtp_heads[0].in_proj_qkv, nn.Linear)

    def test_prediction_config_inherits_exact_selection_and_group_size(self):
        with tempfile.TemporaryDirectory() as directory:
            modules = ["decoder.transformer_layers.0.mlp.down_proj"]
            config = {
                "model": {"architecture": "transformer_lm", "decoder": {"layers": 1, "hidden_size": 128, "heads": 2}},
                "training": {
                    "quant_type": "autoround",
                    "w_bit": 4,
                    "group_size": 64,
                    "quant_layers": ["down_proj"],
                    "quantized_modules": modules,
                },
            }
            (Path(directory) / "config.json").write_text(json.dumps(config))
            predict = PredictConfig(model_path=directory, src="unused", self_attn_backend="pytorch")
            self.assertEqual(predict.quantized_modules, modules)
            self.assertEqual(predict.group_size, 64)

    def test_sharded_qwen_conversion_with_float_controls_and_mtp(self):
        self._check_qwen_conversion(indexed=True)

    def test_unindexed_qwen_conversion_with_float_controls_and_mtp(self):
        self._check_qwen_conversion(indexed=False)

    def _check_qwen_conversion(self, indexed):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "quantization_config": quant_config(),
                "text_config": {
                    "hidden_size": 256,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 4,
                    "head_dim": 64,
                    "mtp_num_hidden_layers": 1,
                },
            }
            (root / "config.json").write_text(json.dumps(config))
            _, packed, scales, shape = packed_fixture()
            prefix = "model.language_model.layers.0.linear_attn.in_proj_qkv"
            # Compression metadata deliberately lives in a different shard.
            first = {
                prefix + ".weight_packed": packed,
                "model.language_model.layers.0.linear_attn.in_proj_a.weight": torch.randn(64, 256),
                "mtp.layers.0.self_attn.q_proj.weight": torch.randn(64, 256),
            }
            second = {prefix + ".weight_scale": scales, prefix + ".weight_shape": shape}
            weight_map = {}
            for name, tensors in (("model-1.safetensors", first), ("model-2.safetensors", second)):
                save_file(tensors, str(root / name))
                weight_map.update({key: name for key in tensors})
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
            if not indexed:
                save_file({**first, **second}, str(root / "model.safetensors"))
            hf = HuggingfaceFiles(
                token=None,
                tokenizer_config_json=None,
                config_path=str(root / "config.json"),
                model_dir=str(root),
                wmap_path=str(root / "model.safetensors.index.json") if indexed else None,
                model_path=str(root / "model.safetensors") if not indexed else None,
            )
            model_config, training, params = build_config_dict(hf)
            self.assertEqual(training["quantized_modules"], ["decoder.transformer_layers.0.linear_attn.in_proj_qkv"])
            self.assertEqual(hf.mtp_layer_prefix, "mtp.layers.")
            output = root / "out"
            output.mkdir()
            args = SimpleNamespace(nshards=1, output=str(output), dtype="bf16")
            all_keys, consumed, details = build_shards(model_config, hf, args, params)
            self.assertEqual(all_keys, consumed)
            self.assertEqual(check_conversion_equality(hf, details, torch.bfloat16), [])
            tensors = load_file(str(output / "model.00.safetensors"))
            dest = "decoder.transformer_layers.0.linear_attn.in_proj_qkv"
            self.assertTrue(torch.equal(tensors[dest + ".qweight"], packed.t().contiguous()))
            self.assertEqual(tensors[dest + ".qweight"].dtype, torch.int32)
            self.assertEqual(tensors[dest + ".g_idx"].dtype, torch.int32)
            self.assertEqual(tensors[dest + ".scales"].dtype, torch.bfloat16)
            self.assertIn("decoder.transformer_layers.0.linear_attn.in_proj_a.weight", tensors)
            self.assertIn("mtp_heads.0.layer.self_attn.linear_query.weight", tensors)
            store = _InMemoryTensorStore()
            HFLoader(str(root))._build_tensors(hf, model_config, params, store)
            self.assertEqual(set(store.keys()), set(tensors))
            for key in tensors:
                source = store.get_tensor(key)
                expected = source.to(torch.bfloat16) if source.is_floating_point() else source
                self.assertTrue(torch.equal(tensors[key], expected), key)

    def test_legacy_module_selection_is_unchanged(self):
        model = nn.Sequential(nn.Linear(128, 64), nn.Linear(128, 64))

        class QuantLinear(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()

        with patch("eole.modules.autoround_linear._get_autoround_quant_linear_cls", return_value=(QuantLinear, False)):
            replace_autoround_linear(model, ["0"])
        self.assertIsInstance(model[0], QuantLinear)
        self.assertIsInstance(model[1], nn.Linear)


if __name__ == "__main__":
    unittest.main()
