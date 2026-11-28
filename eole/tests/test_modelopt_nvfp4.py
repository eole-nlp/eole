"""ModelOpt NVFP4 scale conventions, provenance and compressed vocabulary heads."""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch
from safetensors.torch import load_file, save_file

from eole.bin.convert.compressed_tensors import ModelOptCheckpoint, validate_modelopt_config
from eole.bin.convert.convert_HF import HuggingfaceFiles, build_config_dict, build_shards, check_conversion_equality
from eole.modules.quant.compressed_float_linear import CompressedFloatLinear


def quant_config():
    spec = dict(dynamic=False, num_bits=4, type="float", group_size=16)
    return dict(
        quant_method="modelopt",
        quant_algo="NVFP4",
        config_groups={"group_0": dict(targets=["Linear"], weights=spec.copy(), input_activations=spec.copy())},
    )


def projection(prefix):
    return {
        prefix + ".weight": torch.randint(256, (128, 128), dtype=torch.uint8),
        prefix + ".weight_scale": torch.ones(128, 16).to(torch.float8_e4m3fn),
        prefix + ".weight_scale_2": torch.tensor(0.25),
        prefix + ".input_scale": torch.tensor(0.5),
    }


class TestModelOptNVFP4(unittest.TestCase):
    def test_aliases_scales_and_missing_metadata(self):
        tensors = projection("p")
        view = ModelOptCheckpoint(tensors, tensors.get)
        self.assertIs(view["p.qweight"], tensors["p.weight"])
        self.assertIs(view["p.scales"], tensors["p.weight_scale"])
        self.assertEqual(view["p.weight_global_scale"].item(), 4)
        self.assertEqual(view["p.input_global_scale"].item(), 2)
        self.assertEqual(view.source_keys("p.qweight"), {"p.weight"})
        self.assertEqual(view.source_keys("p.weight_global_scale"), {"p.weight_scale_2"})
        for suffix in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
            incomplete = {k: v for k, v in tensors.items() if k != "p." + suffix}
            with self.assertRaisesRegex(ValueError, "Incomplete ModelOpt"):
                ModelOptCheckpoint(incomplete, incomplete.get)
        raw = {"p.weight": tensors["p.weight"]}
        with self.assertRaisesRegex(ValueError, "lacks NVFP4 scale metadata"):
            ModelOptCheckpoint(raw, raw.get)["p.weight"]
        tensors["p.input_scale"] = torch.tensor(0.0)
        with self.assertRaisesRegex(ValueError, "positive finite"):
            ModelOptCheckpoint(tensors, tensors.get)["p.input_global_scale"]

    def test_rejects_other_modelopt_schemes(self):
        config = quant_config()
        self.assertEqual(validate_modelopt_config(config), 16)
        config["quant_algo"] = "FP8"
        with self.assertRaisesRegex(ValueError, "NVFP4"):
            validate_modelopt_config(config)
        config = quant_config()
        config["config_groups"]["group_0"]["weights"]["group_size"] = 32
        with self.assertRaisesRegex(ValueError, "group-16"):
            validate_modelopt_config(config)

    def test_generator_is_replaced_without_quantizing_other_modules(self):
        from eole.models.model import BaseModel

        model = torch.nn.Module()
        model.generator = torch.nn.Linear(256, 128, bias=False)
        model.decoder = torch.nn.Linear(256, 256, bias=False)
        model.share_decoder_embeddings = False
        BaseModel.maybe_quantize(
            model,
            SimpleNamespace(
                quant_layers=["generator"],
                quant_type="nvfp4",
                compute_dtype=torch.bfloat16,
            ),
        )
        self.assertIsInstance(model.generator, CompressedFloatLinear)
        self.assertIsInstance(model.decoder, torch.nn.Linear)

    def test_conversion_with_quantized_head_and_separate_metadata_shard(self):
        self._check_conversion(appended_mtp=False)

    def test_appended_mtp_has_no_decoder_duplicate_or_negative_indices(self):
        self._check_conversion(appended_mtp=True)

    def test_single_file_checkpoint_keeps_packed_head(self):
        self._check_conversion(appended_mtp=False, indexed=False)

    def _check_conversion(self, appended_mtp, indexed=True):
        from eole.models.hf_loader import HFLoader, _InMemoryTensorStore

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "quantization_config": quant_config(),
                "vision_config": {"depth": 1},
                "text_config": dict(
                    hidden_size=256,
                    num_hidden_layers=1,
                    num_attention_heads=4,
                    head_dim=64,
                    mtp_num_hidden_layers=int(appended_mtp),
                ),
            }
            (root / "config.json").write_text(json.dumps(config))
            tensors = {
                **projection("model.language_model.layers.0.linear_attn.in_proj_qkv"),
                **projection("model.language_model.layers.0.mlp.gate_proj"),
                **projection("lm_head"),
            }
            if appended_mtp:
                tensors["model.language_model.layers.1.self_attn.q_proj.weight"] = torch.randn(128, 256)
            tensors["model.language_model.embed_tokens.weight"] = torch.randn(128, 256).bfloat16()
            first = {k: v for k, v in tensors.items() if k.endswith(".weight")}
            second = {k: v for k, v in tensors.items() if k not in first}
            weight_map = {}
            if indexed:
                for filename, data in (("model-1.safetensors", first), ("model-2.safetensors", second)):
                    save_file(data, str(root / filename))
                    weight_map.update({k: filename for k in data})
                (root / "model.safetensors.index.json").write_text(json.dumps(dict(weight_map=weight_map)))
            else:
                save_file(tensors, str(root / "model.safetensors"))
            hf = HuggingfaceFiles(
                token=None,
                tokenizer_config_json=None,
                config_path=str(root / "config.json"),
                model_dir=str(root),
                wmap_path=str(root / "model.safetensors.index.json") if indexed else None,
                model_path=None if indexed else str(root / "model.safetensors"),
            )
            model_config, training, params = build_config_dict(hf)
            self.assertEqual(training["quant_type"], "nvfp4")
            self.assertIn("generator", training["quant_layers"])
            out = root / "out"
            out.mkdir()
            all_keys, consumed, details = build_shards(
                model_config,
                hf,
                SimpleNamespace(nshards=1, output=str(out), dtype="bf16"),
                params,
            )
            self.assertEqual(all_keys, consumed)
            self.assertEqual(check_conversion_equality(hf, details, torch.bfloat16), [])
            converted = load_file(str(out / "model.00.safetensors"))
            if appended_mtp:
                self.assertIn("mtp_heads.0.layer.self_attn.linear_query.weight", converted)
                self.assertFalse(any(k.startswith("decoder.transformer_layers.1.") for k in converted))
            else:
                self.assertFalse(any(k.startswith("mtp_heads.") for k in converted))
            self.assertFalse(any(k.startswith("mtp_heads.-") for k in converted))
            self.assertNotIn("generator.weight", converted)
            torch.testing.assert_close(converted["generator.qweight"], tensors["lm_head.weight"], rtol=0, atol=0)
            self.assertEqual(converted["generator.scales"].dtype, torch.float8_e4m3fn)
            self.assertEqual(converted["generator.weight_global_scale"].item(), 4)
            self.assertEqual(converted["generator.input_global_scale"].item(), 2)
            direct = _InMemoryTensorStore()
            HFLoader(str(root))._build_tensors(hf, model_config, params, direct)
            self.assertEqual(set(direct.keys()), converted.keys())
            for key in ("generator.qweight", "generator.scales", "generator.input_global_scale"):
                value = direct.get_tensor(key)
                self.assertEqual(value.dtype, converted[key].dtype)
                self.assertTrue(
                    torch.equal(value.reshape(-1).view(torch.uint8), converted[key].reshape(-1).view(torch.uint8))
                )
            if not indexed:
                hf.config["text_config"]["tie_word_embeddings"] = True
                with self.assertRaisesRegex(ValueError, "cannot share decoder embeddings"):
                    build_config_dict(hf)


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0), "SM120 required")
class TestModelOptGPU(unittest.TestCase):
    def test_modelopt_scale_convention_matches_dequantized_reference(self):
        from flashinfer.fp4_quantization import fp4_quantize, e2m1_and_ufp8sf_scale_to_float

        torch.manual_seed(17)
        w = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        weight_scale_2 = w.float().abs().max() / 2688
        input_scale = torch.tensor(0.0125, device="cuda")
        packed, _ = fp4_quantize(w, weight_scale_2.reciprocal(), nvfp4_4over6=None)
        scales = (w.float().reshape(128, 16, 16).abs().amax(-1) / (6 * weight_scale_2)).to(torch.float8_e4m3fn)
        tensors = {
            "p.weight": packed,
            "p.weight_scale": scales,
            "p.weight_scale_2": weight_scale_2,
            "p.input_scale": input_scale,
        }
        view = ModelOptCheckpoint(tensors, tensors.get)
        layer = CompressedFloatLinear(256, 128).cuda()
        layer.qweight = view["p.qweight"]
        layer.scales = view["p.scales"]
        layer.weight_global_scale = view["p.weight_global_scale"]
        layer.input_global_scale = view["p.input_global_scale"]
        layer.post_init()
        x = torch.randn(3, 256, device="cuda", dtype=torch.bfloat16)
        qx, sx = fp4_quantize(x, input_scale.reciprocal(), nvfp4_4over6=None)
        xd = e2m1_and_ufp8sf_scale_to_float(qx, sx, input_scale).float().cuda()
        lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], device="cuda")
        codes = torch.stack((packed & 15, packed >> 4), -1).reshape(128, 256).long()
        wd = lut[codes] * scales.float().repeat_interleave(16, -1) * weight_scale_2
        torch.testing.assert_close(layer(x).float(), xd @ wd.t(), rtol=0.02, atol=0.25)
