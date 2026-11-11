"""Mixed NVFP4/FP8 checkpoint bytes, dtype preservation and native inference."""

import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from safetensors.torch import save_file, load_file
import torch

from eole.bin.convert.compressed_tensors import FloatCheckpoint
from eole.modules.compressed_float_linear import CompressedFloatLinear


class TestCompressedFloat(unittest.TestCase):
    def test_aliases_preserve_codes_and_native_weights(self):
        tensors = {
            "a.weight_packed": torch.tensor([[0x81, 0x7F]], dtype=torch.uint8),
            "a.weight_scale": torch.ones(1, 1).to(torch.float8_e4m3fn),
            "a.weight_global_scale": torch.tensor([2.0]),
            "b.weight": torch.ones(2, 16).to(torch.float8_e4m3fn),
            "b.weight_scale": torch.tensor([[1.0], [2.0]]),
            "native.weight": torch.ones(2, 16, dtype=torch.bfloat16),
        }
        view = FloatCheckpoint(tensors, tensors.get)
        self.assertNotIn("b.weight", view)
        for virtual, source in (
            ("a.qweight", "a.weight_packed"),
            ("a.scales", "a.weight_scale"),
            ("b.qweight", "b.weight"),
            ("native.weight", "native.weight"),
        ):
            self.assertIs(view[virtual], tensors[source])
            self.assertEqual(view.source_keys(virtual), {source})

    def test_direct_gptq_key_keeps_indexed_checkpoint(self):
        from eole.bin.convert.convert_HF import HuggingfaceFiles, get_weight

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "gptq"}}))
            tensor = torch.arange(8, dtype=torch.int32)
            save_file({"layer.qweight": tensor}, str(root / "model-1.safetensors"))
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"layer.qweight": "model-1.safetensors"}})
            )
            hf = HuggingfaceFiles(
                token=None,
                tokenizer_config_json=None,
                config_path=str(root / "config.json"),
                model_dir=str(root),
                wmap_path=str(root / "model.safetensors.index.json"),
            )
            checkpoint = hf.checkpoint_for_tensor("layer.qweight")
            torch.testing.assert_close(get_weight(checkpoint, "layer.qweight"), tensor)

    def test_mixed_conversion_with_separate_metadata_shard(self):
        from eole.bin.convert.convert_HF import (
            HuggingfaceFiles,
            build_config_dict,
            build_shards,
            check_conversion_equality,
        )
        from eole.bin.convert.compressed_tensors import validate_float_config

        quant = {
            "quant_method": "compressed-tensors",
            "format": "mixed-precision",
            "config_groups": {
                "fp4": {
                    "format": "nvfp4-pack-quantized",
                    "targets": ["Linear"],
                    "weights": {
                        "type": "float",
                        "num_bits": 4,
                        "symmetric": True,
                        "strategy": "tensor_group",
                        "group_size": 16,
                        "scale_dtype": "torch.float8_e4m3fn",
                    },
                    "input_activations": {
                        "type": "float",
                        "num_bits": 4,
                        "symmetric": True,
                        "strategy": "tensor_group",
                        "group_size": 16,
                        "dynamic": "local",
                    },
                },
                "fp8": {
                    "format": "float-quantized",
                    "targets": ["Linear"],
                    "weights": {"type": "float", "num_bits": 8, "symmetric": True, "strategy": "channel"},
                    "input_activations": {
                        "type": "float",
                        "num_bits": 8,
                        "symmetric": True,
                        "strategy": "token",
                        "dynamic": True,
                    },
                },
            },
        }
        validate_float_config(quant)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "quantization_config": quant,
                "text_config": {
                    "hidden_size": 256,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 4,
                    "head_dim": 64,
                    "mtp_num_hidden_layers": 1,
                },
            }
            (root / "config.json").write_text(json.dumps(config))
            fp4 = "model.language_model.layers.0.mlp.gate_proj"
            fp8 = "model.language_model.layers.0.linear_attn.in_proj_qkv"
            first = {
                fp4 + ".weight_packed": torch.randint(256, (128, 128), dtype=torch.uint8),
                fp8 + ".weight": torch.randn(128, 256).to(torch.float8_e4m3fn),
                "mtp.layers.0.self_attn.q_proj.weight": torch.randn(128, 256),
            }
            second = {
                fp4 + ".weight_scale": torch.ones(128, 16).to(torch.float8_e4m3fn),
                fp4 + ".weight_global_scale": torch.tensor(5504.0, dtype=torch.bfloat16),
                fp4 + ".input_global_scale": torch.tensor([660.0]),
                fp8 + ".weight_scale": torch.rand(128, 1),
            }
            weight_map = {}
            for name, tensors in (("model-1.safetensors", first), ("model-2.safetensors", second)):
                save_file(tensors, str(root / name))
                weight_map.update({key: name for key in tensors})
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
            hf = HuggingfaceFiles(
                token=None,
                tokenizer_config_json=None,
                config_path=str(root / "config.json"),
                model_dir=str(root),
                wmap_path=str(root / "model.safetensors.index.json"),
            )
            model_config, training, params = build_config_dict(hf)
            self.assertEqual(training["quant_type"], "nvfp4")
            self.assertEqual(len(training["quant_layers"]), 2)
            out = root / "out"
            out.mkdir()
            all_keys, consumed, details = build_shards(
                model_config, hf, SimpleNamespace(nshards=2, output=str(out), dtype="bf16"), params
            )
            self.assertEqual(all_keys, consumed)
            self.assertEqual(check_conversion_equality(hf, details, torch.bfloat16), [])
            converted = load_file(str(out / "model.00.safetensors"))
            converted.update(load_file(str(out / "model.01.safetensors")))
            self.assertIn("mtp_heads.0.layer.self_attn.linear_query.weight", converted)
            self.assertEqual(
                converted["decoder.transformer_layers.0.linear_attn.in_proj_qkv.qweight"].dtype, torch.float8_e4m3fn
            )
            scale = converted["decoder.transformer_layers.0.linear_attn.in_proj_qkv.scales"]
            torch.testing.assert_close(scale, second[fp8 + ".weight_scale"], rtol=0, atol=0)

    def test_dtype_conversion_preserves_quantization(self):
        layer = CompressedFloatLinear(16, 2, bias=True)
        layer.qweight = torch.ones(2, 16).to(torch.float8_e4m3fn)
        layer.scales = torch.tensor([[1.25], [2.25]], dtype=torch.float32)
        layer.bfloat16()
        self.assertEqual(layer.qweight.dtype, torch.float8_e4m3fn)
        self.assertEqual(layer.scales.dtype, torch.float32)
        self.assertEqual(layer.bias.dtype, torch.bfloat16)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0), "SM120 required")
    def test_fp8_fusion_and_compilation(self):
        torch.manual_seed(2)
        layers = []
        x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
        for _ in range(3):
            layer = CompressedFloatLinear(256, 128).cuda()
            weight = torch.randn(128, 256, device="cuda")
            layer.scales = weight.abs().amax(1, keepdim=True).clamp_min(1e-12) / 448
            layer.qweight = (weight / layer.scales).to(torch.float8_e4m3fn)
            layer.post_init()
            layers.append(layer)
        layers[0].bias = torch.nn.Parameter(torch.randn(128, device="cuda", dtype=torch.bfloat16), requires_grad=False)
        expected = torch.cat([layer(x) for layer in layers], dim=-1)
        fused = CompressedFloatLinear.fuse(layers)
        torch.testing.assert_close(fused(x), expected, rtol=0, atol=0)
        compiled = torch.compile(fused, fullgraph=True)
        torch.testing.assert_close(compiled(x), expected, rtol=0.01, atol=0.125)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0), "SM120 required")
    def test_nvfp4_reference_and_graph(self):
        from flashinfer.fp4_quantization import fp4_quantize, e2m1_and_ufp8sf_scale_to_float

        torch.manual_seed(1)
        layer = CompressedFloatLinear(256, 128).cuda()
        w = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        layer.weight_global_scale = (2688 / w.float().abs().max()).reshape(1)
        layer.input_global_scale = torch.tensor([512.0], device="cuda")
        packed, _ = fp4_quantize(w, layer.weight_global_scale, nvfp4_4over6=None)
        # Unswizzled checkpoint scales, independently reconstructed from groups.
        blocks = w.float().reshape(128, 16, 16)
        layer.scales = (blocks.abs().amax(-1) * layer.weight_global_scale / 6).to(torch.float8_e4m3fn)
        layer.qweight = packed
        layer.post_init()
        x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
        a, sa = fp4_quantize(x, layer.input_global_scale, nvfp4_4over6=None)
        xd = e2m1_and_ufp8sf_scale_to_float(a, sa, 1 / layer.input_global_scale).float().cuda()
        codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(128, 256).long()
        lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], device="cuda")
        wd = lut[codes] * layer.scales.float().repeat_interleave(16, dim=-1) / layer.weight_global_scale
        ref = xd @ wd.t()
        torch.testing.assert_close(layer(x).float(), ref, rtol=0.02, atol=0.25)
        import copy

        other = copy.deepcopy(layer)
        other.qweight = other.qweight.roll(1, dims=0)
        other.scales = other.scales.view(torch.uint8).roll(1, dims=0).view(torch.float8_e4m3fn)
        other.post_init()
        separate = torch.cat([layer(x), other(x)], dim=-1)
        fused = CompressedFloatLinear.fuse([layer, other])
        torch.testing.assert_close(fused(x), separate, rtol=0.01, atol=0.125)
        for name in ("input_global_scale", "weight_global_scale"):
            original = getattr(other, name).clone()
            getattr(other, name).mul_(2)
            self.assertIsNone(CompressedFloatLinear.fuse([layer, other]))
            getattr(other, name).copy_(original)
        compiled = torch.compile(layer, fullgraph=True)
        expected = compiled(x).clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = compiled(x)
        graph.replay()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(layer.half()(x.half()).float(), ref, rtol=0.02, atol=0.25)
