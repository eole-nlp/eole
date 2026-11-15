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
    def test_old_extension_without_fp8_image_check_is_not_used(self):
        from unittest.mock import patch
        from eole.ops import fp8_mma_supported

        with patch("eole.ops._CPP_OPS_AVAILABLE", True):
            with patch("eole.ops._ops", SimpleNamespace(), create=True):
                self.assertFalse(fp8_mma_supported(torch.device("cuda:0")))

    def test_marlin_is_opt_in_and_requires_matching_extension(self):
        from unittest.mock import patch
        from eole.config.inference import InferenceConfig

        self.assertEqual(InferenceConfig().fp8_backend, "native")
        layer = CompressedFloatLinear(128, 64)
        with patch("eole._ops", SimpleNamespace(fp8_marlin_supported=False), create=True):
            with self.assertRaisesRegex(ValueError, "rebuilt Eole CUDA"):
                layer.prepare_marlin()

    def test_marlin_warns_when_no_fp8_projections(self):
        from unittest.mock import patch
        from eole.predict.inference import Inference
        from eole.utils.logging import logger

        target = SimpleNamespace(model=torch.nn.Module())
        with patch("eole.modules.hybrid_experts.prepare_hybrid_experts"):
            with self.assertLogs(logger, level="WARNING") as logged:
                Inference._prepare_inference_projections(target, SimpleNamespace(fp8_backend="marlin"))
        self.assertIn("no FP8 projections", logged.output[0])

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


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0), "SM120 required")
class TestSmallRowFP8(unittest.TestCase):
    def test_reference_tail_rows_and_graph(self):
        from eole.ops import fp8_mm, _DYNAMIC_QUANT_OPS_AVAILABLE

        self.assertTrue(_DYNAMIC_QUANT_OPS_AVAILABLE, "Rebuild Eole CUDA kernels")
        torch.manual_seed(13)
        for dtype in (torch.bfloat16, torch.float16):
            weight = torch.randn(128, 512, device="cuda").to(torch.float8_e4m3fn)
            scales_w = torch.rand(128, 1, device="cuda") + 0.1
            for rows in (1, 3, 8, 16):
                x = torch.randn(rows, 512, device="cuda").to(torch.float8_e4m3fn)
                scales_x = torch.rand(rows, 1, device="cuda") + 0.1
                expected = torch._scaled_mm(x, weight.t(), scale_a=scales_x, scale_b=scales_w.t(), out_dtype=dtype)
                fn = torch.compile(
                    lambda a: fp8_mm(a, weight, scales_x, scales_w, dtype == torch.bfloat16), fullgraph=True
                )
                actual = fn(x)
                torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.125)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = fn(x)
                graph.replay()
                torch.testing.assert_close(output, actual, rtol=0, atol=0)
        torch._dynamo.reset()

    def test_unsupported_compiled_image_uses_scaled_mm(self):
        from unittest.mock import patch

        layer = CompressedFloatLinear(512, 128).cuda()
        layer.qweight = torch.randn(128, 512, device="cuda").to(torch.float8_e4m3fn)
        layer.scales = torch.ones(128, 1, device="cuda")
        with patch("eole.modules.compressed_float_linear.fp8_mma_supported", return_value=False):
            layer.post_init()
        self.assertFalse(layer.use_cuda_fp8)
        x = torch.randn(3, 512, device="cuda", dtype=torch.bfloat16)
        scale = x.float().abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448
        q = (x.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        expected = torch._scaled_mm(q, layer.qweight.t(), scale_a=scale, scale_b=layer.scales.t(), out_dtype=x.dtype)
        torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)

    def test_unsupported_width_uses_scaled_mm(self):
        layer = CompressedFloatLinear(256, 128).cuda()
        layer.qweight = torch.ones(128, 256, device="cuda").to(torch.float8_e4m3fn)
        layer.scales = torch.ones(128, 1, device="cuda")
        layer.post_init()
        self.assertFalse(layer.use_cuda_fp8)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
    "SM120 required",
)
class TestMarlinFP8(unittest.TestCase):
    def test_weight_only_reference_layout_and_capture(self):
        torch.manual_seed(29)
        layer = CompressedFloatLinear(512, 128).cuda()
        layer.qweight = torch.randn(128, 512, device="cuda").to(torch.float8_e4m3fn)
        layer.scales = torch.rand(128, 1, device="cuda") * 0.01 + 0.001
        layer.post_init()
        weight = layer.qweight.float().clone()
        scales = layer.scales.clone()
        layer.prepare_marlin()
        self.assertEqual(layer.qweight.numel(), 0)
        self.assertEqual(layer.marlin_qweight.numel() * 4, weight.numel())
        layer.prepare_marlin()  # Idempotent setup.
        with self.assertRaisesRegex(ValueError, "fusion must precede"):
            CompressedFloatLinear.fuse([layer, layer])
        for dtype in (torch.bfloat16, torch.float16):
            layer.to(dtype=dtype)
            for rows in (1, 3, 8, 16, 33, 128):
                x = torch.randn(rows, 512, device="cuda", dtype=dtype)
                expected = x.float() @ weight.t() * scales.t()
                actual = layer(x)
                rms = ((actual.float() - expected).square().mean() / expected.square().mean()).sqrt()
                self.assertLess(rms.item(), 0.007)
            compiled = torch.compile(layer, fullgraph=True)
            torch.testing.assert_close(compiled(x), layer(x), rtol=0, atol=0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = compiled(x)
            x.normal_()
            graph.replay()
            torch.testing.assert_close(output, layer(x), rtol=0, atol=0)
        torch._dynamo.reset()

    def test_all_finite_fp8_codes(self):
        codes = torch.arange(256, device="cuda", dtype=torch.int16)
        codes = torch.where((codes & 127) == 127, 0, codes).to(torch.uint8)
        weight = codes[:, None].expand(256, 128).contiguous().view(torch.float8_e4m3fn)
        layer = CompressedFloatLinear(128, 256).cuda()
        layer.qweight = weight
        layer.scales = torch.full((256, 1), 0.01, device="cuda")
        layer.post_init()
        layer.prepare_marlin()
        for dtype in (torch.bfloat16, torch.float16):
            x = torch.ones(3, 128, device="cuda", dtype=dtype)
            reference = (x.float() @ weight.float().t()) * 0.01
            torch.testing.assert_close(layer(x).float(), reference, rtol=0.01, atol=1e-5)

    def test_fp16_small_channel_scales(self):
        from eole.modules.marlin_utils import marlin_permute_scales

        layer = CompressedFloatLinear(128, 128).cuda()
        layer.qweight = torch.ones(128, 128, device="cuda").to(torch.float8_e4m3fn)
        scales = torch.tensor([1e-8, 1e-7, 1e-6, 1e-5, 3e-5, 6e-5, 1e-4, 1e-3], device="cuda").repeat(16)
        layer.scales = scales[:, None]
        layer.post_init()
        layer.prepare_marlin()
        expected_scales = marlin_permute_scales((scales * 256).half().reshape(1, -1), 128, 128, -1)
        torch.testing.assert_close(layer.marlin_scales_fp16, expected_scales, rtol=0, atol=0)
        x = torch.ones(3, 128, device="cuda", dtype=torch.float16)
        torch.testing.assert_close(layer(x).float(), (128 * scales).expand(3, -1), rtol=0.003, atol=6e-8)

    def test_marlin_rejects_pre_ampere_compiled_image(self):
        from unittest.mock import patch

        layer = CompressedFloatLinear(128, 128).cuda()
        layer.qweight = torch.ones(128, 128, device="cuda").to(torch.float8_e4m3fn)
        layer.scales = torch.ones(128, 1, device="cuda")
        layer.post_init()
        with patch("eole._ops.fp8_compiled_arch", return_value=750):
            with self.assertRaisesRegex(ValueError, r"SM80\+ compiled CUDA image"):
                layer.prepare_marlin()
