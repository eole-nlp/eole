"""Qwen GGUF decoder/MTP separation and split-half RoPE regressions."""

import tempfile
import unittest
from unittest.mock import patch, PropertyMock
from types import SimpleNamespace

import numpy as np
import torch

from eole.bin.convert.convert_gguf import (
    GGUFMetadata,
    build_model_config,
    build_safetensors,
    _gguf_to_eole_name,
    _packed_module_paths,
)


def metadata(mtp=1):
    meta = GGUFMetadata.__new__(GGUFMetadata)
    values = {
        "qwen35.block_count": 64 + mtp,
        "qwen35.nextn_predict_layers": mtp,
        "qwen35.embedding_length": 5120,
        "qwen35.attention.head_count": 24,
        "qwen35.attention.head_count_kv": 4,
        "qwen35.attention.key_length": 256,
        "qwen35.feed_forward_length": 17408,
        "qwen35.rope.freq_base": 10000000,
        "qwen35.rope.dimension_count": 64,
        "qwen35.ssm.state_size": 128,
        "qwen35.ssm.group_count": 16,
        "qwen35.ssm.time_step_rank": 48,
        "qwen35.ssm.inner_size": 6144,
        "qwen35.ssm.conv_kernel": 4,
    }
    meta._scalar = lambda key, default=None: values.get(key, default)
    meta._str = lambda key, default=None: "qwen35" if key == "general.architecture" else default
    meta._get_field = lambda key: None
    meta._reader = SimpleNamespace(tensors=[])
    return meta


class TestGGUFMTP(unittest.TestCase):
    def test_decoder_depth_and_rope_with_and_without_mtp(self):
        for count in (0, 1, 2):
            with self.subTest(count=count):
                meta = metadata(count)
                cfg = build_model_config(meta, frozenset(i for i in range(64) if i % 4 != 3))
                self.assertEqual(cfg["layers"], 64)
                self.assertEqual(len(cfg["decoder"]["layer_types"]), 64)
                self.assertEqual(cfg["decoder"].get("num_mtp_heads", 0), count)
                self.assertEqual(cfg["rope_config"]["rotary_dim"], 64)
                self.assertEqual(cfg["rope_config"]["rotary_theta"], 10000000)
                self.assertFalse(cfg["rope_config"]["rotary_interleave"])
                if count:
                    self.assertTrue(cfg["decoder"]["mtp_emb_norm"])

    def test_mrope_sections_preserved_with_split_half_pairing(self):
        with patch.object(GGUFMetadata, "rope_dim_sections", new_callable=PropertyMock, return_value=[11, 11, 10]):
            cfg = build_model_config(metadata())
        self.assertEqual(cfg["rope_config"]["xdrope_section"], [11, 11, 10])
        self.assertFalse(cfg["rope_config"]["rotary_interleave"])

    def test_tensor_mapping(self):
        mapping = {
            "nextn.eh_proj.weight": "proj.weight",
            "nextn.enorm.weight": "emb_norm.weight",
            "nextn.hnorm.weight": "enorm.weight",
            "nextn.shared_head_norm.weight": "norm.weight",
            "attn_q.weight": "layer.self_attn.linear_query.weight",
            "ffn_gate.weight": "layer.mlp.gate_up_proj.weight",
        }
        for suffix, target in mapping.items():
            for block, head in ((64, 0), (65, 1)):
                self.assertEqual(
                    _gguf_to_eole_name(f"blk.{block}.{suffix}", decoder_layers=64), f"mtp_heads.{head}.{target}"
                )
        self.assertEqual(
            _gguf_to_eole_name("blk.63.attn_q.weight", decoder_layers=64),
            "decoder.transformer_layers.63.self_attn.linear_query.weight",
        )

    def test_invalid_mtp_count(self):
        meta = metadata(-1)
        with self.assertRaisesRegex(ValueError, "Invalid"):
            _ = meta.decoder_block_count

    def test_incomplete_head_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Incomplete GGUF MTP"):
                build_safetensors(metadata(), directory, torch.bfloat16)

    def test_mtp_tensors_written_under_head_paths(self):
        suffixes = [
            "nextn.eh_proj.weight",
            "nextn.enorm.weight",
            "nextn.hnorm.weight",
            "nextn.shared_head_norm.weight",
            "attn_norm.weight",
            "post_attention_norm.weight",
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "attn_q_norm.weight",
            "attn_k_norm.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
        ]
        meta = metadata()
        meta._reader.tensors = [
            SimpleNamespace(
                name="blk.64." + suffix,
                tensor_type=SimpleNamespace(name="F32", value=0),
                data=np.ones((2, 2), dtype=np.float32),
                shape=(2, 2),
            )
            for suffix in suffixes
        ]
        with tempfile.TemporaryDirectory() as directory:
            written, _ = build_safetensors(meta, directory, torch.bfloat16)
        self.assertIn("mtp_heads.0.proj.weight", written)
        self.assertTrue(all(key.startswith("mtp_heads.0.") for key in written))
        self.assertEqual(written["mtp_heads.0.emb_norm.weight"].dtype, torch.bfloat16)
        self.assertTrue(torch.equal(written["mtp_heads.0.emb_norm.weight"], torch.zeros(2, 2, dtype=torch.bfloat16)))

    def test_packed_mtp_projection_and_float_controls_selection(self):
        from eole.modules.gguf_linear import GGUFLinear, replace_gguf_linear
        from torch import nn

        model = nn.Module()
        model.decoder = nn.Module()
        model.decoder.in_proj_a = nn.Linear(8, 8)
        model.mtp_heads = nn.ModuleList([nn.Module()])
        model.mtp_heads[0].proj = nn.Linear(16, 8)
        written = {
            "mtp_heads.0.proj.gguf_qtype": torch.tensor([14]),
            "mtp_heads.0.proj.weight": torch.zeros(8, 16),
            "decoder.in_proj_a.weight": torch.zeros(8, 8),
        }
        paths = _packed_module_paths(written)
        self.assertEqual(paths, ["mtp_heads.0.proj"])
        replace_gguf_linear(model, paths)
        self.assertIsInstance(model.mtp_heads[0].proj, GGUFLinear)
        self.assertIsInstance(model.decoder.in_proj_a, nn.Linear)
