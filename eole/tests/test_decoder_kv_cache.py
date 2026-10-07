"""Decoder KV caches must accommodate the heads projected on each rank."""

import unittest

import torch

from eole.config.inference import InferenceConfig
from eole.config.models import TransformerDecoderConfig
from eole.decoders.transformer import TransformerDecoder


class TestDecoderKVCache(unittest.TestCase):
    def test_cache_accepts_local_prefill_and_decode_keys(self):
        for heads_kv in (None, 2):
            for parallel_mode, world_size in (("data_parallel", 2), ("tensor_parallel", 1), ("tensor_parallel", 2)):
                for dynamic_shapes in (True, False):
                    with self.subTest(heads_kv=heads_kv, mode=parallel_mode, ranks=world_size, dynamic=dynamic_shapes):
                        config = TransformerDecoderConfig(
                            layers=1,
                            hidden_size=64,
                            heads=4,
                            heads_kv=heads_kv,
                            transformer_ff=128,
                            with_cross_attn=False,
                            max_position_embeddings=8,
                        )
                        running = InferenceConfig(
                            parallel_mode=parallel_mode,
                            world_size=world_size,
                            self_attn_backend="pytorch",
                            dynamic_shapes=dynamic_shapes,
                        )
                        decoder = TransformerDecoder(config, running_config=running).eval()
                        # These tests construct modules without loading a checkpoint.
                        with torch.no_grad():
                            for parameter in decoder.parameters():
                                parameter.fill_(0.01)
                        inputs = torch.randn(2, 3, 64)
                        decoder._init_cache(inputs, torch.zeros(2, 1, 3, dtype=torch.bool))
                        attention = decoder.transformer_layers[0].self_attn
                        key, value, query = attention._prepare_inputs(inputs, inputs, inputs)
                        local_heads = (heads_kv or 4) // running.parallel_gpu
                        capacity = 3 if dynamic_shapes else 8
                        self.assertEqual(attention.kcache.shape, (2, capacity, local_heads, 16))
                        self.assertEqual(attention.vcache.shape, attention.kcache.shape)
                        attention._update_cache_w_inputs(query, key, value, decoder.cache_seqlens, torch.arange(3))
                        torch.testing.assert_close(attention.kcache[:, :3], key)
                        torch.testing.assert_close(attention.vcache[:, :3], value)

                        decoder.cache_seqlens.fill_(3)
                        decoder._extend_cache(addzeros=2)
                        self.assertEqual(attention.kcache.shape[2], local_heads)
                        # Extension must preserve the populated prefix.
                        torch.testing.assert_close(attention.kcache[:, :3], key)
                        token = torch.randn(2, 1, 64)
                        next_key, next_value, next_query = attention._prepare_inputs(token, token, token)
                        attention._update_cache_w_inputs(
                            next_query, next_key, next_value, decoder.cache_seqlens, torch.tensor([3])
                        )
                        torch.testing.assert_close(attention.kcache[:, 3:4], next_key)
                        torch.testing.assert_close(attention.vcache[:, 3:4], next_value)
