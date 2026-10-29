"""Safe request-wide FlashAttention views and changing-bound graph capture."""

import unittest
from unittest.mock import patch

import torch

from eole.config.inference import InferenceConfig
from eole.config.models import TransformerDecoderConfig
from eole.decoders.transformer import TransformerDecoder
from eole.ops import _FLASH_ATTN_AVAILABLE
from eole.ops.kv_cache import bounded_kv_cache


def decoder(**options):
    return TransformerDecoder(
        TransformerDecoderConfig(
            layers=1,
            hidden_size=64,
            heads=4,
            heads_kv=2,
            transformer_ff=128,
            with_cross_attn=False,
            max_position_embeddings=16384,
        ),
        running_config=InferenceConfig(context_length=16384, **options),
    ).eval()


class TestKVCacheView(unittest.TestCase):
    def test_prefix_keeps_storage_and_writes(self):
        k = torch.zeros(2, 64, 2, 16)
        v = torch.zeros_like(k)
        small_k, small_v = bounded_kv_cache(k, v, torch.tensor(32))
        self.assertEqual(small_k.shape, (2, 32, 2, 16))
        self.assertEqual(small_k.untyped_storage().data_ptr(), k.untyped_storage().data_ptr())
        self.assertEqual(small_k.stride(), k.stride())
        small_k[:, 31] = 7
        small_v[:, 31] = 9
        self.assertTrue(torch.all(k[:, 31] == 7))
        self.assertTrue(torch.all(v[:, 31] == 9))
        for bound, table in [(None, None), (torch.tensor(32), torch.zeros(1))]:
            a, b = bounded_kv_cache(k, v, bound, table)
            self.assertIs(a, k)
            self.assertIs(b, v)
        self.assertEqual(bounded_kv_cache(k, v, torch.tensor(100))[0].shape, k.shape)
        for bound in [torch.tensor(0), torch.tensor(-1), torch.tensor([1, 2]), torch.tensor(1.0)]:
            with self.assertRaises(ValueError):
                bounded_kv_cache(k, v, bound)

    def test_request_bound_resets_graph_and_clamps(self):
        model = decoder(dynamic_shapes=False)
        identity = model._kv_cache_view_bound
        for tokens, expected in [(1, 512), (513, 1024), (5145 + 512 + 4, 8192), (20000, 16384), (None, 16384)]:
            with self.subTest(tokens=tokens), patch.object(model, "_reset_decode_graph") as reset:
                model.set_kv_cache_view_bound(tokens)
                reset.assert_called_once()
                self.assertIs(model._kv_cache_view_bound, identity)
                self.assertEqual(identity.item(), expected)
        model.dynamic_shapes = True
        model.set_kv_cache_view_bound(100)
        self.assertEqual(identity.item(), 16384)
        model.dynamic_shapes = False
        model.with_cross_attn = True
        model.set_kv_cache_view_bound(100)
        self.assertEqual(identity.item(), 16384)

    def test_legacy_or_unsupported_capture_does_not_receive_cpu_input(self):
        model = decoder(dynamic_shapes=False)
        inputs = torch.zeros(1, 2, 64)
        mask = torch.zeros(1, 1, 2, dtype=torch.bool)
        with patch("eole.decoders.transformer.EOLE_TORCH_COMPILE", True):
            for enabled, request, dynamic, cross in (
                (True, True, False, False),
                (False, True, False, False),
                (True, False, False, False),
                (True, True, True, False),
                (True, True, False, True),
            ):
                model._decode_graph_enabled = enabled
                model._decode_graph_request_allowed = request
                model.dynamic_shapes = dynamic
                model.with_cross_attn = cross
                model._init_cache(inputs, mask)
                bound = model.transformer_layers[0].self_attn._cache_view_bound
                if enabled and request and not dynamic and not cross:
                    self.assertIs(bound, model._kv_cache_view_bound)
                else:
                    self.assertIsNone(bound)


@unittest.skipUnless(torch.cuda.is_available() and _FLASH_ATTN_AVAILABLE, "CUDA FlashAttention required")
class TestKVCacheViewCUDA(unittest.TestCase):
    def test_attention_matches_full_cache_and_rebinds_without_recompile(self):
        from eole.ops import flash_attn_kvcache
        from eole import EOLE_TORCH_COMPILE

        if not EOLE_TORCH_COMPILE:
            self.skipTest("custom-op compile path requires EOLE_TORCH_COMPILE=1")
        from eole.utils.decoder_graph import DecoderGraph

        torch.manual_seed(19)
        for dtype in (torch.float16, torch.bfloat16):
            for rows in (1, 4):
                q = torch.randn(2, rows, 4, 32, device="cuda", dtype=dtype)
                k = torch.randn(2, rows, 2, 32, device="cuda", dtype=dtype)
                v = torch.randn_like(k)
                cache_k = torch.randn(2, 4096, 2, 32, device="cuda", dtype=dtype)
                cache_v = torch.randn_like(cache_k)
                reference_k, reference_v = cache_k.clone(), cache_v.clone()
                lengths = torch.tensor([25, 30], device="cuda", dtype=torch.int32)
                left = torch.tensor([5, 0], device="cuda", dtype=torch.int32)
                bound = torch.tensor(512)
                mask = torch.zeros(2, 1, rows, device="cuda", dtype=torch.bool)
                compilations = []

                def function(inputs, *, tgt_pad_mask):
                    return flash_attn_kvcache(
                        inputs,
                        cache_k,
                        cache_v,
                        k,
                        v,
                        cache_seqlens=lengths,
                        cache_leftpad=left,
                        causal=True,
                        cache_view_bound=bound,
                    )

                def backend(gm, inputs):
                    compilations.append(gm)
                    return gm.forward

                compiled = torch.compile(function, backend=backend, fullgraph=True)
                runner = DecoderGraph(compiled, compiled)
                for capacity in (512, 2048, 1024):
                    runner.reset()
                    bound.fill_(capacity)
                    # Include writes at the last slot of the bounded view.
                    lengths.fill_(capacity - rows)
                    for _ in range(4):
                        q.normal_()
                        expected = flash_attn_kvcache(
                            q, reference_k, reference_v, k, v, cache_seqlens=lengths, cache_leftpad=left, causal=True
                        )
                        actual = runner(q, tgt_pad_mask=mask)
                        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
                    torch.testing.assert_close(cache_k, reference_k, rtol=0, atol=0)
                    torch.testing.assert_close(cache_v, reference_v, rtol=0, atol=0)
                self.assertEqual(len(compilations), 1)
