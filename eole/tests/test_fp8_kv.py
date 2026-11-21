"""Exercise direct FP8 reads with different per-row offsets on CUDA replay."""

import importlib.util
import unittest

import torch

from eole.modules.fp8_kv import fp8_kv_attention


@unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec("flashinfer"), "requires CUDA FlashInfer")
class TestFP8KV(unittest.TestCase):
    @torch.inference_mode()
    def test_decode_and_verify_replay_after_rewind(self):
        torch.manual_seed(39)
        capacity, heads, kv_heads, dim = 128, 8, 2, 128
        dtype = torch.bfloat16
        for width in (1, 4, 8):
            with self.subTest(width=width):
                cache = [
                    torch.randn(2, capacity, kv_heads, dim, device="cuda").to(torch.float8_e4m3fn) for _ in range(2)
                ]
                updates = [torch.randn(2, width, kv_heads, dim, device="cuda", dtype=dtype) for _ in range(2)]
                query = torch.randn(2, width, heads, dim, device="cuda", dtype=dtype)
                lengths = torch.tensor([33, 76], device="cuda", dtype=torch.int32)
                compiled = torch.compile(fp8_kv_attention, fullgraph=True, options={"triton.cudagraphs": False})
                for _ in range(2):
                    compiled(query, *cache, *updates, lengths, dim**-0.5)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    result = compiled(query, *cache, *updates, lengths, dim**-0.5)
                # Replay at new unequal offsets, including an overwrite after rejection.
                for offsets in ([33, 76], [34, 55], [34, 56]):
                    lengths.copy_(torch.tensor(offsets))
                    updates[0].normal_()
                    graph.replay()
                    for row, start in enumerate(offsets):
                        for actual, update in zip(cache, updates):
                            torch.testing.assert_close(
                                actual[row, start : start + width].float(),
                                update[row].to(torch.float8_e4m3fn).float(),
                            )
                        k, v = [x[row, : start + width].to(dtype).transpose(0, 1)[None] for x in cache]
                        mask = torch.arange(start + width, device="cuda") <= (
                            start + torch.arange(width, device="cuda")[:, None]
                        )
                        ref = torch.nn.functional.scaled_dot_product_attention(
                            query[row].transpose(0, 1)[None], k, v, attn_mask=mask, enable_gqa=True
                        )
                        torch.testing.assert_close(result[row], ref[0].transpose(0, 1), atol=0.005, rtol=0.02)
