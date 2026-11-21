"""Exercise direct FP8 reads with different per-row offsets on CUDA replay."""

import importlib.util
import unittest

import torch

from eole.modules.fp8_kv import fp8_kv_attention


@unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec("triton"), "requires CUDA Triton")
class TestFP8KV(unittest.TestCase):
    @torch.inference_mode()
    def test_decode_and_verify_replay_after_rewind(self):
        torch.manual_seed(39)
        capacity, heads, kv_heads, dim = 2048, 8, 2, 128
        dtype = torch.bfloat16
        for width in (1, 4, 8):
            with self.subTest(width=width):
                cache = [
                    torch.randn(2, capacity, kv_heads, dim, device="cuda").to(torch.float8_e4m3fn) for _ in range(2)
                ]
                updates = [torch.randn(2, width, kv_heads, dim, device="cuda", dtype=dtype) for _ in range(2)]
                query = torch.randn(2, width, heads, dim, device="cuda", dtype=dtype)
                lengths = torch.tensor([509, 1021], device="cuda", dtype=torch.int32)
                compiled = torch.compile(fp8_kv_attention, fullgraph=True, options={"triton.cudagraphs": False})
                scales = (0.017, 0.031)
                for _ in range(2):
                    compiled(query, *cache, *updates, lengths, dim**-0.5, *scales)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    result = compiled(query, *cache, *updates, lengths, dim**-0.5, *scales)
                # Replay at new unequal offsets, including an overwrite after rejection.
                for offsets in ([509, 1021], [510, 995], [510, 996]):
                    lengths.copy_(torch.tensor(offsets))
                    updates[0].normal_()
                    graph.replay()
                    for row, start in enumerate(offsets):
                        for actual, update, scale in zip(cache, updates, scales):
                            torch.testing.assert_close(
                                actual[row, start : start + width].float(),
                                (update[row].float() / scale).to(torch.float8_e4m3fn).float(),
                            )
                        k, v = [
                            (x[row, : start + width].float() * scale).to(dtype).transpose(0, 1)[None]
                            for x, scale in zip(cache, scales)
                        ]
                        mask = torch.arange(start + width, device="cuda") <= (
                            start + torch.arange(width, device="cuda")[:, None]
                        )
                        ref = torch.nn.functional.scaled_dot_product_attention(
                            query[row].transpose(0, 1)[None], k, v, attn_mask=mask, enable_gqa=True
                        )
                        torch.testing.assert_close(result[row], ref[0].transpose(0, 1), atol=0.005, rtol=0.02)
