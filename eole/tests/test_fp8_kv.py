"""Exercise direct FP8 reads with different per-row offsets on CUDA replay."""

import importlib.util
import unittest
from unittest.mock import patch

import torch

from eole.modules.fp8_kv import configure_fp8_cache, fp8_kv_attention


@unittest.skipUnless(
    torch.cuda.is_available() and importlib.util.find_spec("triton") and torch.cuda.get_device_capability() >= (8, 9),
    "requires FP8 CUDA and Triton",
)
class TestFP8KV(unittest.TestCase):
    @torch.inference_mode()
    def test_append_leaves_invalid_rows_unchanged(self):
        query = torch.ones(3, 2, 4, 64, device="cuda", dtype=torch.bfloat16)
        key = query[:, :, :2]
        caches = [torch.zeros((3, 8, 2, 64), device="cuda", dtype=torch.float8_e4m3fn) for _ in range(2)]
        lengths = torch.tensor([7, -1, 6], device="cuda", dtype=torch.int32)
        output = fp8_kv_attention(query, *caches, key, key, lengths, 64**-0.5)
        for cache in caches:
            self.assertEqual(cache[:2].float().count_nonzero().item(), 0)
            torch.testing.assert_close(cache[2, -2:].float(), key[2].float())
        self.assertEqual(output[:2].count_nonzero().item(), 0)

    @torch.inference_mode()
    def test_decode_and_verify_replay_after_rewind(self):
        torch.manual_seed(39)
        capacity, heads, kv_heads, dim = 2048, 16, 4, 256
        dtype = torch.bfloat16
        for width in (1, 4, 8, 15, 16):
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


@unittest.skipUnless(
    torch.cuda.is_available()
    and importlib.util.find_spec("flashinfer")
    and torch.cuda.get_device_capability() >= (8, 9),
    "requires FP8 CUDA and FlashInfer",
)
class TestContinuousFP8KV(unittest.TestCase):
    def test_rejects_fp8_if_flash_cache_is_unavailable(self):
        from eole.constants import PositionEncodingType
        from eole.tests.test_continuous_gpu import ContinuousDecoderGpuTests

        model, runner = ContinuousDecoderGpuTests().fixture()
        runner.close()
        decoder = model.decoder
        with patch("eole.decoders.transformer._FLASH_ATTN_AVAILABLE", False):
            with self.assertRaisesRegex(ValueError, "usable FlashAttention KV-cache backend"):
                configure_fp8_cache(decoder, next(model.parameters()), None)
        with patch.object(decoder, "position_encoding_type", PositionEncodingType.Alibi):
            with self.assertRaisesRegex(ValueError, "usable FlashAttention KV-cache backend"):
                configure_fp8_cache(decoder, next(model.parameters()), None)
        self.assertNotIn("fp8_kv", decoder._verifier_cache_config)
        self.assertTrue(all(not hasattr(layer.self_attn, "_fp8_kv_scales") for layer in decoder.transformer_layers))

    def test_rejects_continuous_native_with_standalone_fp8(self):
        from eole.predict.continuous import ContinuousGreedy
        from eole.tests.test_continuous_gpu import ContinuousDecoderGpuTests

        model, runner = ContinuousDecoderGpuTests().fixture()
        runner.close()
        configure_fp8_cache(model.decoder, next(model.parameters()), None)
        with self.assertRaisesRegex(ValueError, "Continuous native KV conflicts"):
            ContinuousGreedy(runner.predictor, kv_cache_dtype="native")

    @torch.inference_mode()
    def test_chunked_prefill_reads_scaled_prefix(self):
        from eole.modules.fp8_kv import prefill

        torch.manual_seed(912)
        q = torch.randn(1, 23, 4, 128, device="cuda", dtype=torch.bfloat16)
        k, v = [torch.randn_like(q[:, :, :2]) for _ in range(2)]
        caches = [torch.zeros_like(x, dtype=torch.float8_e4m3fn) for x in (k, v)]
        lengths = torch.zeros(1, device="cuda", dtype=torch.int32)
        scales = (0.021, 0.061)
        outputs = []
        for start, end in ((0, 11), (11, 23)):
            lengths.fill_(start)
            outputs.append(
                prefill(q[:, start:end], *caches, k[:, start:end], v[:, start:end], lengths, 128**-0.5, *scales)
            )
        dequant = [(x.float() * scale).to(q.dtype) for x, scale in zip(caches, scales)]
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), *(x.transpose(1, 2) for x in dequant), is_causal=True, enable_gqa=True
        )
        torch.testing.assert_close(torch.cat(outputs, 1), ref.transpose(1, 2), atol=0.02, rtol=0.025)

    @torch.inference_mode()
    def test_fp8_prompt_reuse_in_all_modes(self):
        from eole import EOLE_TORCH_COMPILE
        from eole.tests import test_continuous_dflash2_gpu as helpers

        fixtures = helpers.ContinuousDFlash2GpuTests()
        fixtures.setUp()
        options = dict(kv_cache_dtype="fp8_e4m3", kv_cache_scales=[None, [0.019, 0.023]])
        for mode in ("greedy", "mtp", "dflash2"):
            with self.subTest(mode=mode):
                if mode == "dflash2":
                    _, runner = fixtures._runner(EOLE_TORCH_COMPILE, gdn=True, prefix_cache_entries=1, **options)
                else:
                    _, runner = fixtures.tools.fixture(
                        gdn=True, compiled=EOLE_TORCH_COMPILE, mtp=mode == "mtp", prefix_cache_entries=1, **options
                    )
                try:
                    runner.prefill_idle_chunk_size = 32
                    prefix = list(range(3, 23))
                    for name, ids in (("initial", prefix), ("hit", prefix + [23, 24]), ("cold", prefix + [23, 24])):
                        request = runner.submit(name, ids, 6)
                        while runner.has_work:
                            runner.tick()
                        self.assertEqual(len(request.tokens), 6)
                    self.assertEqual(runner.requests["hit"].reused_prefix_tokens, len(prefix))
                    self.assertEqual(runner.requests["cold"].reused_prefix_tokens, 0)
                    self.assertEqual(runner.requests["hit"].tokens, runner.requests["cold"].tokens)
                    caches = [state["kcache"] for _, state in runner.bank.layers if "kcache" in state]
                    self.assertTrue(caches)
                    self.assertTrue(all(cache.dtype == torch.float8_e4m3fn for cache in caches))
                finally:
                    runner.close()
