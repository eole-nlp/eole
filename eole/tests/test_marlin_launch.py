"""Dense Marlin launch coverage across decode, verification and prefill tiles."""

import unittest

import torch

from eole.ops import _CPP_OPS_AVAILABLE
from eole.modules.marlin_linear import MarlinQuantLinear


@unittest.skipUnless(torch.cuda.is_available() and _CPP_OPS_AVAILABLE, "Eole CUDA extension required")
class TestMarlinLaunch(unittest.TestCase):
    def projection(self, bits, group_size, dtype, n, desc_act=False, seed=476):
        k = 512
        torch.manual_seed(seed)
        layer = MarlinQuantLinear(bits, group_size, False, True, k, n, bias=True).cuda().to(dtype)
        group_size = layer.group_size
        pack = 32 // bits
        unsigned = torch.randint(0, 1 << bits, (k, n), device="cuda", dtype=torch.int64)
        scales = (torch.rand(k // group_size, n, device="cuda", dtype=dtype) * 0.02 + 0.01) * (
            7 / ((1 << (bits - 1)) - 1)
        )
        groups = torch.arange(k, device="cuda") // group_size
        if desc_act:
            # Full-K activation order exercises permutation before GEMM.
            groups = groups[torch.randperm(k, device="cuda")]
        reference = (unsigned - (1 << (bits - 1))).to(dtype) * scales[groups]
        shifts = torch.arange(pack, device="cuda", dtype=torch.int64) * bits
        packed = (unsigned.reshape(k // pack, pack, n) << shifts[None, :, None]).sum(1)
        layer.qweight.data.copy_(packed.to(torch.int32))
        layer.scales.data.copy_(scales)
        layer.g_idx.data.copy_(groups.to(torch.int32))
        layer.qzeros.data.zero_()
        # Constant bias avoids depending on the separate bias layout conversion.
        layer.bias.fill_(0.025)
        layer.post_init()
        if desc_act:
            # Exercise the CUDA permutation path directly with its int32 ABI.
            from eole.ops import gptq_marlin_repack

            perm = torch.argsort(groups).to(torch.int32)
            layer.qweight.data = gptq_marlin_repack(packed.to(torch.int32), perm, k, n, bits)
            layer.g_idx.data = groups[perm.long()].to(torch.int32)
            layer.g_idx_sort_indices = torch.nn.Parameter(perm, requires_grad=False)
        return layer, reference

    def test_dense_tiles_and_cached_launches(self):
        # N=64 needs the narrow fallback; N=256 exercises the former default
        # tile too. Groups/dtypes/bit widths select different specializations.
        for dtype in (torch.float16, torch.bfloat16):
            for bits in (4, 8):
                for group in (-1, 32, 64, 128):
                    for n in (64, 256):
                        with self.subTest(dtype=dtype, bits=bits, group=group, n=n):
                            layer, weight = self.projection(bits, group, dtype, n)
                            for m in (1, 4, 8, 9, 16, 32, 64, 129, 256):
                                x = torch.randn(m, 512, device="cuda", dtype=dtype)
                                expected = (x.float() @ weight.float() + layer.bias.float()).to(dtype)
                                for _ in range(2):
                                    actual = layer(x)
                                    torch.testing.assert_close(actual, expected, atol=0.04, rtol=0.02)

    def test_activation_order(self):
        for dtype in (torch.float16, torch.bfloat16):
            layer, weight = self.projection(4, 128, dtype, 256, desc_act=True)
            for m in (1, 4, 32, 64):
                x = torch.randn(m, 512, device="cuda", dtype=dtype)
                expected = (x.float() @ weight.float() + layer.bias.float()).to(dtype)
                torch.testing.assert_close(layer(x), expected, atol=0.04, rtol=0.02)

    def test_cuda_graph_replay(self):
        layer, weight = self.projection(4, 128, torch.bfloat16, 256)
        x = torch.randn(4, 512, device="cuda", dtype=torch.bfloat16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            layer(x)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = layer(x)
        for _ in range(3):
            x.normal_()
            graph.replay()
            expected = (x.float() @ weight.float() + layer.bias.float()).to(x.dtype)
            torch.testing.assert_close(result, expected, atol=0.04, rtol=0.02)

    def test_wide_output_multiple_ctas(self):
        for bits in (4, 8):
            layer, weight = self.projection(bits, 128, torch.bfloat16, 24576)
            for m in (1, 4):
                x = torch.randn(m, 512, device="cuda", dtype=torch.bfloat16)
                expected = (x.float() @ weight.float() + layer.bias.float()).to(x.dtype)
                torch.testing.assert_close(layer(x), expected, atol=0.04, rtol=0.02)

    def test_shared_moe_kernel(self):
        from eole.modules.fused_moe_marlin import _marlin_gemm
        from eole.modules.moe_quant_utils import _moe_align_block_size

        for dtype in (torch.float16, torch.bfloat16):
            layers, weights = zip(*(self.projection(4, 128, dtype, 256, seed=476 + expert) for expert in range(2)))
            qweight = torch.stack([layer.qweight for layer in layers])
            scales = torch.stack([layer.scales for layer in layers])
            empty = torch.empty(2, 0, device="cuda", dtype=torch.int32)
            for m, block in ((1, 8), (4, 8), (32, 32), (128, 64)):
                x = torch.randn(m, 512, device="cuda", dtype=dtype)
                ids = (torch.arange(m, device="cuda") % 2).to(torch.int32).view(m, 1)
                sorted_ids, experts, padded = _moe_align_block_size(ids, block, 2)
                output = torch.empty(m, 256, device="cuda", dtype=dtype)
                _marlin_gemm(
                    x,
                    output,
                    qweight,
                    scales,
                    empty,
                    empty,
                    empty,
                    layers[0].workspace,
                    sorted_ids,
                    experts,
                    padded,
                    torch.ones(m, 1, device="cuda"),
                    block,
                    1,
                    m,
                    256,
                    512,
                    layers[0].weight_type.id,
                )
                reference = torch.empty_like(output)
                for expert in range(2):
                    mask = ids[:, 0] == expert
                    reference[mask] = (x[mask].float() @ weights[expert].float()).to(dtype)
                torch.testing.assert_close(output, reference, atol=0.04, rtol=0.02)
