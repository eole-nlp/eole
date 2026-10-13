"""Packed checkpoint shards preserve the unsharded quantized projection."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from eole.models.model import BaseModel, MTPDraftingMixin
from eole.modules.marlin_linear import MarlinQuantLinear
from eole.modules.quantization_loading import slice_gptq_checkpoint
from eole.ops import _CPP_OPS_AVAILABLE


def fixture(bits, group_size):
    k, n = 512, 128
    factor = 32 // bits
    codes = (torch.arange(k * n).reshape(k, n) * 7 + torch.arange(k).view(-1, 1)) % (1 << bits)
    shifts = torch.arange(factor) * bits
    qweight = (codes.reshape(k // factor, factor, n).to(torch.int64) << shifts.view(1, -1, 1)).sum(1).int()
    groups = 1 if group_size == -1 else k // group_size
    scales = (torch.arange(groups * n).reshape(groups, n) % 19 + 1).half() / 64
    zero = (1 << (bits - 1)) - 1
    qzeros = torch.full((groups, n // factor), sum(zero << int(s) for s in shifts), dtype=torch.int64).int()
    g_idx = torch.zeros(k, dtype=torch.int32) if group_size == -1 else torch.arange(k, dtype=torch.int32) // group_size
    tensors = dict(qweight=qweight, scales=scales, qzeros=qzeros, g_idx=g_idx)
    weight = (codes.float() - (1 << (bits - 1))) * scales[g_idx].float()
    return tensors, weight


def dequantize(module):
    shifts = torch.arange(module.pack_factor) * module.bits
    codes = ((module.qweight.long().unsqueeze(1) >> shifts.view(1, -1, 1)) & ((1 << module.bits) - 1)).reshape(
        module.in_features, module.out_features
    )
    zeros = ((module.qzeros.long().unsqueeze(-1) >> shifts) & ((1 << module.bits) - 1)).reshape(
        module.scales.size(0), module.out_features
    ) + 1
    return (codes.float() - zeros[module.g_idx]) * module.scales[module.g_idx].float()


class TestPackedTensorParallel(unittest.TestCase):
    def load(self, module, tensors, rank, base_name):
        model = BaseModel.__new__(BaseModel)
        nn.Module.__init__(model)
        for name, tensor in tensors.items():
            BaseModel._load_param(model, base_name, module, name, getattr(module, name), [], tensor, rank)

    def test_both_ranks_preserve_column_and_row_projections(self):
        for bits in (4, 8):
            for group_size in (128, -1):
                tensors, weight = fixture(bits, group_size)
                x = torch.randn(3, 512, generator=torch.Generator().manual_seed(17))
                for column in (True, False):
                    with self.subTest(bits=bits, group_size=group_size, column=column):
                        outputs = []
                        for rank in (0, 1):
                            module = MarlinQuantLinear(
                                bits, group_size, False, True, 512 if column else 256, 64 if column else 128
                            )
                            self.load(module, tensors, rank, "linear_query" if column else "down_proj")
                            local = dequantize(module)
                            expected = (
                                weight[:, rank * 64 : (rank + 1) * 64]
                                if column
                                else weight[rank * 256 : (rank + 1) * 256]
                            )
                            torch.testing.assert_close(local, expected, rtol=0, atol=0)
                            outputs.append((x if column else x[:, rank * 256 : (rank + 1) * 256]) @ local)
                        result = torch.cat(outputs, dim=-1) if column else sum(outputs)
                        torch.testing.assert_close(result, x @ weight, rtol=1e-5, atol=1e-4)

    def test_gdn_projection_remains_replicated_on_rank_one(self):
        tensors, weight = fixture(4, 128)
        module = MarlinQuantLinear(4, 128, False, True, 512, 128)
        self.load(module, tensors, 1, "in_proj_qkv")
        torch.testing.assert_close(dequantize(module), weight, rtol=0, atol=0)

    def test_column_bias_and_group_indices(self):
        module = MarlinQuantLinear(4, 128, False, True, 512, 64, bias=True)
        tensors, _ = fixture(4, 128)
        tensors["bias"] = torch.arange(128).half()
        model = BaseModel.__new__(BaseModel)
        nn.Module.__init__(model)
        BaseModel._load_param(
            model, "linear_query", module, "bias", module.bias, ["linear_query.bias"], tensors["bias"], 1
        )
        torch.testing.assert_close(module.bias, tensors["bias"][64:])
        self.load(module, {k: v for k, v in tensors.items() if k != "bias"}, 1, "linear_query")
        torch.testing.assert_close(module.g_idx, tensors["g_idx"])

    def test_rejects_partial_groups_and_row_activation_reordering(self):
        module = MarlinQuantLinear(4, 128, True, True, 256, 128)
        tensors, _ = fixture(4, 128)
        with self.assertRaisesRegex(ValueError, "activation reordering"):
            slice_gptq_checkpoint(module, "qweight", module.qweight, tensors["qweight"], 0, row_parallel=True)
        module.desc_act = False
        module.group_size = 512
        with self.assertRaisesRegex(ValueError, "complete quantization groups"):
            slice_gptq_checkpoint(module, "qweight", module.qweight, tensors["qweight"], 0, row_parallel=True)

    def test_unsharded_activation_order_is_unchanged(self):
        tensors, _ = fixture(4, 128)
        tensors["g_idx"] = tensors["g_idx"].flip(0)
        module = MarlinQuantLinear(4, 128, True, True, 512, 128)
        self.load(module, tensors, 0, "down_proj")
        for name, tensor in tensors.items():
            torch.testing.assert_close(getattr(module, name), tensor, rtol=0, atol=0)

    def test_mtp_cache_uses_local_kv_heads_and_restores_previous_cache(self):
        for partitions in (1, 2, 4):
            with self.subTest(partitions=partitions):
                attention = SimpleNamespace(
                    heads_kv=8,
                    parallel_gpu=partitions,
                    dim_per_head=4,
                    kcache=None,
                    vcache=None,
                    cache_leftpad=None,
                )
                model = MTPDraftingMixin()
                model.mtp_heads = [SimpleNamespace(emb_norm=nn.Identity(), layer=SimpleNamespace(self_attn=attention))]
                model.init_mtp_cache(torch.zeros(1, 1, 16), torch.zeros(1, 1, dtype=torch.long), 3)
                self.assertEqual(attention.kcache.shape, (1, 5, 8 // partitions, 4))
                self.assertEqual(attention.vcache.shape, attention.kcache.shape)
                model.clear_mtp_cache()
                self.assertIsNone(attention.kcache)
                self.assertIsNone(attention.vcache)

    def test_rejects_bad_checkpoint_shape(self):
        module = MarlinQuantLinear(4, 128, False, True, 512, 64)
        tensors, _ = fixture(4, 128)
        tensors["qweight"] = tensors["qweight"][:16]
        with self.assertRaisesRegex(ValueError, "Invalid packed tensor-parallel shard"):
            self.load(module, tensors, 0, "linear_query")

    @unittest.skipUnless(
        torch.cuda.device_count() >= 2 and _CPP_OPS_AVAILABLE, "requires two GPUs and Eole CUDA kernels"
    )
    def test_two_gpu_marlin_outputs_match_full_projection(self):
        tensors, _ = fixture(4, 128)
        x = torch.randn(3, 512, generator=torch.Generator().manual_seed(17)).bfloat16()
        full = MarlinQuantLinear(4, 128, False, True, 512, 128)
        self.load(full, tensors, 0, "in_proj_qkv")
        with torch.cuda.device(0):
            full.cuda(0).post_init()
            reference = full(x.cuda(0)).cpu().float()
        for column in (True, False):
            outputs = []
            for rank in (0, 1):
                module = MarlinQuantLinear(4, 128, False, True, 512 if column else 256, 64 if column else 128)
                self.load(module, tensors, rank, "linear_query" if column else "down_proj")
                with torch.cuda.device(rank):
                    module.cuda(rank).post_init()
                    local_x = x if column else x[:, rank * 256 : (rank + 1) * 256].contiguous()
                    outputs.append(module(local_x.cuda(rank)).cpu().float())
            result = torch.cat(outputs, -1) if column else sum(outputs)
            torch.testing.assert_close(result, reference, rtol=0.02, atol=0.1)
