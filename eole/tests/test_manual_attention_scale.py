"""fp16 manual attention must scale the query before Q·Kᵀ."""

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from eole.config.models import TransformerDecoderConfig
from eole.constants import PositionEncodingType
from eole.modules.multi_headed_attn import SelfMHA
from eole.modules.relative_position_bias import gen_relative_positions, relative_matmul


def _attention(attn_scaling, relative):
    config = TransformerDecoderConfig(
        hidden_size=64,
        heads=1,
        transformer_ff=64,
        attn_scaling=attn_scaling,
        position_encoding_type=(
            PositionEncodingType.Relative if relative else PositionEncodingType.SinusoidalInterleaved
        ),
        n_positions=4 if relative else None,
    )
    attention = SelfMHA(config, is_decoder=True).eval()
    nn.init.eye_(attention.final_linear.weight)
    if relative:
        nn.init.normal_(attention.relative_positions_embeddings.weight, std=0.1)
    return attention


class TestManualAttentionScale(unittest.TestCase):
    def test_fp16_scores_stay_finite_when_the_raw_product_overflows(self):
        dim = 64
        query = torch.full((1, 1, 1, dim), 70.0)
        key = torch.zeros((1, 2, 1, dim))
        key[:, 0] = 35
        key[:, 1] = 20
        value = torch.zeros((1, 2, 1, dim))
        value[:, 0] = 1
        value[:, 1] = -1
        raw = torch.matmul(query.half().transpose(1, 2), key.half().transpose(1, 2).transpose(2, 3))
        self.assertTrue(torch.isinf(raw).any())

        for attn_scaling in (None, 16.0):
            with self.subTest(attn_scaling=attn_scaling):
                attention = _attention(attn_scaling, relative=False)
                reference, _ = attention._compute_attention(key, value, query)
                half = attention.half()
                got, _ = half._compute_attention(key.half(), value.half(), query.half())

                self.assertTrue(torch.isfinite(got).all())
                torch.testing.assert_close(got.float(), reference, atol=1e-2, rtol=1e-2)

    def test_relative_term_keeps_the_unscaled_query(self):
        torch.manual_seed(0)
        attention = _attention(attn_scaling=None, relative=True)
        dim = attention.dim_per_head
        query = torch.randn(1, 3, 1, dim)
        key = torch.randn(1, 3, 1, dim)
        value = torch.randn(1, 3, 1, dim)
        _, got = attention._compute_attention(key, value, query, return_attn=True)

        scale = dim**-0.5
        query_heads = query.transpose(1, 2)
        scores = torch.matmul(query_heads * scale, key.transpose(1, 2).transpose(2, 3))
        matrix = gen_relative_positions(key.size(1), attention.n_positions)
        relations = attention.relative_positions_embeddings(matrix)
        scores = scores + relative_matmul(query_heads, relations, True)
        expected = F.softmax(scores.float(), dim=-1)

        torch.testing.assert_close(got.float(), expected)
