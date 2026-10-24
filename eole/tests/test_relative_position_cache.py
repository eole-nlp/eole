"""Cached relative-position decode must match a teacher-forced full forward."""

import unittest

import torch
import torch.nn as nn

from eole.config.inference import InferenceConfig
from eole.config.models import TransformerDecoderConfig
from eole.constants import PositionEncodingType
from eole.decoders.transformer import TransformerDecoder
from eole.modules.relative_position_bias import compute_bias, gen_relative_positions


class TestRelativePositionMatrix(unittest.TestCase):
    def test_offset_rows_match_the_square_matrix(self):
        full = gen_relative_positions(6, n_positions=4)
        got = gen_relative_positions(8, n_positions=4, query_len=2, query_offset=torch.tensor(2))

        torch.testing.assert_close(got[:, :6], full[2:4])

    def test_bucket_offset_matches_rows_of_the_unshifted_matrix(self):
        full = compute_bias(6, 8, is_decoder=True, n_positions=16, relative_positions_buckets=8)
        got = compute_bias(
            2,
            8,
            is_decoder=True,
            n_positions=16,
            relative_positions_buckets=8,
            query_offset=torch.tensor(2),
        )

        torch.testing.assert_close(got, full[2:4])


class TestRelativePositionCache(unittest.TestCase):
    def _decoder(self, dynamic_shapes, buckets):
        config = TransformerDecoderConfig(
            layers=2,
            hidden_size=32,
            heads=4,
            transformer_ff=64,
            with_cross_attn=False,
            max_position_embeddings=32,
            position_encoding_type=PositionEncodingType.Relative,
            n_positions=8,
            relative_positions_buckets=buckets,
        )
        running = InferenceConfig(
            self_attn_backend="pytorch",
            dynamic_shapes=dynamic_shapes,
            context_length=32,
        )
        decoder = TransformerDecoder(config, running_config=running).eval()
        for parameter in decoder.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
            else:
                nn.init.zeros_(parameter)
        return decoder

    def _full_forward(self, decoder, emb):
        decoder._disable_cache()
        pad = torch.zeros(emb.size(0), 1, emb.size(1), dtype=torch.bool)
        out, _ = decoder(emb, tgt_pad_mask=pad)
        return out

    def _cached_forward(self, decoder, emb, prefill):
        decoder._disable_cache()
        outputs = []
        position = 0
        if prefill:
            chunk = emb[:, :prefill]
            pad = torch.zeros(chunk.size(0), 1, chunk.size(1), dtype=torch.bool)
            decoder._init_cache(chunk, pad)
            out, _ = decoder(chunk, tgt_pad_mask=pad)
            outputs.append(out)
            decoder._extend_cache()
            position = prefill
        while position < emb.size(1):
            chunk = emb[:, position : position + 1]
            pad = torch.zeros(chunk.size(0), 1, 1, dtype=torch.bool)
            if position == 0:
                decoder._init_cache(chunk, pad)
            out, _ = decoder(chunk, tgt_pad_mask=pad)
            outputs.append(out)
            decoder._extend_cache()
            position += 1
        return torch.cat(outputs, dim=1)

    def test_cached_decode_matches_full_forward(self):
        sequence = 7
        for buckets in (0, 8):
            for dynamic_shapes in (False, True):
                for prefill in (1, 4):
                    with self.subTest(buckets=buckets, dynamic=dynamic_shapes, prefill=prefill):
                        torch.manual_seed(0)
                        decoder = self._decoder(dynamic_shapes, buckets)
                        emb = torch.randn(1, sequence, decoder.hidden_size)
                        with torch.no_grad():
                            full = self._full_forward(decoder, emb)
                            cached = self._cached_forward(decoder, emb, prefill)
                        torch.testing.assert_close(cached, full, atol=1e-4, rtol=1e-4)
