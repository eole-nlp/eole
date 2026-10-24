"""Cached relative-position decode must match a teacher-forced full forward."""

import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from eole.config.inference import InferenceConfig
from eole.config.models import TransformerDecoderConfig
from eole.constants import PositionEncodingType
from eole.decoders.transformer import TransformerDecoder
from eole.modules import multi_headed_attn
from eole.modules.relative_position_bias import compute_bias, gen_relative_positions

# Shorter than the static cache (context_length=32) and the clip window (n_positions=8),
# so a query at the end of the buffer is a different distance from any real position.
SEQUENCE_LENGTH = 7


def _gen_cache_end_positions(length, n_positions, cache=False, device=None, query_len=None, query_offset=None):
    """Old cached decode: the query is the last slot of the key buffer."""
    if query_offset is None:
        return gen_relative_positions(
            length,
            n_positions,
            cache=cache,
            device=device,
            query_len=query_len,
            query_offset=query_offset,
        )
    return gen_relative_positions(length, n_positions, cache=True, device=device)


def _compute_cache_end_bias(
    query_length,
    key_length,
    is_decoder,
    n_positions,
    relative_positions_buckets,
    device=None,
    query_offset=None,
):
    """Old cached decode: take the last query rows of a key-length bias."""
    if query_offset is None:
        return compute_bias(
            query_length,
            key_length,
            is_decoder,
            n_positions,
            relative_positions_buckets,
            device=device,
        )
    full = compute_bias(
        key_length,
        key_length,
        is_decoder,
        n_positions,
        relative_positions_buckets,
        device=device,
    )
    return full[-query_length:]


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
        for name, parameter in decoder.named_parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
            elif name.endswith("bias"):
                nn.init.zeros_(parameter)
            else:
                # LayerNorm and RMSNorm scales
                nn.init.ones_(parameter)
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
        for buckets in (0, 8):
            for dynamic_shapes in (False, True):
                for prefill in (1, 4):
                    with self.subTest(buckets=buckets, dynamic=dynamic_shapes, prefill=prefill):
                        torch.manual_seed(0)
                        decoder = self._decoder(dynamic_shapes, buckets)
                        emb = torch.randn(1, SEQUENCE_LENGTH, decoder.hidden_size)
                        with torch.no_grad():
                            full = self._full_forward(decoder, emb)
                            cached = self._cached_forward(decoder, emb, prefill)
                        self.assertGreater(full.abs().max().item(), 0)
                        torch.testing.assert_close(cached, full, atol=1e-4, rtol=1e-4)

    def test_cache_end_offset_disagrees_with_full_forward(self):
        """The pre-fix formula places the query at the end of the KV buffer."""
        for buckets in (0, 8):
            with self.subTest(buckets=buckets):
                torch.manual_seed(0)
                decoder = self._decoder(dynamic_shapes=False, buckets=buckets)
                emb = torch.randn(1, SEQUENCE_LENGTH, decoder.hidden_size)
                with torch.no_grad():
                    full = self._full_forward(decoder, emb)
                    with (
                        patch.object(multi_headed_attn, "gen_relative_positions", _gen_cache_end_positions),
                        patch.object(multi_headed_attn, "compute_bias", _compute_cache_end_bias),
                    ):
                        cached = self._cached_forward(decoder, emb, prefill=1)
                self.assertGreater(full.abs().max().item(), 0)
                self.assertGreater((cached - full).abs().max().item(), 1e-2)

    def test_single_token_forward_without_cache(self):
        torch.manual_seed(0)
        decoder = self._decoder(dynamic_shapes=False, buckets=0)
        emb = torch.randn(1, 1, decoder.hidden_size)
        with torch.no_grad():
            out = self._full_forward(decoder, emb)

        self.assertEqual(tuple(out.shape), (1, 1, decoder.hidden_size))
        self.assertTrue(torch.isfinite(out).all())
