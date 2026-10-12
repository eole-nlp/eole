"""Decode graph capture must process real tokens exactly once."""

import unittest
from unittest.mock import patch

import torch

from eole.utils.decoder_graph import DecoderGraph


@unittest.skipUnless(torch.cuda.is_available(), "CUDA graph tests require CUDA")
class TestDecoderGraph(unittest.TestCase):
    def setUp(self):
        self.state = torch.zeros((), device="cuda")
        self.calls = 0

        def forward(emb, *, tgt_pad_mask):
            self.calls += 1
            self.state.add_(1)
            return emb.masked_fill(tgt_pad_mask, -9) + self.state, {"std": None}

        self.runner = DecoderGraph(forward, forward)

    @torch.no_grad()
    def test_capture_and_replay_advance_state_once_and_copy_inputs(self):
        for index in range(6):
            emb = torch.full((1, 1, 8), float(index), device="cuda")
            mask = torch.full((1, 1, 1), index == 4, device="cuda", dtype=torch.bool)
            output, attn = self.runner(emb, tgt_pad_mask=mask)
            expected = (-9 if index == 4 else index) + index + 1
            torch.testing.assert_close(output, torch.full_like(output, expected))
            self.assertIsNone(attn["std"])
            self.assertEqual(self.state.item(), index + 1)
        self.assertEqual(self.calls, 3)
        self.assertIsNotNone(self.runner.graph)

    @torch.no_grad()
    def test_compiled_program_does_not_replay_an_inner_graph(self):
        def function(emb, *, tgt_pad_mask):
            self.state.add_(1)
            return emb.masked_fill(tgt_pad_mask, -9) + self.state, {"std": None}

        compiled = torch.compile(function, fullgraph=True, options={"triton.cudagraphs": False})
        runner = DecoderGraph(compiled, compiled)
        original_replay = torch.cuda.CUDAGraph.replay

        def checked_replay(graph):
            self.assertFalse(torch.cuda.is_current_stream_capturing())
            return original_replay(graph)

        with patch.object(torch.cuda.CUDAGraph, "replay", checked_replay):
            for step in range(5):
                emb = torch.zeros(1, 1, 8, device="cuda")
                mask = torch.zeros(1, 1, 1, device="cuda", dtype=torch.bool)
                output, _ = runner(emb, tgt_pad_mask=mask)
                torch.testing.assert_close(output, torch.full_like(output, step + 1))
        self.assertIsNotNone(runner.graph)
        self.assertFalse(runner.failed)

    @torch.no_grad()
    def test_shape_change_and_cache_reset_bind_new_graph(self):
        for batch in (1, 2):
            for _ in range(4):
                emb = torch.zeros(batch, 1, 8, device="cuda")
                mask = torch.zeros(batch, 1, 1, device="cuda", dtype=torch.bool)
                output, _ = self.runner(emb, tgt_pad_mask=mask)
                torch.testing.assert_close(output, self.state.expand_as(output))
            self.assertIsNotNone(self.runner.graph)
        self.assertEqual(self.calls, 6)
        self.runner.reset()
        self.assertIsNone(self.runner.graph)
        self.state.zero_()
        output, _ = self.runner(emb, tgt_pad_mask=mask)
        torch.testing.assert_close(output, torch.ones_like(output))
        self.assertEqual(self.state.item(), 1)

    @torch.no_grad()
    def test_failed_capture_falls_back_without_advancing_twice(self):
        emb = torch.zeros(1, 1, 8, device="cuda")
        mask = torch.zeros(1, 1, 1, device="cuda", dtype=torch.bool)
        for _ in range(2):
            self.runner(emb, tgt_pad_mask=mask)
        with patch("torch.cuda.graph", side_effect=RuntimeError("unsupported capture")):
            for expected in (3, 4):
                output, _ = self.runner(emb, tgt_pad_mask=mask)
                torch.testing.assert_close(output, torch.full_like(output, expected))
        self.assertTrue(self.runner.failed)
        self.assertEqual(self.state.item(), 4)


class TestDecoderGraphInvalidation(unittest.TestCase):
    def test_cache_creation_mapping_and_cleanup_reset_bindings(self):
        from eole.config.inference import InferenceConfig
        from eole.config.models import TransformerDecoderConfig
        from eole.decoders.transformer import TransformerDecoder

        config = TransformerDecoderConfig(layers=1, hidden_size=64, heads=4, transformer_ff=128)
        decoder = TransformerDecoder(config, running_config=InferenceConfig(dynamic_shapes=False))
        from unittest.mock import Mock

        runner = Mock()
        decoder._decode_graph = runner
        emb = torch.zeros(2, 3, 64)
        decoder._init_cache(emb, torch.zeros(2, 1, 3, dtype=torch.bool))
        self.assertEqual(runner.reset.call_count, 1)
        decoder.map_state(lambda tensor: tensor[:1])
        self.assertEqual(runner.reset.call_count, 2)
        self.assertEqual(decoder.cache_seqlens.size(0), 1)
        decoder._disable_cache()
        self.assertEqual(runner.reset.call_count, 3)
        self.assertIsNone(decoder.cache_seqlens)
