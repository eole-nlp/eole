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
    def test_verifier_only_replays_full_speculative_chunks(self):
        from types import SimpleNamespace
        from eole.decoders.transformer import TransformerDecoder

        decoder = SimpleNamespace(
            _decode_graph_enabled=True,
            _verify_graph_request_allowed=True,
            _verify_graph_sequence_length=4,
            _speculative_forward=True,
            training=False,
            flash=True,
            dynamic_shapes=False,
            with_cross_attn=False,
            cache_seqlens=torch.zeros(1, device="cuda", dtype=torch.int32),
        )
        for length, expected in ((4, True), (3, False), (2, False), (1, False)):
            emb = torch.zeros(1, length, 8, device="cuda", dtype=torch.bfloat16)
            kwargs = {"tgt_pad_mask": torch.zeros(1, 1, length, device="cuda", dtype=torch.bool)}
            self.assertEqual(TransformerDecoder._can_replay_verifier(decoder, emb, kwargs), expected)
        emb = torch.zeros(1, 4, 8, device="cuda", dtype=torch.bfloat16)
        kwargs = {"tgt_pad_mask": torch.zeros(1, 1, 4, device="cuda", dtype=torch.bool)}
        decoder._speculative_forward = False
        self.assertFalse(TransformerDecoder._can_replay_verifier(decoder, emb, kwargs))
        decoder._speculative_forward = True
        decoder._verify_graph_request_allowed = False
        self.assertFalse(TransformerDecoder._can_replay_verifier(decoder, emb, kwargs))

    @torch.no_grad()
    def test_verification_replay_leaves_acceptance_commit_outside(self):
        state = torch.zeros(1, 1, 1, device="cuda")
        transaction = torch.empty(1, 4, 1, device="cuda")
        position = torch.zeros((), device="cuda", dtype=torch.long)

        def verify(emb, *, tgt_pad_mask):
            transaction.copy_(state + emb.masked_fill(tgt_pad_mask.transpose(1, 2), 0).cumsum(1))
            position.add_(emb.size(1))
            return transaction, {"std": None}

        runner = DecoderGraph(verify, verify)
        expected_state, expected_position = 0, 0
        mask = torch.zeros(1, 1, 4, device="cuda", dtype=torch.bool)
        for index, kept in enumerate((1, 4, 2, 3, 1, 4)):
            emb = torch.full((1, 4, 1), float(index + 1), device="cuda")
            output, _ = runner(emb, tgt_pad_mask=mask)
            expected = expected_state + (index + 1) * torch.arange(1, 5, device="cuda").view(1, 4, 1)
            torch.testing.assert_close(output, expected.to(output.dtype))
            self.assertEqual(state.item(), expected_state)
            self.assertEqual(position.item(), expected_position + 4)
            # Acceptance/rollback happens after verification, outside replay.
            state.copy_(transaction[:, kept - 1 : kept])
            position.sub_(4 - kept)
            expected_state += (index + 1) * kept
            expected_position += kept
        self.assertEqual(state.item(), expected_state)
        self.assertEqual(position.item(), expected_position)
        self.assertIsNotNone(runner.graph)

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
        verifier = Mock()
        decoder._verify_graph = verifier
        decoder._verify_buffer_signature = ({},)
        emb = torch.zeros(2, 3, 64)
        decoder._init_cache(emb, torch.zeros(2, 1, 3, dtype=torch.bool))
        self.assertEqual(runner.reset.call_count, 1)
        decoder.map_state(lambda tensor: tensor[:1])
        self.assertEqual(runner.reset.call_count, 2)
        self.assertEqual(decoder.cache_seqlens.size(0), 1)
        decoder._disable_cache()
        self.assertEqual(runner.reset.call_count, 3)
        self.assertEqual(verifier.reset.call_count, 3)
        self.assertIsNone(decoder._verify_buffer_signature)
        self.assertIsNone(decoder.cache_seqlens)

    def test_verifier_rebinds_replaced_speculative_buffers(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from eole.decoders.transformer import TransformerDecoder

        runner = Mock()
        buffers = {"query": torch.zeros(1, 4, 1)}
        linear = SimpleNamespace(_spec_buffers=buffers)
        decoder = SimpleNamespace(
            transformer_layers=[SimpleNamespace(linear_attn=linear), SimpleNamespace()],
            _verify_graph=runner,
            _verify_buffer_signature=None,
        )
        emb = torch.zeros(1, 4, 8)
        kwargs = {"tgt_pad_mask": torch.zeros(1, 1, 4, dtype=torch.bool)}
        TransformerDecoder._replay_verifier(decoder, emb, kwargs)
        self.assertEqual(runner.reset.call_count, 1)
        # In-place writes preserve the binding; replacing the buffers does not.
        buffers["query"].fill_(2)
        TransformerDecoder._replay_verifier(decoder, emb, kwargs)
        self.assertEqual(runner.reset.call_count, 1)
        linear._spec_buffers = {"query": torch.zeros(1, 4, 1)}
        TransformerDecoder._replay_verifier(decoder, emb, kwargs)
        self.assertEqual(runner.reset.call_count, 2)
        self.assertIs(decoder._verify_buffer_signature[0], linear._spec_buffers)
