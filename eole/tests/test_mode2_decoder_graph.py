"""Outer capture must not leave layers on the bound compilation path."""

import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

from eole.decoders.transformer import TransformerDecoder


class TestMode2DecoderGraph(unittest.TestCase):
    def test_bound_layer_flags_are_restored_on_success_and_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                layers = [SimpleNamespace(), SimpleNamespace()]

                def forward(emb, *, tgt_pad_mask):
                    self.assertTrue(all(layer._use_bound_graph for layer in layers))
                    if fail:
                        raise RuntimeError("capture failed")
                    return emb, tgt_pad_mask

                decoder = SimpleNamespace(transformer_layers=layers, _forward_eager=forward)
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "capture failed"):
                        TransformerDecoder._forward_mode2_graph(decoder, "emb", tgt_pad_mask="mask")
                else:
                    self.assertEqual(
                        TransformerDecoder._forward_mode2_graph(decoder, "emb", tgt_pad_mask="mask"), ("emb", "mask")
                    )
                self.assertTrue(all(not layer._use_bound_graph for layer in layers))

    def test_mode2_verifier_captures_eager_program_without_whole_decoder_compilation(self):
        decoder = SimpleNamespace(
            transformer_layers=[], _verify_graph=None, _verify_buffer_signature=None, _forward_eager=Mock()
        )
        runner = Mock()
        with patch("eole.decoders.transformer.EOLE_COMPILE_MODE", "2"), patch(
            "torch.compile", side_effect=AssertionError("mode 2 must not compile the whole decoder")
        ), patch("eole.utils.decoder_graph.DecoderGraph", return_value=runner) as factory:
            TransformerDecoder._replay_verifier(decoder, "emb", {"tgt_pad_mask": "mask"})
        factory.assert_called_once_with(decoder._forward_eager, decoder._forward_eager)
        runner.assert_called_once_with("emb", tgt_pad_mask="mask")

    def test_mode2_outer_decode_capture_keeps_legacy_eager_fallback(self):
        decoder = SimpleNamespace(_decode_graph=None, _forward_mode2_graph=Mock(), _forward_eager=Mock())
        runner = Mock()
        with patch("eole.decoders.transformer.EOLE_COMPILE_MODE", "2"), patch(
            "torch.compile", side_effect=AssertionError("mode 2 must not compile the whole decoder")
        ), patch("eole.utils.decoder_graph.DecoderGraph", return_value=runner) as factory:
            TransformerDecoder._replay_decoder(decoder, "emb", {"tgt_pad_mask": "mask"})
        factory.assert_called_once_with(decoder._forward_mode2_graph, decoder._forward_eager)
        runner.assert_called_once_with("emb", tgt_pad_mask="mask")
