"""Tests for Multi-Token Prediction (MTP) support.

These tests exercise:
* MTPHead forward pass.
* DecoderModel with MTP heads (build + forward in training mode).
* LossCompute MTP auxiliary loss.
* Statistics mtp_loss accumulation.
"""

import unittest
import torch
import torch.nn as nn

from collections import Counter
import pyonmttok

from eole.constants import DefaultTokens
from eole.modules.mtp import MTPHead
from eole.config.models import TransformerDecoderConfig, TransformerLMModelConfig
from eole.models.model import DecoderModel
from eole.utils.statistics import Statistics


def _small_decoder_config():
    """Return a minimal TransformerDecoderConfig for unit tests."""
    return TransformerDecoderConfig(
        decoder_type="transformer",
        layers=2,
        hidden_size=32,
        heads=2,
        transformer_ff=64,
        num_mtp_heads=2,
        mtp_lambda=0.2,
    )


class _FakeGenerator(nn.Linear):
    """Tiny generator for CE-loss computation."""

    def __init__(self, hidden, vocab):
        super().__init__(hidden, vocab, bias=False)


class TestMTPHead(unittest.TestCase):
    def setUp(self):
        self.decoder_cfg = _small_decoder_config()
        self.head = MTPHead(self.decoder_cfg)

    def test_forward_shape(self):
        """MTPHead output must preserve (batch, seq_len, hidden)."""
        B, T, H = 2, 5, 32
        h = torch.randn(B, T, H)
        emb_k = torch.randn(B, T, H)
        out = self.head(h, emb_k)
        self.assertEqual(out.shape, (B, T, H))

    def test_no_grad_on_hidden(self):
        """Forward must work even when h is detached (no grad)."""
        B, T, H = 2, 5, 32
        h = torch.randn(B, T, H).detach()
        emb_k = torch.randn(B, T, H, requires_grad=True)
        out = self.head(h, emb_k)
        self.assertEqual(out.shape, (B, T, H))
        # Gradient must flow through emb_k
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(emb_k.grad)


class TestStatisticsMTP(unittest.TestCase):
    def test_mtp_loss_init(self):
        stat = Statistics()
        self.assertEqual(stat.mtp_loss, 0.0)

    def test_mtp_loss_update(self):
        s1 = Statistics(mtp_loss=1.5, mtp_ntokens=10)
        s2 = Statistics(mtp_loss=0.5, mtp_ntokens=5)
        s1.update(s2)
        self.assertAlmostEqual(s1.mtp_loss, 2.0)
        self.assertEqual(s1.mtp_ntokens, 15)

    def test_mtp_xent(self):
        stat = Statistics(mtp_loss=2.0, mtp_ntokens=4)
        self.assertAlmostEqual(stat.mtp_xent(), 0.5)

    def test_mtp_xent_zero_tokens(self):
        stat = Statistics(mtp_loss=1.0, mtp_ntokens=0)
        self.assertEqual(stat.mtp_xent(), 0.0)


class TestMTPLoss(unittest.TestCase):
    """Test that _compute_mtp_loss produces reasonable values."""

    def setUp(self):
        from eole.utils.loss import LossCompute

        vocab_size = 16
        pad_idx = 1
        gen = _FakeGenerator(32, vocab_size)
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="sum")
        vocabs = {
            "specials": {
                "pad_token": DefaultTokens.PAD,
                "unk_token": DefaultTokens.UNK,
                "eos_token": DefaultTokens.EOS,
            },
            "tgt": _make_tiny_vocab(pad_idx),
        }
        self.compute = LossCompute(
            criterion=criterion,
            generator=gen,
            tgt_shift_index=0,
            vocabs=vocabs,
            mtp_lambda=0.1,
        )

    def test_mtp_loss_positive(self):
        """MTP auxiliary loss should be positive for random inputs."""
        B, T, H = 2, 6, 32
        # mtp_outputs: 2 heads, each (B, T-1, H)
        mtp_outputs = [torch.randn(B, T - 1, H), torch.randn(B, T - 2, H)]
        tgt = torch.randint(2, 15, (B, T))
        batch = {"tgt": tgt}
        mtp_loss, raw_loss, n_mtp_tokens = self.compute._compute_mtp_loss(mtp_outputs, batch, 0)
        self.assertGreater(mtp_loss.item(), 0.0)
        self.assertGreater(raw_loss, 0.0)
        self.assertGreater(n_mtp_tokens, 0)

    def test_mtp_loss_zero_lambda(self):
        """With lambda=0 the MTP loss should not affect total loss."""
        from eole.utils.loss import LossCompute

        vocab_size = 16
        pad_idx = 1
        gen = _FakeGenerator(32, vocab_size)
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="sum")
        vocabs = {
            "specials": {
                "pad_token": DefaultTokens.PAD,
                "unk_token": DefaultTokens.UNK,
                "eos_token": DefaultTokens.EOS,
            },
            "tgt": _make_tiny_vocab(pad_idx),
        }
        compute = LossCompute(
            criterion=criterion,
            generator=gen,
            tgt_shift_index=0,
            vocabs=vocabs,
            mtp_lambda=0.0,
        )
        B, T, H = 2, 6, 32
        mtp_outputs = [torch.randn(B, T - 1, H)]
        tgt = torch.randint(2, 15, (B, T))
        batch = {"tgt": tgt}
        mtp_loss, _, _ = compute._compute_mtp_loss(mtp_outputs, batch, 0)
        # With lambda=0 the mtp_loss should be zero
        self.assertAlmostEqual(mtp_loss.item(), 0.0, places=5)


def _make_tiny_vocab(pad_idx, extra_tokens=0):
    """Build a minimal pyonmttok vocab with DefaultTokens specials."""
    tokens = Counter({f"tok{i}": 1 for i in range(extra_tokens)})
    vocab = pyonmttok.build_vocab_from_tokens(
        tokens,
        maximum_size=0,
        minimum_frequency=1,
        special_tokens=[
            DefaultTokens.UNK,
            DefaultTokens.PAD,
            DefaultTokens.BOS,
            DefaultTokens.EOS,
        ],
    )
    return vocab


class TestDecoderModelMTP(unittest.TestCase):
    """Integration test: build a real DecoderModel with MTP heads and run
    its training forward pass, checking per-head output shapes."""

    def _build_model_and_vocabs(self, num_mtp_heads=2):
        pad_idx = 1
        vocab = _make_tiny_vocab(pad_idx, extra_tokens=16)
        vocabs = {
            "tgt": vocab,
            "specials": {
                "pad_token": DefaultTokens.PAD,
                "unk_token": DefaultTokens.UNK,
                "eos_token": DefaultTokens.EOS,
                "bos_token": DefaultTokens.BOS,
            },
        }
        model_config = TransformerLMModelConfig(
            hidden_size=16,
            embeddings={"tgt_word_vec_size": 16},
            decoder={
                "decoder_type": "transformer",
                "layers": 2,
                "heads": 2,
                "hidden_size": 16,
                "transformer_ff": 32,
                "num_mtp_heads": num_mtp_heads,
                "mtp_lambda": 0.1,
            },
        )
        model = DecoderModel.build_blocks(model_config, vocabs, running_config=None)
        return model, vocabs

    def test_build_and_training_forward_shapes(self):
        model, vocabs = self._build_model_and_vocabs(num_mtp_heads=2)
        self.assertEqual(len(model.mtp_heads), 2)
        model.train()

        B, T = 2, 6
        pad_idx = vocabs["tgt"][DefaultTokens.PAD]
        src = torch.randint(2, 10, (B, T))
        src_len = torch.full((B,), T, dtype=torch.long)

        output = model(src, None, src_len)

        self.assertIsNotNone(output.mtp_outputs)
        self.assertEqual(len(output.mtp_outputs), 2)
        # Head k output should have length (T - k) and hidden dim 16.
        for k, mtp_out in enumerate(output.mtp_outputs, start=1):
            self.assertEqual(mtp_out.shape[0], B)
            self.assertEqual(mtp_out.shape[1], T - k)
            self.assertEqual(mtp_out.shape[2], 16)
        self.assertEqual(pad_idx, model.pad_idx)

    def test_no_mtp_outputs_in_eval_mode(self):
        model, _ = self._build_model_and_vocabs(num_mtp_heads=2)
        model.eval()
        B, T = 2, 6
        src = torch.randint(2, 10, (B, T))
        src_len = torch.full((B,), T, dtype=torch.long)
        output = model(src, None, src_len)
        self.assertIsNone(output.mtp_outputs)

    def test_update_dropout_propagates_to_mtp_heads(self):
        model, _ = self._build_model_and_vocabs(num_mtp_heads=2)
        model.update_dropout(0.37, 0.42)
        for head in model.mtp_heads:
            self.assertAlmostEqual(head.layer.dropout.p, 0.37)


if __name__ == "__main__":
    unittest.main()
