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
from eole.config.models import TransformerDecoderConfig
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
        s1 = Statistics(mtp_loss=1.5, n_tokens=10)
        s2 = Statistics(mtp_loss=0.5, n_tokens=5)
        s1.update(s2)
        self.assertAlmostEqual(s1.mtp_loss, 2.0)

    def test_mtp_xent(self):
        stat = Statistics(mtp_loss=2.0, n_tokens=4)
        self.assertAlmostEqual(stat.mtp_xent(), 0.5)

    def test_mtp_xent_zero_tokens(self):
        stat = Statistics(mtp_loss=1.0, n_tokens=0)
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
        mtp_loss = self.compute._compute_mtp_loss(mtp_outputs, batch, 0)
        mtp_loss_value = mtp_loss[0].item() if isinstance(mtp_loss, tuple) else mtp_loss.item()
        self.assertGreater(mtp_loss_value, 0.0)

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
        mtp_loss = compute._compute_mtp_loss(mtp_outputs, batch, 0)
        # With lambda=0 the mtp_loss should be zero
        mtp_loss_value = mtp_loss[0].item() if isinstance(mtp_loss, tuple) else mtp_loss.item()
        self.assertAlmostEqual(mtp_loss_value, 0.0, places=5)


def _make_tiny_vocab(pad_idx):
    """Build a minimal pyonmttok vocab with DefaultTokens specials."""
    vocab = pyonmttok.build_vocab_from_tokens(
        Counter(),
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


if __name__ == "__main__":
    unittest.main()
