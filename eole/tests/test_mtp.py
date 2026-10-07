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
from eole.config.inference import InferenceConfig
from eole.config.models import TransformerDecoderConfig, TransformerLMModelConfig
from eole.config.training import TrainingConfig
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

    def test_qwen35_fused_projection_matches_checkpoint_layout(self):
        decoder_cfg = _small_decoder_config()
        decoder_cfg.mtp_emb_norm = True
        head = MTPHead(decoder_cfg)
        hidden_size = decoder_cfg.hidden_size
        self.assertEqual(head.proj.weight.shape, (hidden_size, hidden_size * 2))

        class IdentityLayer(nn.Module):
            def forward(self, inputs, **kwargs):
                return inputs, None

        head.layer = IdentityLayer()
        hidden = torch.randn(2, 3, hidden_size)
        embedding = torch.randn(2, 3, hidden_size)
        expected = head.norm(head.proj(torch.cat([head.emb_norm(embedding), head.enorm(hidden)], dim=-1)))
        actual = head(hidden, embedding)
        torch.testing.assert_close(actual, expected)

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

    def test_inference_builds_and_uses_mtp_heads(self):
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
                "num_mtp_heads": 2,
                "mtp_lambda": 0.1,
            },
        )
        model = DecoderModel.build_blocks(model_config, vocabs, running_config=InferenceConfig())
        model.build_generator(model_config, InferenceConfig(), vocabs)
        # Construction uses skip_init; these tests do not load a checkpoint.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(3431)
            model.init_weights(TrainingConfig(param_init_method="uniform", param_init=0.1))
        model.eval()

        self.assertEqual(len(model.mtp_heads), 2)
        hidden = torch.randn(1, 1, 16)
        seed = torch.randint(2, len(vocab), (1, 1))
        drafts = model.draft_mtp_tokens(hidden, seed, position=0)
        self.assertEqual(len(drafts), 2)
        self.assertEqual([tuple(token.shape) for token in drafts], [(1, 1), (1, 1)])

    def test_qwen_recurrent_drafting_grows_causal_prefix(self):
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
                "num_mtp_heads": 1,
                "mtp_emb_norm": True,
            },
        )
        model = DecoderModel.build_blocks(model_config, vocabs, running_config=InferenceConfig())
        model.build_generator(model_config, InferenceConfig(), vocabs)
        # Construction uses skip_init; these tests do not load a checkpoint.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(3431)
            model.init_weights(TrainingConfig(param_init_method="uniform", param_init=0.1))
        model.eval()

        hidden_context = torch.randn(1, 4, 16)
        input_ids = torch.randint(2, len(vocab), (1, 4))
        hidden = hidden_context[:, -1:]
        seed = torch.randint(2, len(vocab), (1, 1))
        head = model.mtp_heads[0]
        rope = model.decoder.rope
        hidden_prefix = [hidden_context]
        token_prefix = [torch.cat([input_ids[:, 1:], seed], dim=1)]
        expected = []
        for index in range(3):
            seq_len = 4 + index
            pos_emb = rope.cos_sin[:seq_len] if rope is not None and rope.cos_sin is not None else None
            full_out = head(
                torch.cat(hidden_prefix, dim=1),
                model.tgt_emb.embeddings(torch.cat(token_prefix, dim=1)),
                attn_mask=torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))[None, None],
                position_embeddings=pos_emb,
            )
            next_hidden = full_out[:, -1:, :]
            next_token = model.generator(next_hidden[:, 0, :]).argmax(dim=-1, keepdim=True)
            expected.append(next_token)
            hidden_prefix.append(next_hidden)
            token_prefix.append(next_token)

        model.init_mtp_cache(hidden_context, input_ids, max_new_tokens=3)

        seen = []

        def capture_prefix(module, args, kwargs):
            mask = kwargs.get("attn_mask")
            seen.append((args[0].shape[1], mask.shape, int(mask.sum().item())))

        hook = model.mtp_heads[0].layer.register_forward_pre_hook(capture_prefix, with_kwargs=True)
        try:
            drafts = model.draft_mtp_tokens(hidden, seed, position=3, max_tokens=3)
        finally:
            hook.remove()

        self.assertEqual(len(drafts), 3)
        self.assertTrue(all(torch.equal(actual, ref) for actual, ref in zip(drafts, expected)))
        self.assertEqual(seen, [(1, (1, 1, 1, 8), 4), (1, (1, 1, 1, 8), 5), (1, (1, 1, 1, 8), 6)])
        model.clear_mtp_cache()

        # Compare several proposal cycles to a full causal-prefix reference.
        # The first pass must replace recurrent draft states with target
        # states, including the bonus input after full acceptance.
        model.init_mtp_cache(hidden_context, input_ids, max_new_tokens=40)
        prefix_hidden = hidden_context[:, :-1]
        prefix_tokens = input_ids[:, 1:]
        fresh_hidden, fresh_tokens = hidden, seed
        first_position = prefix_hidden.size(1)
        with torch.no_grad():
            for accepted_inputs in (4, 1, 3, 4, 2):
                absolute_position = first_position + fresh_hidden.size(1) - 1
                cached_drafts = model.draft_mtp_tokens(
                    fresh_hidden[:, -1:], fresh_tokens[:, -1:], absolute_position, max_tokens=3
                )
                reference_hidden = torch.cat([prefix_hidden, fresh_hidden], dim=1)
                reference_tokens = torch.cat([prefix_tokens, fresh_tokens], dim=1)
                target_prefix_hidden, target_prefix_tokens = reference_hidden, reference_tokens
                saved_cache = head.layer.self_attn.kcache, head.layer.self_attn.vcache
                head.layer.self_attn.kcache = head.layer.self_attn.vcache = None
                try:
                    reference_drafts = []
                    for draft_index in range(3):
                        length = reference_hidden.size(1)
                        output = head(
                            reference_hidden,
                            model.tgt_emb.embeddings(reference_tokens),
                            attn_mask=torch.tril(torch.ones(length, length, dtype=torch.bool))[None, None],
                            position_embeddings=rope.cos_sin[:length] if rope.cos_sin is not None else None,
                        )
                        next_hidden = output[:, -1:]
                        next_token = model.generator(next_hidden[:, 0]).argmax(-1, keepdim=True)
                        reference_drafts.append(next_token)
                        reference_hidden = torch.cat([reference_hidden, next_hidden], dim=1)
                        reference_tokens = torch.cat([reference_tokens, next_token], dim=1)
                finally:
                    head.layer.self_attn.kcache, head.layer.self_attn.vcache = saved_cache
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(cached_drafts, reference_drafts)))
                model.commit_mtp_draft(accepted_inputs)
                prefix_hidden, prefix_tokens = target_prefix_hidden, target_prefix_tokens
                self.assertEqual(model._mtp_cache_len, prefix_hidden.size(1))
                first_position = prefix_hidden.size(1)
                fresh_hidden = torch.randn(1, accepted_inputs, 16)
                correction = torch.randint(2, len(vocab), (1, 1))
                fresh_tokens = torch.cat(cached_drafts[: accepted_inputs - 1] + [correction], dim=1)
                model.set_mtp_context(fresh_hidden, fresh_tokens, first_position)
        model.clear_mtp_cache()
        self.assertIsNone(model._mtp_refresh)

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
