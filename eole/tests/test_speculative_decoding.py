"""Tests for MTP-based self-speculative decoding.

These tests exercise:
* ``MTPDraftingMixin.draft_mtp_tokens`` (shape / chaining behaviour),
  shared by ``DecoderModel`` and ``VisionEncoderDecoderModel``.
* End-to-end equivalence between draft+verify+accept/reject speculative
  decoding and plain step-by-step greedy decoding, proving that the
  speculative path is a pure latency optimization (identical outputs).
* ``Inference._resolve_self_speculative_decoding`` gating, including model
  classes that don't mix in ``MTPDraftingMixin`` and therefore don't
  implement ``draft_mtp_tokens``.
"""

import copy
import unittest
from collections import Counter
from types import SimpleNamespace

import pyonmttok
import torch
import torch.nn as nn

from eole.config.models import CustomModelConfig
from eole.constants import DefaultTokens
from eole.models.model import DecoderModel, MTPDraftingMixin, VisionEncoderDecoderModel
from eole.predict.inference import Inference


def _build_model(seed, num_mtp_heads=2, hidden_size=32):
    """Build a tiny, fully-initialized DecoderModel with MTP heads."""
    torch.manual_seed(seed)
    model_config = CustomModelConfig(
        decoder={
            "decoder_type": "transformer",
            "layers": 2,
            "hidden_size": hidden_size,
            "heads": 2,
            "transformer_ff": 64,
            "num_mtp_heads": num_mtp_heads,
            "mtp_lambda": 0.1,
        },
        embeddings={
            "tgt_word_vec_size": hidden_size,
            "src_word_vec_size": hidden_size,
        },
    )
    tgt_vocab = pyonmttok.build_vocab_from_tokens(
        Counter(),
        maximum_size=0,
        minimum_frequency=1,
        special_tokens=[
            DefaultTokens.UNK,
            DefaultTokens.PAD,
            DefaultTokens.BOS,
            DefaultTokens.EOS,
        ]
        + ["a", "b", "c", "d", "e", "f"],
    )
    vocabs = {
        "src": tgt_vocab,
        "tgt": tgt_vocab,
        "specials": {
            "bos_token": DefaultTokens.BOS,
            "pad_token": DefaultTokens.PAD,
            "eos_token": DefaultTokens.EOS,
            "unk_token": DefaultTokens.UNK,
        },
    }
    running_config = SimpleNamespace(
        dropout=[0.0],
        optim=None,
        self_attn_backend="pytorch",
        dynamic_shapes=False,
        param_init_method="xavier_uniform",
        param_init=0.1,
    )
    model = DecoderModel.build_blocks(model_config, vocabs, running_config=running_config)
    model.init_weights(running_config)
    model.eval()
    vocab_size = len(tgt_vocab)
    torch.manual_seed(seed + 1)
    model.generator = nn.Linear(hidden_size, vocab_size, bias=False)
    return model, vocab_size


def _run_true_stepwise(model, prompt, n_extra):
    """Reference implementation: plain token-by-token greedy decoding."""
    pad_idx = model.pad_idx
    emb = model.tgt_emb(prompt, step=0)
    tgt_pad_mask = prompt.eq(pad_idx).unsqueeze(1)
    model.decoder.kvcache_maxsize = prompt.size(1) + n_extra + 1
    model.decoder._init_cache(emb, tgt_pad_mask)
    dec_out, _ = model.decoder(emb, tgt_pad_mask=tgt_pad_mask, step=0)
    logits = model.generator(dec_out[:, -1:, :].squeeze(1))
    cur = logits.argmax(-1, keepdim=True)
    seq = [cur]
    cur_pos = prompt.size(1) - 1
    for i in range(n_extra):
        emb_i = model.tgt_emb(cur, step=cur_pos + 1 + i)
        pad_i = cur.eq(pad_idx).unsqueeze(1)
        dec_out_i, _ = model.decoder(emb_i, tgt_pad_mask=pad_i, step=cur_pos + 1 + i)
        logits_i = model.generator(dec_out_i.squeeze(1))
        cur = logits_i.argmax(-1, keepdim=True)
        seq.append(cur)
    model.decoder._disable_cache()
    return torch.cat(seq, dim=1)


def _run_speculative(model, prompt, n_extra):
    """Draft (via MTP heads) + verify + accept/reject speculative decoding."""
    pad_idx = model.pad_idx
    emb = model.tgt_emb(prompt, step=0)
    tgt_pad_mask = prompt.eq(pad_idx).unsqueeze(1)
    model.decoder.kvcache_maxsize = prompt.size(1) + n_extra + 1
    model.decoder._init_cache(emb, tgt_pad_mask)
    dec_out, _ = model.decoder(emb, tgt_pad_mask=tgt_pad_mask, step=0)
    dec_out_last = dec_out[:, -1:, :]
    logits = model.generator(dec_out_last.squeeze(1))
    seed_token = logits.argmax(-1, keepdim=True)
    cur_pos = prompt.size(1) - 1

    seq = [seed_token]
    total = 0
    cur_dec_out = dec_out_last
    cur_seed = seed_token
    while total < n_extra:
        draft_tokens = model.draft_mtp_tokens(cur_dec_out, cur_seed, cur_pos)
        num_draft = min(len(draft_tokens), n_extra - total)
        draft_tokens = draft_tokens[:num_draft]
        draft_tensor = torch.cat(draft_tokens, dim=1)
        verify_input = torch.cat([cur_seed, draft_tensor], dim=1)
        pad_v = verify_input.eq(pad_idx).unsqueeze(1)
        emb_v = model.tgt_emb(verify_input, step=cur_pos + 1)
        dec_out_v, _ = model.decoder(emb_v, tgt_pad_mask=pad_v, step=cur_pos + 1)
        logits_v = model.generator(dec_out_v)
        predicted = logits_v.argmax(-1)

        accepted = 0
        while accepted < num_draft and torch.equal(predicted[:, accepted], draft_tokens[accepted].squeeze(1)):
            accepted += 1

        rollback = num_draft - accepted
        if rollback > 0:
            model.decoder.cache_seqlens -= rollback

        for i in range(accepted + 1):
            seq.append(predicted[:, i : i + 1])
        total += accepted + 1
        cur_pos = cur_pos + accepted + 1
        cur_seed = predicted[:, accepted : accepted + 1]
        cur_dec_out = dec_out_v[:, accepted : accepted + 1, :]
    model.decoder._disable_cache()
    return torch.cat(seq, dim=1)


class TestDraftMtpTokens(unittest.TestCase):
    def test_draft_mtp_tokens_shapes(self):
        model, vocab_size = _build_model(seed=0, num_mtp_heads=3, hidden_size=16)
        batch, hidden = 2, 16
        h_last = torch.randn(batch, 1, hidden)
        seed_token = torch.randint(4, vocab_size, (batch, 1))
        drafts = model.draft_mtp_tokens(h_last, seed_token, pos_id=5)
        self.assertEqual(len(drafts), 3)
        for d in drafts:
            self.assertEqual(d.shape, (batch, 1))

    def test_draft_mtp_tokens_no_grad(self):
        model, vocab_size = _build_model(seed=1, num_mtp_heads=2, hidden_size=16)
        h_last = torch.randn(2, 1, 16, requires_grad=True)
        seed_token = torch.randint(4, vocab_size, (2, 1))
        drafts = model.draft_mtp_tokens(h_last, seed_token, pos_id=2)
        for d in drafts:
            self.assertFalse(d.requires_grad)


class TestSpeculativeDecodingEquivalence(unittest.TestCase):
    """Speculative decoding must be bit-exact with plain greedy decoding."""

    def test_equivalence_multiple_seeds(self):
        for seed in range(10):
            model1, vocab_size = _build_model(seed=seed, num_mtp_heads=2, hidden_size=32)
            torch.manual_seed(seed + 500)
            prompt = torch.randint(4, vocab_size, (3, 4))
            model2 = copy.deepcopy(model1)

            n_extra = 10
            true_seq = _run_true_stepwise(model1, prompt, n_extra)
            spec_seq = _run_speculative(model2, prompt, n_extra)
            self.assertTrue(
                torch.equal(true_seq, spec_seq),
                f"Mismatch at seed {seed}: true={true_seq} spec={spec_seq}",
            )

    def test_equivalence_single_mtp_head(self):
        model1, vocab_size = _build_model(seed=7, num_mtp_heads=1, hidden_size=32)
        torch.manual_seed(42)
        prompt = torch.randint(4, vocab_size, (2, 3))
        model2 = copy.deepcopy(model1)

        n_extra = 8
        true_seq = _run_true_stepwise(model1, prompt, n_extra)
        spec_seq = _run_speculative(model2, prompt, n_extra)
        self.assertTrue(torch.equal(true_seq, spec_seq))

    def test_equivalence_batch_size_one(self):
        model1, vocab_size = _build_model(seed=3, num_mtp_heads=2, hidden_size=32)
        torch.manual_seed(11)
        prompt = torch.randint(4, vocab_size, (1, 5))
        model2 = copy.deepcopy(model1)

        n_extra = 6
        true_seq = _run_true_stepwise(model1, prompt, n_extra)
        spec_seq = _run_speculative(model2, prompt, n_extra)
        self.assertTrue(torch.equal(true_seq, spec_seq))


class TestResolveSelfSpeculativeDecoding(unittest.TestCase):
    """``Inference._resolve_self_speculative_decoding`` must gate on
    whether the model actually implements ``draft_mtp_tokens`` -- not just
    on whether it carries MTP heads -- since some (hypothetical, future)
    model classes could keep MTP heads around for checkpoint compatibility
    only, without mixing in ``MTPDraftingMixin``, and would otherwise crash
    with an ``AttributeError`` in the decode loop when self-speculative
    decoding is requested."""

    def _model(self, num_mtp_heads=2, with_drafting=True):
        attrs = {"mtp_heads": list(range(num_mtp_heads))}
        if with_drafting:
            attrs["draft_mtp_tokens"] = lambda *a, **k: []
        return SimpleNamespace(**attrs)

    def test_enabled_when_model_supports_drafting(self):
        enabled, reasons = Inference._resolve_self_speculative_decoding(
            self._model(with_drafting=True), True, beam_size=1, top_k=1, temperature=1.0
        )
        self.assertTrue(enabled)
        self.assertEqual(reasons, [])

    def test_disabled_without_crashing_when_model_lacks_drafting(self):
        """Models with MTP heads but no ``draft_mtp_tokens`` must be
        disabled, not crash."""
        enabled, reasons = Inference._resolve_self_speculative_decoding(
            self._model(with_drafting=False), True, beam_size=1, top_k=1, temperature=1.0
        )
        self.assertFalse(enabled)
        self.assertTrue(any("does not implement" in r for r in reasons))

    def test_disabled_when_no_mtp_heads(self):
        enabled, reasons = Inference._resolve_self_speculative_decoding(
            self._model(num_mtp_heads=0, with_drafting=True), True, beam_size=1, top_k=1, temperature=1.0
        )
        self.assertFalse(enabled)
        self.assertTrue(any("no MTP heads" in r for r in reasons))

    def test_disabled_when_beam_size_not_one(self):
        enabled, reasons = Inference._resolve_self_speculative_decoding(
            self._model(with_drafting=True), True, beam_size=4, top_k=1, temperature=1.0
        )
        self.assertFalse(enabled)
        self.assertTrue(any("beam_size" in r for r in reasons))

    def test_disabled_when_not_deterministic(self):
        enabled, reasons = Inference._resolve_self_speculative_decoding(
            self._model(with_drafting=True), True, beam_size=1, top_k=5, temperature=1.0
        )
        self.assertFalse(enabled)
        self.assertTrue(any("top_k" in r for r in reasons))

    def test_no_reasons_when_not_requested(self):
        """When the user never asked for it, there's nothing to warn about,
        even if the model/config wouldn't have supported it anyway."""
        enabled, reasons = Inference._resolve_self_speculative_decoding(
            self._model(num_mtp_heads=0, with_drafting=False), False, beam_size=4, top_k=5, temperature=1.0
        )
        self.assertFalse(enabled)
        self.assertEqual(reasons, [])


class TestVisionEncoderDecoderModelMtpDrafting(unittest.TestCase):
    """Vision-language models (``VisionEncoderDecoderModel``) must support
    self-speculative decoding exactly like decoder-only models: once past
    the (vision) prefill step, incremental decoding only consumes the
    previous token's embedding plus the KV cache -- identical to a
    decoder-only LM -- so the same ``draft_mtp_tokens`` implementation
    applies unchanged."""

    def test_decoder_and_vision_model_share_the_same_drafting_implementation(self):
        # Both classes must resolve `draft_mtp_tokens` to the exact same
        # function (from MTPDraftingMixin), not independent/divergent
        # reimplementations, so fixes/behaviour stay in sync.
        self.assertIs(DecoderModel.draft_mtp_tokens, MTPDraftingMixin.draft_mtp_tokens)
        self.assertIs(VisionEncoderDecoderModel.draft_mtp_tokens, MTPDraftingMixin.draft_mtp_tokens)

    def test_vision_model_passes_the_mtp_drafting_gate(self):
        # Duck-typed gate (Inference._resolve_self_speculative_decoding)
        # must now accept a VisionEncoderDecoderModel-shaped object.
        fake_vlm = SimpleNamespace(
            mtp_heads=[object(), object()],
            draft_mtp_tokens=VisionEncoderDecoderModel.draft_mtp_tokens,
        )
        enabled, reasons = Inference._resolve_self_speculative_decoding(
            fake_vlm, True, beam_size=1, top_k=1, temperature=1.0
        )
        self.assertTrue(enabled)
        self.assertEqual(reasons, [])


if __name__ == "__main__":
    unittest.main()
