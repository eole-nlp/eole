import copy
import unittest
from unittest.mock import patch
from collections import Counter
from types import SimpleNamespace

import pyonmttok
import torch
import torch.nn as nn

from eole.config.models import CustomModelConfig
from eole.constants import DefaultTokens
from eole.models.model import DecoderModel
from eole.predict.generator import GeneratorLM
from eole.decoders import transformer


def _build_model(seed, gated_delta=False, qwen_mtp=False):
    torch.manual_seed(seed)
    hidden = 16
    decoder = {
        "decoder_type": "transformer",
        "layers": 2,
        "hidden_size": hidden,
        "heads": 2,
        "transformer_ff": 32,
        "num_mtp_heads": 1,
        "mtp_lambda": 0.1,
        "mtp_emb_norm": qwen_mtp,
        **(
            {
                "layer_types": ["linear_attention", "full_attention"],
                "linear_conv_kernel_dim": 3,
                "linear_key_head_dim": 4,
                "linear_value_head_dim": 4,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 2,
            }
            if gated_delta
            else {}
        ),
    }
    config = CustomModelConfig(
        decoder=decoder,
        embeddings={"tgt_word_vec_size": hidden, "src_word_vec_size": hidden},
    )
    vocab = pyonmttok.build_vocab_from_tokens(
        Counter(),
        maximum_size=0,
        minimum_frequency=1,
        special_tokens=[DefaultTokens.UNK, DefaultTokens.PAD, DefaultTokens.BOS, DefaultTokens.EOS]
        + ["a", "b", "c", "d", "e", "f"],
    )
    vocabs = {
        "src": vocab,
        "tgt": vocab,
        "specials": {
            "bos_token": DefaultTokens.BOS,
            "pad_token": DefaultTokens.PAD,
            "eos_token": DefaultTokens.EOS,
            "unk_token": DefaultTokens.UNK,
        },
        "decoder_start_token": DefaultTokens.BOS,
    }
    running_config = SimpleNamespace(
        dropout=[0.0],
        optim=None,
        self_attn_backend="pytorch",
        dynamic_shapes=False,
        param_init_method="xavier_uniform",
        param_init=0.1,
        lora_embedding=False,
    )
    model = DecoderModel.build_blocks(config, vocabs, running_config=running_config)
    model.init_weights(running_config)
    model.build_generator(config, running_config, vocabs)
    model.generator = nn.Linear(hidden, len(vocab), bias=True)
    # Keep fixtures running to max_length without using min_length, which
    # disables speculation and otherwise silently tests only the fallback.
    nn.init.zeros_(model.generator.bias)
    with torch.no_grad():
        model.generator.bias[vocab.lookup_token(DefaultTokens.EOS)] = -100
    model.eval()
    return model, vocab


def _generator(model, vocab, speculative):
    gen = GeneratorLM.__new__(GeneratorLM)
    gen.model = model
    gen._tgt_vocab = vocab
    gen._tgt_pad_idx = vocab.lookup_token(DefaultTokens.PAD)
    gen._tgt_bos_idx = vocab.lookup_token(DefaultTokens.BOS)
    gen._tgt_eos_idx = [vocab.lookup_token(DefaultTokens.EOS)]
    gen._tgt_unk_idx = vocab.lookup_token(DefaultTokens.UNK)
    gen._src_pad_idx = gen._tgt_pad_idx
    gen._tgt_start_with = gen._tgt_bos_idx
    gen.n_best = 1
    gen.global_scorer = SimpleNamespace(
        has_cov_pen=False,
        alpha=0.0,
        length_penalty=lambda step, alpha: 1.0,
    )
    gen.min_length = 0
    gen.block_ngram_repeat = 0
    gen._exclusion_idxs = set()
    gen.replace_unk = False
    gen.temperature = 1.0
    gen.top_k = 1
    gen.top_p = 0.0
    gen.beam_size = 1
    gen.ban_unk_token = False
    gen.add_estimator = False
    gen.dump_beam = ""
    gen.stepwise_penalty = False
    gen.ratio = -0.0
    gen.self_speculative_decoding = speculative
    gen.self_speculative_num_tokens = 4
    gen._speculative_drafted_by_position = [0] * gen.self_speculative_num_tokens
    gen._speculative_accepted_by_position = [0] * gen.self_speculative_num_tokens
    gen.max_length = 9
    gen.estim_only = False
    gen.dynamic_shapes = False
    gen.report_time = False
    gen.report_align = False
    gen.tgt_file_prefix = False
    gen.context_length = 64
    gen.n_best = 1
    gen.ignore_when_blocking = set()
    gen._log = lambda message: None
    return gen


class TestSpeculativeDecoding(unittest.TestCase):
    def _compare(self, gated_delta, dynamic_cache=False, qwen_mtp=False):
        model_plain, vocab = _build_model(123, gated_delta=gated_delta, qwen_mtp=qwen_mtp)
        model_spec = copy.deepcopy(model_plain)
        if dynamic_cache:
            # Start with a KV cache sized to the prompt, as eager decoding
            # does, so the first speculative chunk must reserve its full span.
            model_spec.decoder.dynamic_shapes = True
        prompt = torch.tensor([[4, 5, 6, 7]])
        batch = {"srclen": torch.tensor([4]), "src": prompt, "left_pad": True}

        with torch.no_grad():
            plain = _generator(model_plain, vocab, False).predict_batch(batch, attn_debug=False)
            speculative_generator = _generator(model_spec, vocab, True)
            speculative = speculative_generator.predict_batch(batch, attn_debug=False)

        self.assertGreater(speculative_generator._speculative_drafted_tokens, 0)
        self.assertTrue(torch.equal(plain["predictions"][0][0], speculative["predictions"][0][0]))

    def test_matches_normal_greedy(self):
        self._compare(gated_delta=False)

    def test_matches_normal_greedy_with_gated_delta_net(self):
        self._compare(gated_delta=True)

    def test_qwen_refresh_matches_normal_greedy(self):
        self._compare(gated_delta=False, qwen_mtp=True)

    def test_qwen_refresh_matches_normal_greedy_with_gdn(self):
        self._compare(gated_delta=True, qwen_mtp=True)

    def test_matches_normal_greedy_with_dynamic_cache(self):
        self._compare(gated_delta=False, dynamic_cache=True)

    def test_failed_request_restores_mtp_and_decoder_caches(self):
        model, vocab = _build_model(456, gated_delta=True, qwen_mtp=True)
        gen = _generator(model, vocab, speculative=True)
        prompt = torch.tensor([[4, 5, 6, 7]])
        batch = {"srclen": torch.tensor([4]), "src": prompt, "left_pad": True}
        attention = model.mtp_heads[0].layer.self_attn
        old_cache = attention.kcache, attention.vcache
        with torch.no_grad(), patch.object(gen, "_speculative_draft_verify", side_effect=RuntimeError("draft failed")):
            with self.assertRaisesRegex(RuntimeError, "draft failed"):
                gen.predict_batch(batch, attn_debug=False)
        self.assertIs(attention.kcache, old_cache[0])
        self.assertIs(attention.vcache, old_cache[1])
        self.assertIsNone(model._mtp_refresh)
        self.assertIsNone(model.decoder.cache_seqlens)
        self.assertFalse(model.decoder._speculative_forward)

    def test_streaming_speculative_tokens_match_prediction(self):
        model, vocab = _build_model(456)
        gen = _generator(model, vocab, speculative=True)
        prompt = torch.tensor([[4, 5, 6, 7]])
        batch = {"srclen": torch.tensor([4]), "src": prompt, "left_pad": True}
        tokens = []
        ended = []
        streamer = SimpleNamespace(put=lambda token: tokens.append(token.clone()), end=lambda: ended.append(True))
        with torch.no_grad():
            result = gen.predict_batch(batch, attn_debug=False, streamer=streamer)
        self.assertTrue(ended)
        self.assertTrue(torch.equal(torch.cat(tokens), result["predictions"][0][0]))

    def test_step0_timing_is_recorded_before_speculative_verification(self):
        model, vocab = _build_model(456)
        gen = _generator(model, vocab, speculative=True)
        gen.report_time = True
        gen.step0_time = []
        prompt = torch.tensor([[4, 5, 6, 7]])
        batch = {"srclen": torch.tensor([4]), "src": prompt, "left_pad": True}

        with torch.no_grad():
            gen.predict_batch(batch, attn_debug=False)

        self.assertEqual(len(gen.step0_time), 1)

    def test_compile_dispatches_multi_token_verifier_shape(self):
        model, _ = _build_model(321)
        decoder = model.decoder
        original_compile, original_enabled, original_mode = (
            getattr(decoder, "_forward_compile", None),
            transformer.EOLE_TORCH_COMPILE,
            transformer.EOLE_COMPILE_MODE,
        )
        called = []
        decoder._forward_compile = lambda emb, **kwargs: called.append(emb.size(1)) or (emb, {"std": None})
        decoder._speculative_forward = True
        transformer.EOLE_TORCH_COMPILE = True
        transformer.EOLE_COMPILE_MODE = "0"
        try:
            decoder(torch.zeros(1, 2, decoder.hidden_size), tgt_pad_mask=torch.zeros(1, 1, 2, dtype=torch.bool))
        finally:
            decoder._speculative_forward = False
            transformer.EOLE_TORCH_COMPILE = original_enabled
            transformer.EOLE_COMPILE_MODE = original_mode
            if original_compile is None:
                del decoder._forward_compile
            else:
                decoder._forward_compile = original_compile
        self.assertEqual(called, [2])


if __name__ == "__main__":
    unittest.main()
