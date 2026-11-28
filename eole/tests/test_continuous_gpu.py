"""Real CUDA decoder/admission regressions, without external checkpoints.

Run with EOLE_TORCH_COMPILE=1 EOLE_COMPILE_MODE=0 and CUDA available.
EOLE_REQUIRE_GPU_TESTS=1 fails instead of silently skipping an unprepared runner.
"""

import os
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from eole import EOLE_TORCH_COMPILE, EOLE_COMPILE_MODE
from eole.config.inference import InferenceConfig
from eole.config.models import TransformerDecoderConfig
from eole.constants import PositionEncodingType
from eole.decoders.transformer import TransformerDecoder
from eole.modules.mlp.embeddings import Embeddings
from eole.predict.continuous import ContinuousGreedy
from eole.utils.request_cache import RequestCache

REQUIRED = os.environ.get("EOLE_REQUIRE_GPU_TESTS") == "1"
if REQUIRED and (not torch.cuda.is_available() or not EOLE_TORCH_COMPILE or EOLE_COMPILE_MODE != "0"):
    raise RuntimeError("GPU CI requires CUDA, EOLE_TORCH_COMPILE=1 and EOLE_COMPILE_MODE=0")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA integration test")
class ContinuousDecoderGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from eole.ops import _CPP_OPS_AVAILABLE
        import flash_attn  # noqa: F401
        import fla  # noqa: F401

        if not _CPP_OPS_AVAILABLE:
            raise RuntimeError("GPU integration tests require the matching Eole CUDA extension")

    def fixture(self, gdn=False, compiled=False, mtp=False, prefix_cache_entries=0, **runner_options):
        torch.manual_seed(139)
        config = TransformerDecoderConfig(
            layers=2,
            hidden_size=128,
            heads=2,
            heads_kv=1,
            transformer_ff=256,
            with_cross_attn=False,
            max_position_embeddings=64,
            position_encoding_type=PositionEncodingType.Rotary,
            layer_types=["linear_attention", "full_attention"] if gdn else ["full_attention"] * 2,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=64,
            linear_value_head_dim=64,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
        )
        model = nn.Module()
        model.decoder = TransformerDecoder(
            config,
            running_config=InferenceConfig(
                self_attn_backend="flash",
                dynamic_shapes=False,
                context_length=64,
                prefill_cache_size=0,
            ),
        )
        model.tgt_emb = Embeddings(128, 32, 0, position_encoding_type=PositionEncodingType.Rotary)
        model.generator = nn.Linear(128, 32, bias=False)
        if mtp:
            from eole.modules.speculation.mtp import MTPHead

            model.mtp_heads = nn.ModuleList(
                [
                    MTPHead(
                        config.model_copy(update={"mtp_emb_norm": True}),
                        running_config=InferenceConfig(self_attn_backend="flash"),
                    )
                ]
            )
        # Eole's skip_init projections must be explicitly initialized in fixtures.
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if parameter.ndim >= 2:
                    parameter.normal_(0, 0.1)
                elif name.endswith("weight") and "norm" in name:
                    parameter.fill_(1)
                else:
                    parameter.zero_()
        model.to(device="cuda", dtype=torch.bfloat16).eval()
        predictor = SimpleNamespace(
            model=model,
            self_speculative_decoding=mtp,
            self_speculative_num_tokens=3,
            parallel_gpu=1,
            beam_size=1,
            n_best=1,
            top_k=1,
            _tgt_eos_idx=[],
            max_length=6,
        )
        factory = ContinuousGreedy
        if mtp:
            from eole.predict.continuous_mtp import ContinuousMTP

            factory = ContinuousMTP
        runner = factory(
            predictor,
            slots=3,
            prefill_chunk_size=2,
            compiled=compiled,
            prefix_cache_entries=prefix_cache_entries,
            **runner_options
        )
        return model, runner

    def prefill(self, model, ids):
        decoder = model.decoder
        decoder._disable_cache()
        ids = torch.tensor([ids], device="cuda")
        emb = model.tgt_emb(ids)
        mask = torch.zeros(1, 1, ids.size(1), dtype=torch.bool, device="cuda")
        decoder._init_cache(emb, mask)
        hidden, _ = decoder._forward_eager(emb, tgt_pad_mask=mask)
        return RequestCache.take(decoder), hidden.clone()

    def ordinary_step(self, model, cache, token):
        cache.bind()
        emb = model.tgt_emb(torch.tensor([[token]], device="cuda"), step=1)
        hidden, _ = model.decoder._forward_eager(
            emb, tgt_pad_mask=torch.zeros(1, 1, 1, dtype=torch.bool, device="cuda")
        )
        return hidden.clone()

    def assert_state(self, actual, expected, slot, atol=0.02):
        torch.testing.assert_close(
            actual.metadata["cache_seqlens"][slot : slot + 1], expected.metadata["cache_seqlens"], rtol=0, atol=0
        )
        length = int(expected.metadata["cache_seqlens"].item())
        for (_, bank), (_, reference) in zip(actual.layers, expected.layers):
            for name, tensor in bank.items():
                a, b = tensor[slot : slot + 1], reference[name]
                if name in ("kcache", "vcache"):
                    a, b = a[:, :length], b[:, :length]
                torch.testing.assert_close(a, b, rtol=0.03, atol=atol, msg=name)

    @torch.inference_mode()
    def check_numerics(self, gdn, compiled):
        model, runner = self.fixture(gdn, compiled)
        try:
            references = [self.prefill(model, ids)[0] for ids in ([3, 4], [5, 6, 7, 8, 9], [10, 11, 12])]
            for slot, cache in enumerate(references):
                runner.bank.copy_row(slot, cache)
            for step in range(4):
                tokens = [13 + step, 18 + step, 23 + step]
                runner.bank.bind()
                emb = model.tgt_emb(torch.tensor(tokens, device="cuda")[:, None], step=1)
                if compiled:
                    hidden, _ = runner.step_graph(
                        emb, tgt_pad_mask=torch.zeros(3, 1, 1, dtype=torch.bool, device="cuda")
                    )
                else:
                    hidden, _ = model.decoder.forward_step(emb)
                hidden = hidden.clone()
                logits = model.generator(hidden[:, -1]).clone()
                for slot, (cache, token) in enumerate(zip(references, tokens)):
                    expected = self.ordinary_step(model, cache, token)
                    torch.testing.assert_close(hidden[slot : slot + 1], expected, rtol=0.02, atol=0.02)
                    torch.testing.assert_close(
                        logits[slot : slot + 1], model.generator(expected[:, -1]), rtol=0.02, atol=0.02
                    )
                    self.assert_state(runner.bank, cache, slot)
                if step == 1:
                    # Reuse an occupied row with unrelated, shorter state. Neighbours must survive.
                    replacement = self.prefill(model, [27])[0]
                    runner.bank.copy_row(1, replacement)
                    references[1] = replacement
                    self.assertEqual(runner.bank.metadata["cache_seqlens"].tolist(), [4, 1, 5])
        finally:
            runner.close()

    def test_eager_dense_and_gdn_logits_and_state_match_independent_ordinary_steps(self):
        for gdn in (False, True):
            with self.subTest(gdn=gdn):
                self.check_numerics(gdn, False)

    @unittest.skipUnless(EOLE_TORCH_COMPILE and EOLE_COMPILE_MODE == "0", "Set compile mode 0 for graph test")
    def test_compiled_dense_and_gdn_logits_and_state_match_independent_ordinary_steps(self):
        for gdn in (False, True):
            with self.subTest(gdn=gdn):
                self.check_numerics(gdn, True)

    @torch.inference_mode()
    def test_scheduler_late_arrival_partial_cancel_and_slot_reuse_match_solo_tokens(self):
        for gdn in (False, True):
            with self.subTest(gdn=gdn):
                model, runner = self.fixture(gdn)
                prompts = {"first": [3, 4], "second": [5, 6, 7, 8, 9], "replacement": [10, 11, 12], "late": [17]}
                try:
                    first = runner.submit("first", prompts["first"], 5)
                    second = runner.submit("second", prompts["second"], 5)
                    for _ in range(40):
                        runner.tick()
                        if second.status == "prefilling":
                            break
                    self.assertEqual(second.status, "prefilling")
                    runner.cancel("second")
                    replacement = runner.submit("replacement", prompts["replacement"], 5)
                    late = None
                    for _ in range(80):
                        if first.status == "length" and late is None:
                            late = runner.submit("late", prompts["late"], 5)
                        if not runner.has_work:
                            break
                        runner.tick()
                    self.assertFalse(runner.has_work)
                    self.assertEqual(second.status, "cancelled")
                    self.assertIsNotNone(late)
                    for request in (first, replacement, late):
                        cache, hidden = self.prefill(model, prompts[request.request_id])
                        expected = []
                        for _ in range(5):
                            token = int(model.generator(hidden[:, -1]).log_softmax(-1).argmax(-1).item())
                            expected.append(token)
                            hidden = self.ordinary_step(model, cache, token)
                        self.assertEqual(request.tokens, expected)
                finally:
                    runner.close()

    def test_tick_owns_inference_context_even_when_caller_does_not(self):
        self.assertFalse(torch.is_inference_mode_enabled())
        model, runner = self.fixture()
        try:
            self.assertTrue(torch.is_inference(runner.next_tokens))
            request = runner.submit("context", [3, 4], 3)
            runner.tick()
            runner.tick()
            self.assertEqual(len(request.tokens), 2)
            self.assertFalse(torch.is_inference_mode_enabled())
        finally:
            runner.close()
