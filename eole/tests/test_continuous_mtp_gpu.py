"""Batched MTP CUDA numerics: row-specific commits, native drafting and lifecycle."""

import unittest
from types import MethodType

import torch

from eole.modules.gdn.gated_delta_net import GatedDeltaNet
from eole.models.model import MTPDraftingMixin
from eole.tests import test_continuous_gpu as gpu_helpers


@unittest.skipUnless(torch.cuda.is_available(), "CUDA MTP integration")
class ContinuousMTPGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gpu_helpers.ContinuousDecoderGpuTests.setUpClass()

    def setUp(self):
        self.tools = gpu_helpers.ContinuousDecoderGpuTests()

    @torch.inference_mode()
    def solo(self, model, ids, count):
        cache, hidden = self.tools.prefill(model, ids)
        result = []
        for _ in range(count):
            token = int(model.generator(hidden[:, -1]).log_softmax(-1).argmax(-1).item())
            result.append(token)
            hidden = self.tools.ordinary_step(model, cache, token)
        return result

    @torch.inference_mode()
    def test_different_gdn_commit_lengths_match_independent_prefixes(self):
        for counts in ([0, 1, 4], [1, 2, 3], [4, 0, 4], [0, 0, 0]):
            with self.subTest(counts=counts):
                model, runner = self.tools.fixture(gdn=True, mtp=True)
                try:
                    refs = [self.tools.prefill(model, ids)[0] for ids in ([3, 4], [5, 6, 7], [8])]
                    for slot, cache in enumerate(refs):
                        # Source caches allocated by the independent ordinary decoder need padding for admission.
                        runner._pad_cache(cache)
                        runner.bank.copy_row(slot, cache)
                    program = runner.programs[0]
                    runner._bind(program)
                    inputs = torch.tensor([[10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21]], device="cuda")
                    for layer in runner.linear_layers:
                        layer.begin_speculation(4)
                    model.decoder._speculative_forward = True
                    try:
                        runner._verify_step(
                            model.tgt_emb(inputs), tgt_pad_mask=torch.zeros(3, 1, 4, device="cuda", dtype=torch.bool)
                        )
                    finally:
                        model.decoder._speculative_forward = False
                        for layer in runner.linear_layers:
                            layer.end_speculation()
                    GatedDeltaNet.commit_speculation_group(runner.linear_layers, counts)
                    runner.bank.metadata["cache_seqlens"].sub_(
                        4 - torch.tensor(counts, device="cuda", dtype=torch.int32)
                    )
                    for slot, (cache, count) in enumerate(zip(refs, counts)):
                        for token in inputs[slot, :count].tolist():
                            self.tools.ordinary_step(model, cache, token)
                        self.tools.assert_state(runner.bank, cache, slot, atol=0.04)
                finally:
                    runner.close()

    @torch.inference_mode()
    def test_native_recurrent_drafts_match_standalone_mtp(self):
        model, runner = self.tools.fixture(gdn=True, mtp=True)
        for name in ("init_mtp_cache", "clear_mtp_cache", "draft_mtp_tokens"):
            setattr(model, name, MethodType(getattr(MTPDraftingMixin, name), model))
        try:
            runner.prefill_idle_chunk_size = 64
            ids = [3, 4, 5, 6, 7]
            _, hidden = self.tools.prefill(model, ids)
            request = runner.submit("draft", ids, 8)
            runner.tick()
            slot = request.slot
            program = runner.programs[slot + 1]
            runner._bind(program)
            actual, keep = runner._draft_step(program.mtp.hidden, tgt_pad_mask=None)
            actual = actual.clone()
            actual_keys = program.mtp.keys[:, : len(ids) + 2].clone()
            model.init_mtp_cache(hidden, torch.tensor([ids], device="cuda"), 8)
            expected = model.draft_mtp_tokens(hidden[:, -1:], runner.next_tokens[slot : slot + 1], len(ids) - 1, 3)
            self.assertEqual(actual.tolist(), torch.cat(expected, 1).tolist())
            self.assertEqual(keep.tolist(), [len(ids)])
            torch.testing.assert_close(
                actual_keys, model._mtp_cache_attention.kcache[:, : len(ids) + 2], rtol=0.02, atol=0.02
            )
        finally:
            model.clear_mtp_cache()
            runner.close()

    def test_dynamic_acceptance_cancellation_and_reuse(self):
        from eole import EOLE_TORCH_COMPILE

        for gdn in (False, True):
            for compiled in (False, True):
                if compiled and not EOLE_TORCH_COMPILE:
                    continue
                with self.subTest(gdn=gdn, compiled=compiled), torch.inference_mode():
                    model, runner = self.tools.fixture(gdn=gdn, mtp=True, compiled=compiled)
                    prompts = {"zero": [3, 4], "one": [5, 6, 7], "full": [8, 9, 10, 11], "reuse": [12]}
                    expected = {key: self.solo(model, ids, 20) for key, ids in prompts.items()}
                    desired = {"zero": 0, "one": 1, "full": 3, "reuse": 2}

                    # A controlled drafter forces simultaneous partial/full acceptance.
                    # The verifier and all cache operations still use real model kernels.
                    def proposals(hidden, *, tgt_pad_mask):
                        slots = list(runner.active) if hidden.size(0) == 1 else range(runner.slots)
                        result = torch.zeros(hidden.size(0), 3, device="cuda", dtype=torch.long)
                        for row, slot in enumerate(slots):
                            request = runner.active.get(slot)
                            if request is None:
                                continue
                            offset = len(request.tokens)
                            for i in range(3):
                                correct = expected[request.request_id][offset + i]
                                result[row, i] = correct if i < desired[request.request_id] else (correct + 1) % 32
                        return result, runner._draft_lengths + runner._draft_counts

                    # Preserve the production compiled target verifier; replace drafting only.
                    runner._draft_step = proposals
                    for program in runner.programs:
                        if program.drafter is not None:
                            program.drafter.reset()
                            program.drafter = None
                    try:
                        requests = {key: runner.submit(key, prompts[key], 12) for key in ("zero", "one", "full")}
                        reused = False
                        for _ in range(100):
                            if not runner.has_work:
                                break
                            runner.tick()
                            for key, request in requests.items():
                                if not gdn:
                                    self.assertEqual(request.tokens, expected[key][: len(request.tokens)], key)
                            if len(requests["one"].tokens) >= 3 and not reused:
                                runner.cancel("one")
                                requests["reuse"] = runner.submit("reuse", prompts["reuse"], 7)
                                reused = True
                        if gdn:
                            # Multi-token BF16 convolution/recurrent kernels can
                            # change argmax at near ties. Check every emitted
                            # token against solo logits under its actual prefix.
                            for key, request in requests.items():
                                cache, hidden = self.tools.prefill(model, prompts[key])
                                for token in request.tokens:
                                    logits = model.generator(hidden[:, -1]).float()
                                    self.assertLessEqual(float(logits.max() - logits[0, token]), 0.08, key)
                                    hidden = self.tools.ordinary_step(model, cache, token)
                        self.assertTrue(reused)
                        self.assertFalse(runner.has_work)
                        self.assertEqual(requests["one"].status, "cancelled")
                        self.assertEqual(requests["full"].status, "length")
                        self.assertGreater(requests["full"].accepted, requests["zero"].accepted)
                        self.assertEqual(len(requests["reuse"].tokens), 7)
                    finally:
                        runner.close()

    @torch.inference_mode()
    def test_native_mtp_stops_at_eos_without_emitting_remaining_proposals(self):
        for gdn in (False, True):
            with self.subTest(gdn=gdn):
                model, runner = self.tools.fixture(gdn=gdn, mtp=True)
                try:
                    prompt = [3, 4]
                    expected = self.solo(model, prompt, 6)
                    eos = expected[2]
                    runner.eos = {eos}
                    request = runner.submit("eos", prompt, 12)
                    while runner.has_work:
                        runner.tick()
                    self.assertEqual(request.status, "stop")
                    self.assertEqual(request.tokens, expected[: expected.index(eos) + 1])
                    self.assertFalse(runner.active)
                finally:
                    runner.close()

    @torch.inference_mode()
    def test_native_mtp_budget_and_physical_cache_boundary(self):
        for gdn in (False, True):
            for budget in (1, 2, 4):
                with self.subTest(gdn=gdn, budget=budget):
                    model, runner = self.tools.fixture(gdn=gdn, mtp=True)
                    try:
                        # Model logical context is 64; verifier scratch must not reduce it.
                        ids = [3, 4, 5] * 20
                        request = runner.submit("edge", ids, budget)
                        for _ in range(100):
                            if not runner.has_work:
                                break
                            runner.tick()
                        self.assertFalse(runner.has_work)
                        self.assertEqual(len(request.tokens), budget)
                        self.assertEqual(request.status, "length")
                    finally:
                        runner.close()
