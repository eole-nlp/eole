"""Real CUDA DFlash2 continuous serving: solo parity, rings, GDN commits, graphs."""

import unittest
from types import SimpleNamespace

import torch

from eole import EOLE_TORCH_COMPILE
from eole.modules.dflash2 import DFlash2Adapter
from eole.predict.continuous_dflash2 import ContinuousDFlash2
from eole.tests import test_continuous_gpu as gpu_helpers


def tiny_drafter(hidden=128, block_size=8, sliding_window=16):
    cfg = SimpleNamespace(
        architectures="DFlash2DraftModel",
        is_causal=False,
        hidden_act="silu",
        hidden_size=hidden,
        head_dim=8,
        num_attention_heads=8,
        num_key_value_heads=2,
        attention_bias=False,
        rms_norm_eps=1e-6,
        sliding_window=sliding_window,
        rope_parameters={"rope_theta": 10000000.0},
        vocab_size=32,
        num_hidden_layers=2,
        intermediate_size=64,
        num_target_layers=2,
        dflash_config={
            "selector_rank": 4,
            "selector_top_k": 3,
            "target_layer_ids": [0, 1],
            "block_size": block_size,
            "conv_group_size": 4,
            "conv_kernel_size": 2,
            "mask_token_id": 30,
        },
    )
    return cfg


@unittest.skipUnless(torch.cuda.is_available(), "CUDA DFlash2 continuous test")
class ContinuousDFlash2GpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gpu_helpers.ContinuousDecoderGpuTests.setUpClass()

    def setUp(self):
        self.tools = gpu_helpers.ContinuousDecoderGpuTests()

    @staticmethod
    def attach(model, adapter):
        # Mirror predictor construction: the drafter adapter rides on the
        # predictor; the decoder learns the feature taps from the demo runner.
        return SimpleNamespace(
            model=model,
            dflash2=adapter,
            self_speculative_decoding=True,
            self_speculative_num_tokens=adapter.block_size - 1,
            parallel_gpu=1,
            beam_size=1,
            n_best=1,
            top_k=1,
            _tgt_eos_idx=[],
        )

    @torch.inference_mode()
    def solo(self, model, ids, count):
        cache, hidden = self.tools.prefill(model, ids)
        result = []
        for _ in range(count):
            token = int(model.generator(hidden[:, -1]).log_softmax(-1).argmax(-1).item())
            result.append(token)
            hidden = self.tools.ordinary_step(model, cache, token)
        return result

    def _runner(self, compiled, gdn=False, prefix_cache_entries=0, **runner_options):
        model, discardable = self.tools.fixture(gdn=gdn, compiled=compiled, mtp=False)
        discardable.close()
        cfg = tiny_drafter()
        adapter_cls = DFlash2Adapter
        if EOLE_TORCH_COMPILE:
            from eole.modules.dflash2_graph import DFlash2GraphAdapter

            adapter_cls = DFlash2GraphAdapter
        drafter = adapter_cls(cfg).to(device="cuda", dtype=torch.bfloat16).eval()
        with torch.no_grad():
            for name, parameter in drafter.named_parameters():
                if parameter.ndim >= 2:
                    parameter.normal_(0, 0.1)
                elif name.endswith("weight") and "norm" in name:
                    parameter.fill_(1)
                else:
                    parameter.zero_()
        predictor = self.attach(model, drafter)
        runner = ContinuousDFlash2(
            predictor,
            slots=3,
            prefill_chunk_size=2,
            compiled=compiled,
            prefix_cache_entries=prefix_cache_entries,
            **runner_options,
        )
        return model, runner

    @torch.inference_mode()
    def test_emitted_stream_matches_solo_greedy_with_ring_wraparound(self):
        for compiled in (False, True):
            if compiled and not EOLE_TORCH_COMPILE:
                continue
            with self.subTest(compiled=compiled):
                model, runner = self._runner(compiled)
                try:
                    prompts = {"short": [3, 4], "window": list(range(5, 21)), "over": list(range(2, 20))[:15]}
                    expected = {key: self.solo(model, ids, 9) for key, ids in prompts.items()}
                    requests = {key: runner.submit(key, ids, 9) for key, ids in prompts.items()}
                    while runner.has_work:
                        runner.tick()
                    for key, request in requests.items():
                        self.assertEqual(request.tokens, expected[key], key)
                        self.assertEqual(request.status, "length")
                finally:
                    runner.close()

    @torch.inference_mode()
    def test_cached_prompt_consumed_only_for_exact_extension(self):
        for mode in ("greedy", "mtp", "dflash2"):
            with self.subTest(mode=mode):
                if mode == "dflash2":
                    _, runner = self._runner(compiled=False, gdn=True, prefix_cache_entries=1)
                else:
                    _, runner = self.tools.fixture(gdn=True, mtp=mode == "mtp", prefix_cache_entries=1)
                self._check_cached_prompt_extension(runner)

    def _check_cached_prompt_extension(self, runner):
        prefix = list(range(3, 23))
        extended = prefix + [23, 24]
        try:
            first = runner.submit("first", prefix, 3)
            while runner.has_work:
                runner.tick()
            self.assertEqual(first.reused_prefix_tokens, 0)
            hit = runner.submit("hit", extended, 6)
            while runner.has_work:
                runner.tick()
            self.assertEqual(hit.reused_prefix_tokens, len(prefix))
            self.assertEqual(hit.prefill_chunks, [2])
            self.assertEqual(list(runner.prompt_cache), [tuple(extended)])
            cold = runner.submit("cold", extended, 6)
            while runner.has_work:
                runner.tick()
            self.assertEqual(cold.reused_prefix_tokens, 0)
            self.assertEqual(cold.tokens, hit.tokens)
            different = [2] + extended[1:] + [25, 26]
            miss = runner.submit("miss", different, 1)
            while runner.has_work:
                runner.tick()
            self.assertEqual(miss.reused_prefix_tokens, 0)
            self.assertEqual(list(runner.prompt_cache), [tuple(different)])
        finally:
            runner.close()

    @torch.inference_mode()
    def test_gdn_commit_and_host_seqlens_stay_consistent(self):
        if not EOLE_TORCH_COMPILE:
            self.skipTest("compiled path requires EOLE_TORCH_COMPILE=1")
        model, runner = self._runner(compiled=True, gdn=True)
        try:
            self.assertTrue(all(program.verifier.graph is not None for program in runner.programs))
            ids = list(range(3, 15))
            request = runner.submit("gdn", ids, 7)
            for _ in range(100):
                if not runner.has_work:
                    break
                runner.tick()
                if request.status == "running":
                    self.assertEqual(runner.host_seqlens[request.slot], request.prefilled + len(request.tokens) - 1)
            # Multi-token BF16 convolution/recurrent kernels can change argmax
            # at near ties; check every emitted token against solo logits under
            # its actual prefix (same standard as the continuous MTP suite).
            request = runner.requests["gdn"]
            self.assertEqual(len(request.tokens), 7)
            cache, hidden = self.tools.prefill(model, ids)
            for token in request.tokens:
                logits = model.generator(hidden[:, -1]).float()
                self.assertLessEqual(float(logits.max() - logits[0, token]), 0.08)
                hidden = self.tools.ordinary_step(model, cache, token)
            self.assertFalse(runner.has_work)
        finally:
            runner.close()

    @torch.inference_mode()
    def test_all_slots_run_together_and_drift_none(self):
        model, runner = self._runner(compiled=bool(EOLE_TORCH_COMPILE))
        try:
            if EOLE_TORCH_COMPILE:
                self.assertIsNotNone(runner.draft_bank._graph.graph)
                for slot, adapter in enumerate(runner.adapters):
                    positions, _ = runner.draft_bank.row_ring(slot)
                    self.assertEqual(adapter._context_positions.data_ptr(), positions.data_ptr())
            prompts = {f"row{i}": [10 + 3 * i, 11 + 3 * i, 12 + 3 * i] for i in range(3)}
            expected = {key: self.solo(model, ids, 6) for key, ids in prompts.items()}
            requests = {key: runner.submit(key, ids, 6) for key, ids in prompts.items()}
            while runner.has_work:
                runner.tick()
                for key, request in requests.items():
                    slot = request.slot
                    if slot is not None and request.status == "running":
                        self.assertEqual(runner.host_seqlens[slot], request.prefilled + len(request.tokens) - 1, key)
            for key, request in requests.items():
                self.assertEqual(request.tokens, expected[key], key)
        finally:
            runner.close()


if __name__ == "__main__":
    unittest.main()
