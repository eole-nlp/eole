"""Executable-cache binding, invalidation, and failure policy on CPU CI."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from eole.decoders.transformer import TransformerDecoder
from eole.predict.generator import GeneratorLM
from eole.utils.verifier_cache import CachedVerifier, cache_identity


class StatefulVerifier(torch.nn.Module):
    def __init__(self, weight=1, state=0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((4,), float(weight)), requires_grad=False)
        self.state = torch.tensor(float(state))  # GDN-like unregistered state
        self._verifier_cache_config = {"context": 128, "bits": 4}

    def _forward_verify_graph(self, emb, *, tgt_pad_mask):
        self.state.add_(1)
        return (emb + self.weight + self.state).masked_fill(tgt_pad_mask, 0)


class TestVerifierCache(unittest.TestCase):
    def setUp(self):
        self.temp = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(
            patch(
                "eole.utils.verifier_cache.runtime_fingerprint",
                return_value={"torch": "test"},
            )
        )
        self.enterContext(patch("eole.utils.verifier_cache.source_fingerprint", return_value="source-1"))
        self.emb = torch.ones(1, 4, 4)
        self.mask = torch.zeros(1, 4, 1, dtype=torch.bool)

    def test_key_changes_for_code_runtime_layout_context_and_verification_length(self):
        model = StatefulVerifier()
        original = cache_identity(model, self.emb, self.mask)
        for component in ("code", "runtime", "context", "state", "shape", "weights"):
            with self.subTest(component=component):
                other = StatefulVerifier()
                emb = self.emb
                mask = self.mask
                if component == "context":
                    other._verifier_cache_config["context"] = 256
                if component == "state":
                    other.state = torch.zeros(2)
                if component == "shape":
                    emb, mask = self.emb[:, :2], self.mask[:, :2]
                if component == "weights":
                    other.weight = torch.nn.Parameter(torch.zeros(8), requires_grad=False)
                with patch(
                    "eole.utils.verifier_cache.source_fingerprint",
                    return_value="source-2" if component == "code" else "source-1",
                ), patch(
                    "eole.utils.verifier_cache.runtime_fingerprint",
                    return_value={"torch": "other" if component == "runtime" else "test"},
                ):
                    self.assertNotEqual(original, cache_identity(other, emb, mask))

    @unittest.skipUnless(
        hasattr(torch.compiler, "load_compiled_function"),
        "PyTorch AOT save/load unavailable",
    )
    @torch.inference_mode()
    def test_reload_rebinds_weights_and_unregistered_state_without_recompiling(self):
        model = StatefulVerifier()
        runner = CachedVerifier(model, self.temp)
        torch.testing.assert_close(runner(self.emb, tgt_pad_mask=self.mask), torch.full_like(self.emb, 3))
        paths = list(Path(self.temp).glob("*/verifier.aot"))
        self.assertEqual(len(paths), 1)
        # A different model instance and values must reuse code, not old weights/state.
        other = StatefulVerifier(weight=7, state=10)
        loaded = CachedVerifier(other, self.temp)
        with patch("torch.compile", wraps=torch.compile) as compile_call:
            torch.testing.assert_close(loaded(self.emb, tgt_pad_mask=self.mask), torch.full_like(self.emb, 19))
        self.assertEqual(compile_call.call_count, 1)  # lazy ordinary fallback construction only
        torch.testing.assert_close(model.state, torch.tensor(1.0))
        torch.testing.assert_close(other.state, torch.tensor(11.0))
        torch.testing.assert_close(loaded(self.emb, tgt_pad_mask=self.mask), torch.full_like(self.emb, 20))
        # Changing the full chunk length within a process selects another program.
        torch.testing.assert_close(
            loaded(self.emb[:, :2].contiguous(), tgt_pad_mask=self.mask[:, :2].contiguous()),
            torch.full((1, 2, 4), 21.0),
        )
        self.assertEqual(len(list(Path(self.temp).glob("*/verifier.aot"))), 2)

    def test_corrupt_cache_is_rebuilt_and_failure_does_not_retry_execution(self):
        model = StatefulVerifier()
        loaded = Mock()
        loaded.guard_check.return_value = True
        loaded.save_compiled_function.side_effect = lambda name, **kw: Path(name).write_bytes(b"new")
        compiled = Mock()
        compiled.aot_compile.return_value = loaded
        with patch("torch.compile", return_value=compiled), patch(
            "torch.compiler.load_compiled_function",
            side_effect=ValueError("corrupt"),
            create=True,
        ) as loader:
            runner = CachedVerifier(model, self.temp)
            runner(self.emb, tgt_pad_mask=self.mask)
            # Resolve the same on-disk key in a new process/model instance.
            runner = CachedVerifier(model, self.temp)
            runner(self.emb, tgt_pad_mask=self.mask)
            loader.assert_called_once()
            self.assertEqual(compiled.aot_compile.call_count, 2)
            loaded.side_effect = RuntimeError("kernel failed after state mutation")
            with self.assertRaisesRegex(RuntimeError, "after state mutation"):
                runner(self.emb, tgt_pad_mask=self.mask)
            self.assertEqual(compiled.aot_compile.call_count, 2)
            compiled.assert_not_called()

    def test_unwritable_cache_falls_back_once_per_shape(self):
        fallback = Mock(return_value="ordinary")
        with patch("torch.compile", return_value=fallback), patch(
            "eole.utils.verifier_cache.cache_lock",
            side_effect=PermissionError("read-only"),
        ):
            runner = CachedVerifier(StatefulVerifier(), self.temp)
            for _ in range(2):
                self.assertEqual(runner(self.emb, tgt_pad_mask=self.mask), "ordinary")
            self.assertEqual(fallback.call_count, 2)
            fallback.aot_compile.assert_not_called()

    def test_unsupported_serialization_keeps_ordinary_compilation(self):
        def fallback(emb, *, tgt_pad_mask):
            return emb

        with patch("torch.compile", return_value=fallback):
            runner = CachedVerifier(StatefulVerifier(), self.temp)
            self.assertIs(runner(self.emb, tgt_pad_mask=self.mask), self.emb)
        self.assertFalse(list(Path(self.temp).glob("*/verifier.aot")))

    def test_failed_save_keeps_using_just_compiled_program(self):
        loaded = Mock(return_value="new program")
        loaded.guard_check.return_value = True
        loaded.save_compiled_function.side_effect = OSError("disk full")
        candidate = Mock()
        candidate.aot_compile.return_value = loaded
        with patch("torch.compile", return_value=candidate):
            runner = CachedVerifier(StatefulVerifier(), self.temp)
            for _ in range(2):
                self.assertEqual(runner(self.emb, tgt_pad_mask=self.mask), "new program")
        candidate.aot_compile.assert_called_once()
        candidate.assert_not_called()
        self.assertFalse(list(Path(self.temp).glob("*/verifier.aot")))


class TestFullChunkPolicy(unittest.TestCase):
    def test_full_chunk_warmup_skips_short_shapes_and_single_token_compile(self):
        for length in (1, 2, 4):
            with self.subTest(length=length):
                decoder = Mock(hidden_size=4, _full_chunk_verifier=True, _speculative_forward=False)
                decoder.modules.return_value = []
                predictor = SimpleNamespace(model=SimpleNamespace(decoder=decoder))
                GeneratorLM._warmup_speculative_decoder(predictor, torch.zeros(1, 8, 4), length)
                self.assertEqual(
                    [call.args[0].size(1) for call in decoder.call_args_list],
                    [length] if length > 1 else [],
                )
                decoder._replay_decoder.assert_not_called()
                self.assertFalse(decoder._speculative_forward)

    def test_mode0_short_tails_use_eager_without_legacy_graphs(self):
        for length in (1, 2, 3):
            with self.subTest(length=length):
                decoder = SimpleNamespace(
                    _full_chunk_verifier=True,
                    _verify_graph_request_allowed=True,
                    _speculative_forward=length > 1,
                    _can_replay_verifier=Mock(
                        side_effect=lambda emb, kw, **opts: opts.get("allow_tail", False) and length > 1
                    ),
                    _can_replay_decoder=Mock(side_effect=lambda emb, kw, **opts: opts.get("for_warmup", False)),
                    _forward_eager=Mock(return_value="eager"),
                )
                with patch("eole.decoders.transformer.EOLE_TORCH_COMPILE", True), patch(
                    "eole.decoders.transformer.EOLE_COMPILE_MODE", "0"
                ):
                    self.assertEqual(
                        TransformerDecoder.forward(decoder, torch.zeros(1, length, 4), tgt_pad_mask="mask"),
                        "eager",
                    )


if __name__ == "__main__":
    unittest.main()
