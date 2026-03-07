"""Unit tests for the GenerationStreamer class."""

import os
import importlib.util
import threading
from argparse import Namespace


def _import_streamer():
    """Import GenerationStreamer directly from its module file,
    bypassing eole.predict.__init__ which requires torch."""
    streamer_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "predict",
        "streamer.py",
    )
    spec = importlib.util.spec_from_file_location("eole.predict.streamer", streamer_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.GenerationStreamer


def _make_vocabs(token_map):
    """Build a minimal vocabs dict that looks like the real one.

    Args:
        token_map: dict mapping token_id (int) -> token_str.

    Returns:
        dict with a ``"tgt"`` key whose value has an ``ids_to_tokens``
        attribute (same interface as the real Vocab object).
    """
    return {"tgt": Namespace(ids_to_tokens=token_map)}


# ---------------------------------------------------------------------------
# Minimal mock objects that let us test GenerationStreamer without needing
# a real model or PyTorch.
# ---------------------------------------------------------------------------


class _FakeTransformPipe:
    """Minimal stand-in for TransformPipe for testing incremental decoding."""

    def __init__(self, tokens):
        """``tokens`` maps token_id -> str piece."""
        self._tokens = tokens

    def apply_reverse(self, token_ids):
        """Concatenate token pieces — simulates HF tokenizer decode."""
        return "".join(self._tokens.get(tid, "") for tid in token_ids)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_streamer_basic_put_end_iter():
    """Tokens put via put()/end() are yielded by the iterator."""
    GenerationStreamer = _import_streamer()

    vocabs = _make_vocabs({0: "Hello", 1: " world", 2: "!"})
    streamer = GenerationStreamer(vocabs=vocabs, transform_pipe=None)

    # Simulate the production thread
    def _producer():
        for tid in [0, 1, 2]:
            streamer.put([tid])
        streamer.end()

    t = threading.Thread(target=_producer)
    t.start()

    collected = list(streamer)
    t.join()

    assert collected == ["Hello", " world", "!"], collected


def test_streamer_with_transform_pipe():
    """With a transform pipe, incremental decoding is used."""
    GenerationStreamer = _import_streamer()

    tokens = {0: "H", 1: "e", 2: "llo"}
    pipe = _FakeTransformPipe(tokens)
    vocabs = _make_vocabs(tokens)

    streamer = GenerationStreamer(vocabs=vocabs, transform_pipe=pipe)

    def _producer():
        for tid in [0, 1, 2]:
            streamer.put([tid])
        streamer.end()

    t = threading.Thread(target=_producer)
    t.start()

    collected = list(streamer)
    t.join()

    # Incremental: each step yields only the new suffix
    assert collected == ["H", "e", "llo"], collected


def test_streamer_put_accepts_scalar():
    """put() handles single-element tensor-like objects."""
    GenerationStreamer = _import_streamer()

    class _FakeTensor:
        def __init__(self, val):
            self._val = val

        def __getitem__(self, idx):
            return _ScalarTensor(self._val)

    class _ScalarTensor:
        def __init__(self, val):
            self._val = val

        def item(self):
            return self._val

    vocabs = _make_vocabs({42: "word"})
    streamer = GenerationStreamer(vocabs=vocabs, transform_pipe=None)

    def _producer():
        streamer.put(_FakeTensor(42))
        streamer.end()

    t = threading.Thread(target=_producer)
    t.start()

    collected = list(streamer)
    t.join()

    assert collected == ["word"], collected


def test_streamer_unknown_token_skipped():
    """Unknown token IDs are silently skipped (no KeyError)."""
    GenerationStreamer = _import_streamer()

    vocabs = _make_vocabs({0: "hi"})
    streamer = GenerationStreamer(vocabs=vocabs, transform_pipe=None)

    def _producer():
        streamer.put([0])
        streamer.put([999])  # unknown token
        streamer.end()

    t = threading.Thread(target=_producer)
    t.start()

    collected = list(streamer)
    t.join()

    assert collected == ["hi"], collected


def test_streamer_empty_generation():
    """Streamer with no tokens yields nothing."""
    GenerationStreamer = _import_streamer()

    vocabs = _make_vocabs({})
    streamer = GenerationStreamer(vocabs=vocabs, transform_pipe=None)
    streamer.end()

    assert list(streamer) == []
