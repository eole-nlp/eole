#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for vocab file format validation and error messages."""

import os
import tempfile
import unittest

from eole.inputters.inputter import _read_vocab_file


class TestVocabFileValidation(unittest.TestCase):
    """Tests for vocab file format validation and error messages."""

    def wrong_vocab(self, test_case):
        """Test error handling for malformed vocab files."""
        content, expected_line, expected_msg = test_case

        with tempfile.NamedTemporaryFile(mode="w", suffix=".vocab", delete=False) as f:
            f.write(content)
            vocab_path = f.name

        try:
            with self.assertRaises(ValueError) as context:
                _read_vocab_file(vocab_path, min_count=1)

            error_msg = str(context.exception)
            self.assertIn(f"line {expected_line}", error_msg)
            self.assertIn(expected_msg, error_msg)
        finally:
            os.unlink(vocab_path)

    def correct_vocab(self, test_case):
        """Test correct vocab file reading with different formats."""
        content, min_count, expected_vocab = test_case

        with tempfile.NamedTemporaryFile(mode="w", suffix=".vocab", delete=False) as f:
            f.write(content)
            vocab_path = f.name

        try:
            vocab = _read_vocab_file(vocab_path, min_count=min_count)
            self.assertEqual(vocab, expected_vocab)
        finally:
            os.unlink(vocab_path)


def _add_test(param_setting, methodname, idx):
    """
    Adds a Test to TestVocabFileValidation according to settings.
    Args:
        param_setting: tuple of test parameters
        methodname: name of the method that gets called
        idx: index for test name
    """

    def test_method(self):
        getattr(self, methodname)(param_setting)

    name = f"test_{methodname}_{idx}"
    setattr(TestVocabFileValidation, name, test_method)
    test_method.__name__ = name


test_wrong_vocab = [
    # (content, expected_line, expected_error_msg)
    ("\t100\n", 1, "token is empty"),
    ("test\t\n", 1, "count is empty"),
]


test_correct_vocab = [
    # (content, min_count, expected_vocab)
    ("hello\t1\n", 1, ["hello"]),
    ("hello\t999999\n", 1, ["hello"]),
    ("hello\t0\n", 1, []),
    ("hello\t1\n", 2, []),
    ("特殊字符\t100\n", 1, ["特殊字符"]),
    ("emoji🎉\t100\n", 1, ["emoji🎉"]),
    ("\u000b\t100\n", 1, ["\u000b"]),
    ("\u000c\t100\n", 1, ["\u000c"]),
    ("\u000d\t100\n", 1, ["\u000d"]),
    ("\u001c\t100\n", 1, ["\u001c"]),
    ("\u001d\t100\n", 1, ["\u001d"]),
    ("\u001e\t100\n", 1, ["\u001e"]),
    ("\u001f\t100\n", 1, ["\u001f"]),
    ("\u0020\t100\n", 1, ["\u0020"]),
    ("\u0085\t100\n", 1, ["\u0085"]),
    ("\u00a0\t100\n", 1, ["\u00a0"]),
    ("\u1680\t100\n", 1, ["\u1680"]),
    ("\u2000\t100\n", 1, ["\u2000"]),
    ("\u2001\t100\n", 1, ["\u2001"]),
    ("\u2002\t100\n", 1, ["\u2002"]),
    ("\u2003\t100\n", 1, ["\u2003"]),
    ("\u2004\t100\n", 1, ["\u2004"]),
    ("\u2005\t100\n", 1, ["\u2005"]),
    ("\u2006\t100\n", 1, ["\u2006"]),
    ("\u2007\t100\n", 1, ["\u2007"]),
    ("\u2008\t100\n", 1, ["\u2008"]),
    ("\u2009\t100\n", 1, ["\u2009"]),
    ("\u200a\t100\n", 1, ["\u200a"]),
    ("\u2028\t100\n", 1, ["\u2028"]),
    ("\u2029\t100\n", 1, ["\u2029"]),
    ("\u202f\t100\n", 1, ["\u202f"]),
    ("\u205f\t100\n", 1, ["\u205f"]),
    ("\u3000\t100\n", 1, ["\u3000"]),
]

for idx, p in enumerate(test_wrong_vocab):
    _add_test(p, "wrong_vocab", idx)

for idx, p in enumerate(test_correct_vocab):
    _add_test(p, "correct_vocab", idx)
