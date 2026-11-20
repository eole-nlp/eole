"""Admission/cancellation controls need no GPU or checkpoint."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from threading import get_ident
import unittest

import torch

from eole.predict.continuous import ContinuousGreedy


class ContinuousAdmissionTests(unittest.TestCase):
    def runner(self):
        # Exercise host controls independently of device allocation/forwards.
        runner = object.__new__(ContinuousGreedy)
        runner.prefix_cache_entries = 0
        runner.worker = get_ident()
        runner.closed = False
        runner.vocab_size = 10
        runner.capacity = 16
        runner.prefilling = None
        runner.pending = deque()
        runner.active = {}
        runner.requests = {}
        runner.max_pending = 1
        runner.next_tokens = torch.zeros(2, 1, dtype=torch.long)
        return runner

    def test_cancel_reclaims_queue_admission_and_forget_releases_handle(self):
        runner = self.runner()
        first = runner.submit("first", [1, 2], 4)
        with self.assertRaisesRegex(ValueError, "full"):
            runner.submit("second", [1], 4)
        runner.cancel("first")
        self.assertEqual(first.status, "cancelled")
        second = runner.submit("second", [2, 3], 4)
        self.assertEqual(list(runner.pending), [second])
        runner.forget("first")
        self.assertNotIn("first", runner.requests)
        with self.assertRaisesRegex(ValueError, "active"):
            runner.forget("second")

    def test_cancelling_partial_prefill_releases_its_cache(self):
        runner = self.runner()
        request = runner.submit("partial", [1, 2], 4)
        runner.pending.popleft()
        request.status, request.slot, request.cache = "prefilling", 0, object()
        runner.prefilling = request
        runner.cancel("partial")
        self.assertEqual(request.status, "cancelled")
        self.assertIsNone(request.slot)
        self.assertIsNone(request.cache)
        self.assertIsNone(runner.prefilling)

    def test_invalid_ids_and_capacity_are_rejected_before_forward(self):
        for prompt, budget in (([10], 1), ([-1], 1), ([1.5], 1), ([1], 16), ([], 1)):
            runner = self.runner()
            with self.assertRaises(ValueError):
                runner.submit("invalid", prompt, budget)
            self.assertFalse(runner.pending)
            self.assertFalse(runner.requests)

    def test_foreign_worker_and_closed_runner_are_rejected(self):
        runner = self.runner()
        with ThreadPoolExecutor(1) as pool:
            with self.assertRaisesRegex(RuntimeError, "owning"):
                pool.submit(runner.submit, "wrong", [1], 1).result()
        runner.closed = True
        with self.assertRaisesRegex(RuntimeError, "closed"):
            runner.submit("closed", [1], 1)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            runner.tick()
