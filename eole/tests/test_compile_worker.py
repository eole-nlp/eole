"""Inductor TLS survives requests and is initialized independently after model swaps."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from eole.inference_engine import InferenceEnginePY
from eole.utils.decoder_graph import initialize_cuda_graph_thread


class TestCompileWorker(unittest.TestCase):
    def test_new_workers_get_independent_state_and_preserve_it(self):
        # Import on the caller: ordinary Python worker threads do not inherit
        # the module's threading.local() state or the autograd TLS dictionary.
        from torch._inductor import cudagraph_trees as trees

        def inspect():
            state = trees.get_obj(trees.local, "tree_manager_containers")
            self.assertIs(torch._C._get_obj_in_tls("tree_manager_containers"), state)
            self.assertIs(torch._C._get_obj_in_tls("tree_manager_locks"), trees.local.tree_manager_locks)
            return state

        states = []
        for _ in range(2):
            with ThreadPoolExecutor(max_workers=1, initializer=initialize_cuda_graph_thread) as worker:
                state = worker.submit(inspect).result()
                state["sentinel"] = object()
                worker.submit(initialize_cuda_graph_thread).result()
                self.assertIs(worker.submit(inspect).result(), state)
                self.assertIn("sentinel", state)
                states.append(state)
        self.assertIsNot(states[0], states[1])
        self.assertIsNot(states[0], trees.get_obj(trees.local, "tree_manager_containers"))

    def test_inherited_autograd_state_is_not_replaced(self):
        from collections import defaultdict
        import threading
        from torch._inductor import cudagraph_trees as trees

        def inherit():
            containers, locks = {}, defaultdict(threading.Lock)
            torch._C._stash_obj_in_tls("tree_manager_containers", containers)
            torch._C._stash_obj_in_tls("tree_manager_locks", locks)
            initialize_cuda_graph_thread()
            self.assertIs(trees.get_obj(trees.local, "tree_manager_containers"), containers)
            self.assertIs(trees.get_obj(trees.local, "tree_manager_locks"), locks)

        with ThreadPoolExecutor(max_workers=1) as worker:
            worker.submit(inherit).result()

    def test_engine_pool_initializes_graph_modes_only(self):
        from torch._inductor import cudagraph_trees as trees

        for mode in ("0", "2"):
            with patch("eole.inference_engine.EOLE_TORCH_COMPILE", True), patch(
                "eole.inference_engine.EOLE_COMPILE_MODE", mode
            ), patch("torch.cuda.is_available", return_value=True):
                with InferenceEnginePY._make_thread_pool(SimpleNamespace()) as worker:
                    state = worker.submit(trees.get_obj, trees.local, "tree_manager_containers").result()
                    self.assertIsInstance(state, dict)
        for enabled, mode, cuda in ((False, "0", True), (True, "1", True), (True, "3", True), (True, "0", False)):
            with patch("eole.inference_engine.EOLE_TORCH_COMPILE", enabled), patch(
                "eole.inference_engine.EOLE_COMPILE_MODE", mode
            ), patch("torch.cuda.is_available", return_value=cuda), patch(
                "eole.utils.decoder_graph.initialize_cuda_graph_thread"
            ) as initialize:
                with InferenceEnginePY._make_thread_pool(SimpleNamespace()) as worker:
                    worker.submit(lambda: None).result()
                initialize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
