"""Rank agreement before capture and replay also has CPU regression coverage."""

import contextlib
from datetime import timedelta
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from eole.decoders.transformer import TransformerDecoder
from eole.utils.decoder_graph import CaptureCoordinator, DecoderGraph


class TestCaptureCoordination(unittest.TestCase):
    def test_control_collective_uses_cpu_gloo_group(self):
        with patch("torch.distributed.new_group") as create:
            coordinator = CaptureCoordinator(timeout=17)
        create.assert_called_once_with(backend="gloo", timeout=timedelta(seconds=17))
        with patch("torch.distributed.all_reduce") as reduce:
            self.assertTrue(coordinator.all_success(True))
            self.assertFalse(coordinator.all_success(False))
        tensor = reduce.call_args.args[0]
        self.assertEqual(tensor.device.type, "cpu")
        self.assertIs(reduce.call_args.kwargs["group"], coordinator.group)

    def test_peer_failure_discards_capture_before_fallback(self):
        for stage in ("allocation", "capture"):
            with self.subTest(stage=stage):
                state = torch.zeros(())

                def forward(emb, *, tgt_pad_mask):
                    state.add_(1)
                    return emb + state

                coordinator = SimpleNamespace(
                    failed=False, all_success=Mock(side_effect=[False] if stage == "allocation" else [True, False])
                )
                runner = DecoderGraph(forward, forward, coordinator=coordinator)
                emb = torch.ones(1, 1, 4)
                mask = torch.zeros(1, 1, 1, dtype=torch.bool)
                for _ in range(2):
                    runner(emb, tgt_pad_mask=mask)
                # Simulated recording produces outputs but executes no GPU state.
                runner.function = Mock(return_value=emb)
                graph = Mock()
                stream = SimpleNamespace(device=emb.device)
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch("torch.cuda.device", return_value=contextlib.nullcontext()))
                    stack.enter_context(patch("torch.cuda.Stream", return_value=stream))
                    stack.enter_context(patch("torch.cuda.current_stream", return_value=stream))
                    stack.enter_context(patch("torch.cuda.stream", return_value=contextlib.nullcontext()))
                    stack.enter_context(patch("torch.cuda.CUDAGraph", return_value=graph))
                    capture = stack.enter_context(patch("torch.cuda.graph", return_value=contextlib.nullcontext()))
                    stack.enter_context(patch("eole.utils.decoder_graph.logger.warning"))
                    if stage == "capture":
                        with self.assertRaisesRegex(RuntimeError, "collective ordering"):
                            runner(emb, tgt_pad_mask=mask)
                    else:
                        result = runner(emb, tgt_pad_mask=mask)
                        torch.testing.assert_close(result, emb + 3)
                self.assertEqual(state.item(), 2 if stage == "capture" else 3)
                self.assertTrue(runner.failed)
                graph.reset.assert_called_once()
                graph.replay.assert_not_called()
                self.assertEqual(capture.call_count, int(stage == "capture"))
                calls = coordinator.all_success.call_count
                runner.reset()
                if stage == "capture":
                    with self.assertRaisesRegex(RuntimeError, "reinitialize"):
                        runner(emb, tgt_pad_mask=mask)
                else:
                    runner(emb, tgt_pad_mask=mask)
                self.assertEqual(coordinator.all_success.call_count, calls)
                self.assertEqual(state.item(), 2 if stage == "capture" else 4)

    def test_decoder_and_verifier_share_one_control_group(self):
        decoder = SimpleNamespace(_graph_parallel_gpu=2, _graph_capture_timeout=17, _graph_capture_coordinator=None)
        function = Mock()
        with patch("eole.utils.decoder_graph.CaptureCoordinator") as create:
            decode = TransformerDecoder._make_decoder_graph(decoder, function, function)
            verify = TransformerDecoder._make_decoder_graph(decoder, function, function)
        create.assert_called_once_with(timeout=17)
        self.assertIs(decode.coordinator, verify.coordinator)


class TestWorkerCaptureCleanup(unittest.TestCase):
    def test_failed_worker_does_not_add_a_cleanup_collective(self):
        from eole.utils.distributed_workers import spawned_infer

        config = SimpleNamespace(gpu_ranks=[0])
        error_queue = Mock()
        with patch("eole.utils.distributed_workers.multi_init", return_value=0), patch(
            "eole.utils.distributed_workers.InferenceWorker"
        ) as worker, patch("eole.utils.distributed_workers.cleanup_distributed") as cleanup:
            worker.return_value.initialize.side_effect = RuntimeError("injected initialization failure")
            spawned_infer(config, 0, error_queue, Mock(), Mock())
        cleanup.assert_called_once_with(barrier=False)
        self.assertIn("injected initialization failure", error_queue.put.call_args.args[0][1])

    def test_normal_stop_keeps_the_cleanup_barrier(self):
        from eole.constants import InferenceConstants
        from eole.utils.distributed_workers import spawned_infer

        instructions = Mock()
        instructions.get.return_value = InferenceConstants.STOP
        with patch("eole.utils.distributed_workers.multi_init", return_value=0), patch(
            "eole.utils.distributed_workers.InferenceWorker"
        ), patch("eole.utils.distributed_workers._drain_result_queue"), patch(
            "eole.utils.distributed_workers.cleanup_distributed"
        ) as cleanup:
            spawned_infer(SimpleNamespace(gpu_ranks=[0]), 0, Mock(), instructions, Mock())
        cleanup.assert_called_once_with(barrier=True)
