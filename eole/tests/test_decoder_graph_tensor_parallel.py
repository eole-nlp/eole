"""Real NCCL capture/replay and rank-local failure handling on two GPUs."""

from datetime import timedelta
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from eole.utils.decoder_graph import CaptureCoordinator, DecoderGraph
from eole.utils.tensor_parallel import broadcast_from_rank_zero


def _check_tensor_parallel_graph(rank, port, scenario, directory):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        device_id=rank,
        timeout=timedelta(seconds=30),
    )
    coordinator = CaptureCoordinator(timeout=30)
    state = torch.zeros((), device=rank)
    emb = torch.ones(1, 1, 4, device=rank)
    mask = torch.zeros(1, 1, 1, device=rank, dtype=torch.bool)

    def forward(emb, *, tgt_pad_mask):
        state.add_(1)
        result = emb.masked_fill(tgt_pad_mask, 0) + state
        if scenario == "capture" and rank == 1 and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("injected failure before this rank records its NCCL operation")
        dist.all_reduce(result)
        return result

    runner = DecoderGraph(forward, forward, coordinator=coordinator)
    fatal = False
    try:
        with torch.no_grad():
            for request in range(2 if scenario == "healthy" else 1):
                state.zero_()
                runner.reset()
                for step in range(5):
                    if scenario == "allocation" and rank == 1 and step == 2:
                        with patch(
                            "torch.cuda.Stream", side_effect=torch.cuda.OutOfMemoryError("injected allocation OOM")
                        ):
                            result = runner(emb, tgt_pad_mask=mask)
                    elif scenario == "capture" and step == 2:
                        try:
                            runner(emb, tgt_pad_mask=mask)
                        except RuntimeError as error:
                            assert "collective ordering" in str(error), str(error)
                            fatal = True
                            assert state.item() == 2
                            assert coordinator.failed
                            runner.reset()
                            try:
                                runner(emb, tgt_pad_mask=mask)
                            except RuntimeError as retry:
                                assert "reinitialize" in str(retry)
                            else:
                                raise AssertionError("retry must reject the failed communicator")
                            break
                        else:
                            raise AssertionError("both ranks must abort capture")
                    else:
                        result = runner(emb, tgt_pad_mask=mask)
                    torch.testing.assert_close(result, torch.full_like(result, 2 * (step + 2)))
                    assert state.item() == step + 1
                if scenario == "healthy":
                    assert runner.graph is not None
                if scenario == "allocation":
                    assert runner.failed and runner.graph is None
        if not fatal:
            tokens = torch.tensor([rank + 1], device=rank)
            scores = torch.tensor([rank + 0.5], device=rank)
            broadcast_from_rank_zero(tokens, 2)
            broadcast_from_rank_zero(scores, 2)
            assert tokens.item() == 1 and scores.item() == 0.5
        Path(directory, f"rank{rank}.json").write_text(json.dumps({"scenario": scenario, "fatal": fatal}))
    finally:
        runner.reset()
        # A partially recorded collective may have changed host-side NCCL
        # ordering. No further NCCL operations, including barriers, are safe.
        if not fatal:
            dist.barrier()
        dist.destroy_process_group()


@unittest.skipUnless(
    torch.cuda.device_count() >= 2 and dist.is_available() and dist.is_nccl_available() and dist.is_gloo_available(),
    "requires two GPUs, NCCL and Gloo",
)
class TestTensorParallelDecoderGraph(unittest.TestCase):
    def run_case(self, scenario):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_check_tensor_parallel_graph, args=(port, scenario, directory), nprocs=2, join=True)
            rows = [json.loads(Path(directory, f"rank{rank}.json").read_text()) for rank in (0, 1)]
            self.assertEqual([row["fatal"] for row in rows], [scenario == "capture"] * 2)

    def test_replay_and_reset_preserve_collective_state(self):
        self.run_case("healthy")

    def test_rank_local_allocation_oom_falls_back_on_both_ranks(self):
        self.run_case("allocation")

    def test_rank_local_capture_error_aborts_both_ranks(self):
        self.run_case("capture")
