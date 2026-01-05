"""Pytorch Distributed utils
"""

import os
import signal
import math
import pickle
import torch.distributed
from typing import List, Any


# Constants
DEFAULT_BUFFER_SIZE = 10485760  # 10MB in bytes
DEFAULT_MAX_GATHER_SIZE = 4096  # Max size for all_gather_list
ENCODING_BASE = 255  # Base for size encoding (supports up to 65k)
ENCODING_MODULO = 256


def all_reduce_and_rescale_tensors(tensors: List, rescale_denom: float, buffer_size: int = DEFAULT_BUFFER_SIZE):
    """All-reduce and rescale tensors in chunks of the specified size.

    Args:
        tensors: list of Tensors to all-reduce
        rescale_denom: denominator for rescaling summed Tensors
        buffer_size: all-reduce chunk size in bytes (default: 10MB)
    """
    # buffer size in bytes, determine equiv. # of elements based on data type
    buffer_t = tensors[0].new(math.ceil(buffer_size / tensors[0].element_size())).zero_()
    buffer = []

    def all_reduce_buffer():
        """All-reduce accumulated buffer and copy back to tensors."""
        # copy tensors into buffer_t
        offset = 0
        for t in buffer:
            numel = t.numel()
            buffer_t[offset : offset + numel].copy_(t.view(-1))
            offset += numel

        # all-reduce and rescale
        torch.distributed.all_reduce(buffer_t[:offset], async_op=False)
        buffer_t.div_(rescale_denom)

        # copy all-reduced buffer back into tensors
        offset = 0
        for t in buffer:
            numel = t.numel()
            t.view(-1).copy_(buffer_t[offset : offset + numel])
            offset += numel

    filled = 0
    for t in tensors:
        sz = t.numel() * t.element_size()

        if sz > buffer_size:
            # tensor is bigger than buffer, all-reduce and rescale directly
            torch.distributed.all_reduce(t, async_op=False)
            t.div_(rescale_denom)
        elif filled + sz > buffer_size:
            # buffer is full, all-reduce and replace buffer with grad
            all_reduce_buffer()
            buffer = [t]
            filled = sz
        else:
            # add tensor to buffer
            buffer.append(t)
            filled += sz

    if len(buffer) > 0:
        all_reduce_buffer()


def all_gather_list(data: Any, max_size: int = DEFAULT_MAX_GATHER_SIZE) -> List[Any]:
    """Gathers arbitrary data from all nodes into a list.

    Args:
        data: Data to gather from this node
        max_size: Maximum size in bytes for encoded data (default: 4096)

    Returns:
        List of data from all nodes in the world

    Raises:
        ValueError: If encoded data exceeds max_size
    """
    world_size = torch.distributed.get_world_size()

    # Initialize or reuse buffers
    if not hasattr(all_gather_list, "_in_buffer") or max_size != all_gather_list._in_buffer.size(0):
        all_gather_list._in_buffer = torch.empty(max_size, dtype=torch.uint8, device="cuda")
        all_gather_list._out_buffers = [
            torch.empty(max_size, dtype=torch.uint8, device="cuda") for _ in range(world_size)
        ]

    in_buffer = all_gather_list._in_buffer
    out_buffers = all_gather_list._out_buffers

    # Encode data
    enc = pickle.dumps(data)
    enc_size = len(enc)

    if enc_size + 2 > max_size:
        raise ValueError(f"encoded data exceeds max_size: {enc_size + 2}")

    assert max_size < ENCODING_BASE * ENCODING_MODULO, "max_size must be < ENCODING_BASE * ENCODING_MODULO"

    # Encode size using 2 bytes (supports up to ~65k)
    in_buffer[0] = enc_size // ENCODING_BASE
    in_buffer[1] = enc_size % ENCODING_BASE
    in_buffer[2 : enc_size + 2].copy_(torch.frombuffer(bytearray(enc), dtype=torch.uint8))

    # All-gather buffers
    torch.distributed.all_gather(out_buffers, in_buffer)

    # Decode results from all nodes
    results = []
    for i in range(world_size):
        out_buffer = out_buffers[i]
        size = (ENCODING_BASE * out_buffer[0].item()) + out_buffer[1].item()

        bytes_list = bytes(out_buffer[2 : size + 2].tolist())
        result = pickle.loads(bytes_list)
        results.append(result)

    return results


class ErrorHandler:
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process.
    """

    def __init__(self, error_queue):
        """Initialize error handler with error queue.

        Args:
            error_queue: Queue for receiving errors from child processes
        """
        import threading

        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid: int):
        """Add a child process PID to monitor.

        Args:
            pid: Process ID of child to monitor
        """
        self.children_pids.append(pid)

    def error_listener(self):
        """Listen for errors from child processes and trigger signal handler."""
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum: int, stackframe):
        """Handle SIGUSR1 signal by killing children and raising exception.

        Args:
            signalnum: Signal number
            stackframe: Current stack frame

        Raises:
            Exception: With original traceback from child process
        """
        # Kill all child processes
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)

        # Get and display original error
        (rank, original_trace) = self.error_queue.get()
        msg = "\n\n-- Tracebacks above this line can probably be ignored --\n\n"
        msg += original_trace
        raise Exception(msg)
