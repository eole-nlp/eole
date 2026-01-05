"""Pytorch Distributed utils
"""

import torch
import torch.distributed
import logging
from typing import Optional, Tuple, Any
from datetime import timedelta
from eole.predict import build_predictor
from eole.transforms import get_transforms_cls, make_transforms
from eole.constants import CorpusTask, InferenceConstants
from eole.utils.logging import init_logger, logger
from eole.inputters.dynamic_iterator import build_dynamic_dataset_iter


def is_master(config, device_id: int) -> bool:
    """Check if the current process is the master process."""
    return config.gpu_ranks[device_id] == 0


def multi_init(config, device_id: int) -> int:
    """Initialize distributed process group.

    Args:
        config: Running configuration
        device_id: Device ID for this process

    Returns:
        GPU rank of the initialized process
    """
    dist_init_method = "tcp://{master_ip}:{master_port}".format(
        master_ip=config.master_ip, master_port=config.master_port
    )
    dist_world_size = config.world_size

    torch.distributed.init_process_group(
        backend=config.gpu_backend,
        init_method=dist_init_method,
        world_size=dist_world_size,
        rank=config.gpu_ranks[device_id],
        timeout=timedelta(seconds=config.timeout),
        device_id=device_id,
    )

    gpu_rank = torch.distributed.get_rank()

    if not is_master(config, device_id):
        logger.setLevel(logging.WARNING)
        # logger.disabled = True

    return gpu_rank


def cleanup_distributed():
    """Cleanup distributed process group."""
    import torch.distributed as dist

    if dist.is_initialized():
        dist.barrier()  # optional but recommended
        dist.destroy_process_group()


def spawned_train(process_fn, config, device_id: int, error_queue):
    """Run `process_fn` on `device_id` with distributed training setup.

    Args:
        process_fn: Training function to execute
        config: Full configuration object
        device_id: Device ID for this process
        error_queue: Queue for error reporting
    """
    try:
        gpu_rank = multi_init(config.training, device_id)
        if gpu_rank != config.training.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in Distributed initialization")

        process_fn(config, device_id=device_id)

    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback

        error_queue.put((config.training.gpu_ranks[device_id], traceback.format_exc()))
    finally:
        cleanup_distributed()


class InferenceWorker:
    """Worker class for distributed inference operations."""

    def __init__(self, config, device_id: int):
        """Initialize inference worker.

        Args:
            config: Configuration object
            device_id: Device ID for this worker
        """
        self.config = config
        self.device_id = device_id
        self.predictor = None
        self.transforms = None
        self.offset = 0
        self.stride = 1

    def initialize(self):
        """Initialize predictor and transforms."""
        torch.cuda.set_device(self.device_id)
        init_logger(self.config.log_file)

        self.predictor = build_predictor(self.config, self.device_id, logger=logger, report_score=True)

        transforms_cls = get_transforms_cls(self.config._all_transform)
        self.transforms = make_transforms(self.config, transforms_cls, self.predictor.vocabs)

        # Set offset and stride based on parallel mode
        if self.config.parallel_mode == "data_parallel":
            # same model on all GPU - stride = nb of gpu
            self.offset = max(0, self.device_id)
            self.stride = max(1, len(self.config.gpu_ranks))
        else:
            self.offset = 0
            self.stride = 1

    def _build_iterator(self, src: Optional[Any] = None, tgt: Optional[Any] = None, use_offset_stride: bool = False):
        """Build dynamic dataset iterator.

        Args:
            src: Source data
            tgt: Target data
            use_offset_stride: Whether to use offset and stride for data parallelism

        Returns:
            Dataset iterator
        """
        kwargs = {
            "task": CorpusTask.INFER,
            "device_id": self.device_id,
        }

        if src is not None:
            kwargs["src"] = src
        if tgt is not None:
            kwargs["tgt"] = tgt

        if use_offset_stride:
            kwargs["offset"] = self.offset
            kwargs["stride"] = self.stride

        return build_dynamic_dataset_iter(self.config, self.transforms, self.predictor.vocabs, **kwargs)

    def _predict(self, infer_iter) -> Tuple:
        """Run prediction on iterator.

        Args:
            infer_iter: Iterator to predict on

        Returns:
            Tuple of (scores, estims, preds)
        """
        return self.predictor._predict(
            infer_iter,
            infer_iter.transforms,
            self.config.attn_debug,
            self.config.align_debug,
        )

    def _score(self, infer_iter):
        """Run scoring on iterator.

        Args:
            infer_iter: Iterator to score on

        Returns:
            Score results
        """
        return self.predictor._score(infer_iter)

    def handle_infer_list(self, instruction, queue_result):
        """Handle infer_list command.

        Args:
            instruction: Command instruction tuple
            queue_result: Queue to put results
        """
        src = instruction[1]
        infer_iter = self._build_iterator(src=src, use_offset_stride=True)
        scores, estims, preds = self._predict(infer_iter)

        queue_result.put(scores)
        queue_result.put(estims)
        queue_result.put(preds)

    def handle_infer_file(self, instruction, queue_result):
        """Handle infer_file command.

        Args:
            instruction: Command instruction tuple
            queue_result: Queue to put results
        """
        self.config.src = instruction[1].src
        infer_iter = self._build_iterator()
        scores, estims, preds = self._predict(infer_iter)

        queue_result.put(scores)
        queue_result.put(estims)
        queue_result.put(preds)

    def handle_score_list(self, instruction, queue_result):
        """Handle score_list command.

        Args:
            instruction: Command instruction tuple
            queue_result: Queue to put results
        """
        tgt = instruction[1]
        infer_iter = self._build_iterator(src=tgt, tgt=tgt)
        score_results = self._score(infer_iter)

        queue_result.put(score_results)

    def handle_score_file(self, instruction, queue_result):
        """Handle score_file command.

        Args:
            instruction: Command instruction tuple
            queue_result: Queue to put results
        """
        self.config.src = instruction[1].src
        self.config.tgt = instruction[1].src
        infer_iter = self._build_iterator()
        score_results = self._score(infer_iter)

        queue_result.put(score_results)

    def update_settings(self, queue_settings):
        """Update predictor settings from queue if available.

        Args:
            queue_settings: Queue containing settings dict
        """
        if queue_settings is not None and not queue_settings.empty():
            settings = queue_settings.get()
            if settings:
                self.predictor.update_settings(**settings)


def spawned_infer(
    config, device_id: int, error_queue, queue_instruct, queue_result, queue_settings: Optional[Any] = None
):
    """Run various functions for prediction in spawned process on `device_id`.

    Args:
        config: Configuration object
        device_id: Device ID for this worker
        error_queue: Queue for error reporting
        queue_instruct: Queue for receiving instructions
        queue_result: Queue for sending results
        queue_settings: Optional queue for settings updates
    """
    try:
        gpu_rank = multi_init(config, device_id)
        if gpu_rank != config.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in Distributed initialization")

        # Initialize worker
        worker = InferenceWorker(config, device_id)
        worker.initialize()

        # Command handler mapping
        command_handlers = {
            InferenceConstants.INFER_LIST: worker.handle_infer_list,
            InferenceConstants.INFER_FILE: worker.handle_infer_file,
            InferenceConstants.SCORE_LIST: worker.handle_score_list,
            InferenceConstants.SCORE_FILE: worker.handle_score_file,
        }

        # Main worker loop
        while True:
            instruction = queue_instruct.get()

            # Extract command from instruction
            cmd = instruction[0] if isinstance(instruction, (list, tuple)) else instruction

            # Handle stop command
            if cmd == InferenceConstants.STOP:
                _drain_result_queue(queue_result)
                break

            # Update settings if available
            worker.update_settings(queue_settings)

            # Execute command
            handler = command_handlers.get(cmd)
            if handler:
                handler(instruction, queue_result)
            else:
                logger.warning(f"Unknown command: {cmd}")
                # Raise an exception so it is reported via error_queue and
                # callers do not wait indefinitely for a missing result.
                raise RuntimeError(f"Unknown inference command received by worker: {cmd}")

    except KeyboardInterrupt:
        pass  # killed by parent
    except Exception:
        import traceback

        error_queue.put((config.gpu_ranks[device_id], traceback.format_exc()))
    finally:
        cleanup_distributed()


def _drain_result_queue(queue_result):
    """Drain result queue to release any remaining CUDA tensors.

    Args:
        queue_result: Queue to drain
    """
    if not queue_result.empty():
        try:
            while True:
                _ = queue_result.get_nowait()
        except Exception:
            pass
