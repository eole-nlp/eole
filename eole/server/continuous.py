"""Opt-in greedy streaming service on one persistent inference worker."""

from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from threading import Event, Lock, BoundedSemaphore
from time import perf_counter
from uuid import uuid4

import torch

from eole.predict.continuous import ContinuousGreedy
from eole.predict.streamer import GenerationStreamer
from eole.utils.logging import logger


@dataclass
class StreamRequest:
    prompt: str
    settings: dict
    streamer: object
    stats: dict | None
    request_id: str = field(default_factory=lambda: uuid4().hex)
    submitted: float = field(default_factory=perf_counter)
    done: Event = field(default_factory=Event)
    error: Exception | None = None
    admitted: bool = False


class ContinuousService:
    """Own the engine worker until close; ordinary engine entry points must not run."""

    def __init__(
        self,
        engine,
        slots=4,
        max_pending=32,
        prefill_chunk_size=128,
        compiled=True,
        runner_factory=None,
        prefill_idle_chunk_size=None,
    ):
        if any(not isinstance(value, int) or value < 1 for value in (slots, max_pending, prefill_chunk_size)):
            raise ValueError("Continuous serving limits must be positive integers")
        if prefill_idle_chunk_size is not None and (
            not isinstance(prefill_idle_chunk_size, int) or prefill_idle_chunk_size < prefill_chunk_size
        ):
            raise ValueError("Idle prefill chunk must be at least as large as the active chunk")
        if not isinstance(compiled, bool):
            raise ValueError("compiled must be a boolean")
        self.engine = engine
        self.options = dict(
            slots=slots,
            max_pending=max_pending,
            prefill_chunk_size=prefill_chunk_size,
            compiled=compiled,
            prefill_idle_chunk_size=prefill_idle_chunk_size,
        )
        if runner_factory is None and getattr(engine.predictor, "dflash2", None) is not None:
            from eole.predict.continuous_dflash2 import ContinuousDFlash2

            runner_factory = ContinuousDFlash2
        if runner_factory is None and getattr(engine.predictor, "self_speculative_decoding", False):
            from eole.predict.continuous_mtp import ContinuousMTP

            runner_factory = ContinuousMTP
        self.runner_factory = runner_factory or ContinuousGreedy
        self.queue = Queue(maxsize=max_pending)
        self.admission = BoundedSemaphore(slots + max_pending)
        self.lock = Lock()
        self.stopping = Event()
        self.ready = Event()
        self.error = None
        self.memory = {}
        self.future = None

    @staticmethod
    def validate(settings):
        allowed = {"max_length", "temperature", "top_k", "top_p"}
        if set(settings) - allowed:
            raise ValueError("Continuous serving supports ordinary greedy text generation only")
        if settings.get("top_k", 1) != 1 and settings.get("temperature", 1) != 0:
            raise ValueError("Continuous serving requires greedy selection (top_k=1 or temperature=0)")
        if settings.get("top_p", 0) not in (0, 1):
            raise ValueError("Continuous serving does not support nucleus sampling; use top_p=0 or 1")
        # Speculation is selected once at startup, never through mutable request settings.
        if settings.get("temperature", 1) != 0 and settings.get("temperature", 1) != 1:
            raise ValueError("Continuous serving supports temperature=0 or the greedy default only")
        if not isinstance(settings.get("max_length", 1), int) or settings.get("max_length", 1) < 1:
            raise ValueError("max_length must be a positive integer")

    @staticmethod
    def _prompt_ids(batch):
        source, lengths = batch.get("src"), batch.get("srclen")
        if not isinstance(source, torch.Tensor) or not isinstance(lengths, torch.Tensor):
            raise ValueError("Continuous admission requires source and srclen tensors")
        if source.ndim != 2 or source.size(0) != 1 or lengths.shape != (1,):
            raise ValueError("Continuous admission requires exactly one source sequence")
        if lengths.item() != source.size(1) or source.size(1) == 0:
            raise ValueError("Continuous admission requires a nonempty unpadded source sequence")
        # EOS/turn-boundary IDs can legitimately occur inside a formatted prompt.
        return source[0].tolist()

    def start(self):
        self.future = self.engine._thread_pool.submit(self.run)
        self.ready.wait()
        if self.error is not None:
            raise RuntimeError("Continuous serving preparation failed") from self.error

    def stream(self, prompt, settings, generation_stats=None, cancel_event=None, preserve_special_tokens=False):
        settings = dict(settings or {})
        self.validate(settings)
        streamer = GenerationStreamer(
            self.engine.vocabs,
            self.engine.transform_pipe,
            eos_token_ids=self.engine.predictor._tgt_eos_idx,
            cancel_event=cancel_event,
            preserve_special_tokens=preserve_special_tokens,
            timeout=600,
        )
        request = StreamRequest(prompt, settings, streamer, generation_stats)
        with self.lock:
            if self.stopping.is_set() or self.error is not None:
                raise RuntimeError("Continuous serving is unavailable; reload the model") from self.error
            if not self.admission.acquire(blocking=False):
                raise RuntimeError("Continuous serving request queue is full")
            request.admitted = True
            try:
                self.queue.put_nowait(request)
            except Full:
                request.admitted = False
                self.admission.release()
                raise RuntimeError("Continuous serving request queue is full") from None
        try:
            yield from streamer
        finally:
            if not request.done.is_set():
                streamer.cancel_event.set()
            request.done.wait()  # do not release a request while its worker owns it
        if request.error is not None:
            raise request.error

    def _finish(self, request, status, error=None):
        if request.done.is_set():
            return
        now = perf_counter()
        if request.stats is not None:
            first = request.streamer.first_token_time
            last = request.streamer.last_token_time
            count = request.streamer.token_count
            seconds = last - first if first is not None and last is not None else 0
            request.stats.update(
                output_tokens=count,
                stream_first_token_time=first,
                stream_last_token_time=last,
                stopped_on_eos=status == "stop",
                cancelled=status == "cancelled",
                first_token_seconds=first - request.submitted if first is not None else None,
                decode_seconds=seconds,
                decode_tokens=max(0, count - 1),
                decode_tokens_per_second=(count - 1) / seconds if seconds > 0 else None,
                total_seconds=now - request.submitted,
            )
            request.stats.setdefault("speculation_mode", "off")
            request.stats.setdefault("mtp_active", False)
        request.error = error
        if request.admitted:
            request.admitted = False
            self.admission.release()
        request.streamer.end()
        request.done.set()

    @staticmethod
    def _collect_statistics(runner, request):
        if request.stats is not None and hasattr(runner, "statistics"):
            request.stats.update(runner.statistics(request.request_id))

    @torch.inference_mode()
    def run(self):
        runner = None
        memory_device = None
        jobs = {}
        try:
            started = perf_counter()
            logger.info("Preparing continuous decode slots and graphs before readiness…")
            runner = self.runner_factory(self.engine.predictor, **self.options)
            logger.info("Continuous serving ready in %.2f s: %s", perf_counter() - started, self.options)
            memory_device = getattr(getattr(runner, "next_tokens", None), "device", None)
            if memory_device is not None and memory_device.type == "cuda":
                self.memory.update(
                    startup_allocated_gib=torch.cuda.memory_allocated(memory_device) / 2**30,
                    startup_reserved_gib=torch.cuda.memory_reserved(memory_device) / 2**30,
                    startup_peak_allocated_gib=torch.cuda.max_memory_allocated(memory_device) / 2**30,
                )
                torch.cuda.reset_peak_memory_stats(memory_device)
                logger.info("Continuous serving startup VRAM: %s", self.memory)
            self.ready.set()
            while not self.stopping.is_set():
                try:
                    request = self.queue.get(timeout=0 if runner.has_work else 0.02)
                except Empty:
                    request = None
                if request is not None:
                    if request.streamer.cancelled:
                        self._finish(request, "cancelled")
                    else:
                        try:
                            iterator = self.engine._build_inference_iterator(src=[request.prompt])
                            batch, _ = next(iter(iterator))
                            ids = self._prompt_ids(batch)
                            runner.submit(
                                request.request_id,
                                ids,
                                request.settings.get("max_length", self.engine.predictor.max_length),
                            )
                        except Exception as exc:
                            self._finish(request, "failed", exc.with_traceback(None))
                        else:
                            jobs[request.request_id] = request
                for identifier, request in list(jobs.items()):
                    if request.streamer.cancelled:
                        runner.cancel(identifier)
                        self._collect_statistics(runner, request)
                        runner.forget(identifier)
                        jobs.pop(identifier)
                        self._finish(request, "cancelled")
                if runner.has_work:
                    for identifier, token, status in runner.tick():
                        request = jobs[identifier]
                        request.streamer.put([token])
                        if status != "running":
                            jobs.pop(identifier)
                            self._collect_statistics(runner, request)
                            runner.forget(identifier)
                            self._finish(request, status)
        except Exception as exc:
            self.error = exc.with_traceback(None)
        finally:
            with self.lock:
                self.stopping.set()
            error = self.error or RuntimeError("Continuous serving stopped")
            for request in jobs.values():
                self._finish(request, "failed", error)
            while True:
                try:
                    self._finish(self.queue.get_nowait(), "failed", error)
                except Empty:
                    break
            if memory_device is not None and memory_device.type == "cuda":
                try:
                    self.memory.update(
                        serving_peak_allocated_gib=torch.cuda.max_memory_allocated(memory_device) / 2**30,
                        serving_peak_reserved_gib=torch.cuda.max_memory_reserved(memory_device) / 2**30,
                    )
                    logger.info("Continuous serving peak VRAM: %s", self.memory)
                except Exception as exc:
                    logger.warning("Could not read serving VRAM statistics: %s", exc)
            if runner is not None:
                try:
                    runner.close()
                except Exception:
                    pass  # a failed CUDA context cannot be recovered here
            self.ready.set()

    def close(self):
        with self.lock:
            self.stopping.set()
        if self.future is not None:
            self.future.result()
