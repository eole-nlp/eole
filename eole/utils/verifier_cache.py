"""Locally generated, runtime-specific compiled MTP verifier programs.

These are trusted executable caches, never checkpoint files or downloaded assets.
Only construction/loading failures may fall back: retrying a failed execution
could advance KV/GDN state twice.
"""

from contextlib import contextmanager
from functools import lru_cache
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from time import perf_counter

import torch

from eole.utils.logging import logger


def verifier_forward(model, emb, *, tgt_pad_mask):
    # Use a module argument, not a closure over weights or request state.
    return model._forward_verify_graph(emb, tgt_pad_mask=tgt_pad_mask)


def compile_verifier(model):
    return torch.compile(
        model._forward_verify_graph,
        fullgraph=True,
        dynamic=False,
        options={
            "guard_filter_fn": lambda guards: [g.guard_type == "TENSOR_MATCH" for g in guards],
            "triton.cudagraphs": False,
        },
    )


@lru_cache(maxsize=1)
def source_fingerprint():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted([*root.rglob("*.py"), *root.glob("_ops*.so")]):
        if "tests" not in path.relative_to(root).parts:
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def tensor_layout(tensor):
    return [list(tensor.shape), list(tensor.stride()), str(tensor.dtype), str(tensor.device)]


def state_layout(model):
    """Include unregistered KV/GDN buffers without serializing their contents."""
    tensors = []

    def visit(name, value):
        if isinstance(value, torch.Tensor):
            tensors.append((name, tensor_layout(value)))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(f"{name}.{index}", item)
        elif isinstance(value, dict):
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
                visit(f"{name}.{key}", item)

    for name, module in model.named_modules():
        for key, value in vars(module).items():
            if key not in ("_modules", "_parameters", "_buffers"):
                visit(f"{name}.{key}", value)
    return tensors


def runtime_fingerprint(device):
    packages = {}
    for name in ("triton", "fla-core", "flash-attn", "auto-round", "causal-conv1d"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    props = torch.cuda.get_device_properties(device)
    driver = Path("/proc/driver/nvidia/version")
    return dict(
        torch=torch.__version__,
        torch_git=torch.version.git_version,
        cuda=torch.version.cuda,
        driver=driver.read_text() if driver.exists() else None,
        python=list(sys.version_info[:3]),
        machine=platform.machine(),
        gpu=[props.name, props.major, props.minor, props.multi_processor_count],
        packages=packages,
        matmul_precision=torch.get_float32_matmul_precision(),
        tf32=torch.backends.cuda.matmul.allow_tf32,
        deterministic=torch.are_deterministic_algorithms_enabled(),
    )


def cache_identity(model, emb, mask):
    return dict(
        format=1,
        source=source_fingerprint(),
        runtime=runtime_fingerprint(emb.device),
        configuration=model._verifier_cache_config,
        inputs=[tensor_layout(emb), tensor_layout(mask)],
        modules=[(name, type(module).__module__, type(module).__qualname__) for name, module in model.named_modules()],
        tensors=[(name, tensor_layout(tensor)) for name, tensor in [*model.named_parameters(), *model.named_buffers()]],
        state=state_layout(model),
    )


@contextmanager
def cache_lock(directory):
    import fcntl

    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (directory / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def save_atomic(compiled, path, references):
    # Torch's saver uses atomic replacement too; the separate temporary name
    # also leaves a previous valid entry untouched if serialization fails.
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".verifier-")
    os.close(fd)
    try:
        compiled.save_compiled_function(name, external_data=references)
        os.chmod(name, 0o600)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


class CachedVerifier:
    """Resolve once per model; bind new request buffers through the model argument."""

    def __init__(self, model, directory):
        self.model = model
        self.directory = Path(os.path.expandvars(directory)).expanduser()
        self.functions = {}

    def _bind(self, compiled):
        model = self.model

        def function(emb, *, tgt_pad_mask):
            return compiled(model, emb, tgt_pad_mask=tgt_pad_mask)

        function.guard_check = lambda emb, mask: compiled.guard_check(model, emb, tgt_pad_mask=mask)
        return function

    def _resolve(self, emb, mask):
        model = self.model
        fallback = compile_verifier(model)
        if not hasattr(fallback, "aot_compile") or not hasattr(torch.compiler, "load_compiled_function"):
            logger.warning("Persistent MTP verifier unavailable in this PyTorch; using ordinary compilation.")
            return fallback
        try:
            identity = cache_identity(model, emb, mask)
            encoded = json.dumps(identity, sort_keys=True, default=str).encode()
            key = hashlib.sha256(encoded).hexdigest()
            directory = self.directory / key
            references = {"decoder" + ("." + name if name else ""): module for name, module in model.named_modules()}
            with cache_lock(directory):
                path = directory / "verifier.aot"
                started = perf_counter()
                if path.exists():
                    try:
                        with path.open("rb") as file:
                            compiled = torch.compiler.load_compiled_function(file, external_data=references)
                        if not compiled.guard_check(model, emb, tgt_pad_mask=mask):
                            raise ValueError("saved verifier guards do not match this model/cache layout")
                        logger.info("Loaded local MTP verifier in %.2f s (%s).", perf_counter() - started, key[:12])
                        return self._bind(compiled)
                    except Exception as exc:
                        logger.warning("Cannot reuse local MTP verifier; recompiling: %s", exc)
                logger.info("Compiling full MTP verification chunk for this runtime (%s)…", key[:12])
                candidate = torch.compile(
                    verifier_forward,
                    fullgraph=True,
                    dynamic=False,
                    options={
                        "guard_filter_fn": lambda guards: [g.guard_type == "TENSOR_MATCH" for g in guards],
                        "triton.cudagraphs": False,
                    },
                )
                compiled = candidate.aot_compile(((model, emb), {"tgt_pad_mask": mask}))
                try:
                    save_atomic(compiled, path, references)
                    (directory / "identity.json").write_bytes(encoded)
                    logger.info("Saved local MTP verifier after %.2f s (%s).", perf_counter() - started, key[:12])
                except Exception as exc:
                    # Keep the just-compiled program even if the disk is full.
                    logger.warning("Could not save MTP verifier; using the in-memory program: %s", exc)
                return self._bind(compiled)
        except Exception as exc:
            logger.warning("Persistent MTP verifier preparation failed; using ordinary compilation: %s", exc)
            return fallback

    def __call__(self, emb, *, tgt_pad_mask):
        signature = json.dumps([tensor_layout(emb), tensor_layout(tgt_pad_mask)])
        function = self.functions.get(signature)
        check = getattr(function, "guard_check", None)
        if function is None or (check is not None and not check(emb, tgt_pad_mask)):
            function = self._resolve(emb, tgt_pad_mask)
            self.functions[signature] = function
        # Deliberately outside the fallback handler: execution can mutate state.
        return function(emb, tgt_pad_mask=tgt_pad_mask)
