"""Prepare one Qwen Chat Pod, then supervise a private Eole server."""

import asyncio
from contextlib import asynccontextmanager, suppress
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import signal
import sys

import httpx
import uvicorn
import yaml

try:
    from .gateway import create_gateway
except ImportError:
    from gateway import create_gateway

LOG = logging.getLogger("eole.runpod")
MODEL_ID = "RedHatAI/Qwen3.8-27B-INT4"
RECIPE = Path(__file__).resolve().parent


def valid_model(path):
    return (
        (path / "config.json").is_file()
        and (path / "tokenizer.json").is_file()
        and any(path.glob("model.*.safetensors"))
    )


def relocate(value, old, new):
    """Conversion writes tokenizer paths before its staging directory is renamed."""
    if isinstance(value, str) and value.startswith(str(old) + "/"):
        return str(new) + value[len(str(old)) :]
    if isinstance(value, dict):
        return {key: relocate(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [relocate(item, old, new) for item in value]
    return value


def kernel_fingerprint(source, runtime):
    digest = hashlib.sha256(json.dumps(runtime, sort_keys=True).encode())
    for path in [source / "setup.py", *sorted((source / "csrc").rglob("*"))]:
        if path.is_file():
            digest.update(str(path.relative_to(source)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:24]


async def stop_process(process):
    if process and process.returncode is None:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            await asyncio.wait_for(process.wait(), 15)
        except asyncio.TimeoutError:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()


class Deployment:
    def __init__(self, root, source, state):
        self.root, self.source, self.state = root, source, state
        self.process = None

    def stage(self, stage, message):
        self.state.update(stage=stage, message=message)
        LOG.info("%s", message)

    async def command(self, *args):
        self.process = await asyncio.create_subprocess_exec(*map(str, args), cwd=self.source, start_new_session=True)
        try:
            code = await self.process.wait()
            if code:
                raise RuntimeError("Preparation command failed; see the private Pod logs")
        finally:
            await stop_process(self.process)
            self.process = None

    async def kernels(self):
        self.stage("gpu", "Checking the GPU and CUDA runtime…")
        runtime_file = self.root / "runtime.json"
        await self.command(
            sys.executable,
            "-c",
            """
import json, platform, subprocess, sys, torch
assert torch.cuda.is_available(), 'The Pod needs a visible NVIDIA GPU'
assert torch.cuda.get_device_properties(0).total_memory >= 30 * 1024**3, 'Choose at least 32 GB advertised VRAM'
info = dict(torch=torch.__version__, python=sys.version, machine=platform.machine(),
            cuda=torch.version.cuda, nvcc=subprocess.check_output(['nvcc','--version'],text=True),
            architectures=sorted(set(torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count()))))
open(sys.argv[1],'w').write(json.dumps(info))
""",
            runtime_file,
        )
        key = kernel_fingerprint(self.source, json.loads(runtime_file.read_text()))
        cache = self.root / "kernels" / key
        probe = """
import torch
from eole import _ops
for i in range(torch.cuda.device_count()):
 torch.cuda.set_device(i)
 x=torch.ones(2,128,device='cuda',dtype=torch.bfloat16); out=torch.empty_like(x)
 _ops.rms_norm(out,x,torch.ones(128,device='cuda',dtype=x.dtype),1e-6)
 torch.testing.assert_close(out,x,rtol=.02,atol=.02)
"""
        self.stage("kernels", "Preparing kernels for this GPU and PyTorch version…")
        binaries = list(cache.glob("_ops*.so")) if (cache / "complete").is_file() else []
        if binaries:
            LOG.info("Reusing cached GPU extension for this runtime.")
            for path in binaries:
                shutil.copy2(path, self.source / "eole" / path.name)
        else:
            # Official image builds are GPU-less. A locally prepared image may
            # already have a compatible extension, which the probe below checks.
            if not list((self.source / "eole").glob("_ops*.so")):
                await self.command("/usr/local/bin/eole-build-kernels")
        try:
            await self.command(sys.executable, "-c", probe)
        except RuntimeError:
            for path in (self.source / "eole").glob("_ops*.so"):
                path.unlink()
            await self.command("/usr/local/bin/eole-build-kernels")
            await self.command(sys.executable, "-c", probe)
        cache.mkdir(parents=True, exist_ok=True)
        for path in (self.source / "eole").glob("_ops*.so"):
            shutil.copy2(path, cache / path.name)
        (cache / "complete").touch()

    async def model(self):
        explicit = os.environ.get("EOLE_MODEL_PATH")
        if explicit:
            model = Path(explicit).resolve()
            if not valid_model(model):
                raise ValueError("EOLE_MODEL_PATH must contain a converted Qwen checkpoint and tokenizer")
            return model
        revision = os.environ.get("EOLE_MODEL_REVISION", "main")
        identity = {"model": MODEL_ID, "revision": revision}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        model = self.root / "models" / key
        marker = model / "deployment.json"
        if marker.is_file() and valid_model(model) and json.loads(marker.read_text()) == identity:
            self.stage("download", "Using the model saved on your volume.")
            return model
        self.stage("download", "Downloading RedHat Qwen3.8-27B INT4. The first download can take several minutes…")
        snapshot = self.root / "snapshot.txt"
        await self.command(
            sys.executable,
            "-c",
            """
from huggingface_hub import snapshot_download
from pathlib import Path
import sys
path = snapshot_download(sys.argv[1], revision=sys.argv[2], ignore_patterns=['*.md', '*.png', '*.jpg'])
Path(sys.argv[3]).write_text(path)
""",
            MODEL_ID,
            revision,
            snapshot,
        )
        self.stage("convert", "Converting the downloaded INT4 weights for Eole…")
        staging = model.with_name(model.name + ".partial")
        shutil.rmtree(staging, ignore_errors=True)
        staging.parent.mkdir(parents=True, exist_ok=True)
        await self.command(
            sys.executable,
            "-m",
            "eole.bin.main",
            "convert",
            "HF",
            "--model_dir",
            snapshot.read_text().strip(),
            "--output",
            staging,
            "--dtype",
            "bf16",
            "--check-tensors",
        )
        if not valid_model(staging):
            raise RuntimeError("Converted checkpoint is incomplete; see the private Pod logs")
        config = staging / "config.json"
        config.write_text(json.dumps(relocate(json.loads(config.read_text()), staging, model), indent=2))
        (staging / "deployment.json").write_text(json.dumps(identity))
        if model.exists():
            raise RuntimeError("An incomplete destination exists; preserve/remove it manually before retrying")
        staging.rename(model)
        return model

    async def run(self):
        # One startup writer per volume; don't corrupt shared caches/models.
        import fcntl

        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "prepare.lock").open("w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("This deployment directory is in use by another Pod") from None
            await self.kernels()
            model = await self.model()
            self.stage("load", "Loading Qwen and starting the workspace. Compilation warmup runs on the first request…")
            os.environ["QWEN38_MODEL"] = str(model)
            os.environ["EOLE_MODEL_DIR"] = str(model.parent)
            config = self.root / "serve.yaml"
            settings = yaml.safe_load((self.source / "recipes/ui/serve.yaml").read_text())
            config.write_text(yaml.safe_dump(settings, sort_keys=False))
            self.process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "eole.bin.main",
                "serve",
                "-c",
                str(config),
                "--host",
                "127.0.0.1",
                "--port",
                "5001",
                cwd=self.source,
                start_new_session=True,
            )
            try:
                async with httpx.AsyncClient(trust_env=False, timeout=2) as client:
                    while self.process.returncode is None:
                        try:
                            response = await client.get("http://127.0.0.1:5001/health")
                            if response.is_success:
                                self.stage("ready", "Your Eole workspace is ready.")
                                break
                        except httpx.RequestError:
                            pass
                        await asyncio.sleep(1)
                await self.process.wait()
                raise RuntimeError("Eole stopped; see the private Pod logs and restart the Pod")
            finally:
                await stop_process(self.process)
                self.process = None


def main():
    logging.basicConfig(level=logging.INFO)
    root = Path(os.environ.get("EOLE_DEPLOY_DIR", "/workspace/eole"))
    source = Path(os.environ.get("EOLE_SOURCE_DIR", "/eole"))
    os.environ.setdefault("HF_HOME", str(root / "huggingface"))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(root / "inductor"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(root / "triton"))
    os.environ.setdefault("EOLE_TORCH_COMPILE", "1")
    os.environ.setdefault("EOLE_COMPILE_MODE", "0")
    state = {"stage": "gpu", "message": "Preparing your instance…"}
    deployment = Deployment(root, source, state)

    @asynccontextmanager
    async def prepare():
        async def supervised():
            try:
                await deployment.run()
            except asyncio.CancelledError:
                raise
            except Exception:
                LOG.exception("Deployment preparation failed")
                state.update(
                    stage="failed",
                    message="Preparation stopped. Check the private Pod logs, then restart after fixing the problem.",
                )

        task = asyncio.create_task(supervised())
        try:
            yield
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    app = create_gateway(
        os.environ.get("EOLE_USERNAME", "eole"),
        os.environ.get("EOLE_PASSWORD", ""),
        state,
        prepare,
        cookie_secure=os.environ.get("EOLE_COOKIE_SECURE", "1") != "0",
    )
    uvicorn.run(app, host="0.0.0.0", port=5000, access_log=False)


if __name__ == "__main__":
    main()
