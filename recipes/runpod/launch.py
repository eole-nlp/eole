"""Prepare a multi-model workspace Pod, then supervise a private Eole server."""

import asyncio
from contextlib import asynccontextmanager, suppress
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import signal
import sys

import httpx
import uvicorn
import yaml

try:
    from .gateway import create_gateway
    from .checkpoints import DFLASH2
except ImportError:
    from gateway import create_gateway
    from checkpoints import DFLASH2

LOG = logging.getLogger("eole.runpod")
MODEL_CATALOG = json.loads((Path(__file__).resolve().parent / "models.json").read_text())
MODEL_ID = MODEL_CATALOG["int4"]["repo"]
RECIPE = Path(__file__).resolve().parent
QWEN_MODELS = {role: MODEL_CATALOG[role]["repo"] for role in ("int4", "bf16")}
WORKSPACE_MODELS = {role: MODEL_CATALOG[role]["repo"] for role in ("translate", "ocr")}
GIB = 1024**3
STORAGE_HELP = "In RunPod deployment, choose Add volume → Volume disk, set 250 GB and mount it at /workspace."


class StorageError(RuntimeError):
    """An actionable preparation error safe to show to the authenticated owner."""


def filesystem_mount(path, mountinfo):
    """Use mountinfo rather than st_dev so same-device bind mounts are recognized."""
    path = Path(path).resolve()
    matches = []
    for line in mountinfo.splitlines():
        before, separator, after = line.partition(" - ")
        fields = before.split()
        if separator and len(fields) >= 5 and after.split():
            mount = Path(re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), fields[4]))
            if path == mount or mount in path.parents:
                matches.append((mount, after.split()[0]))
    return max(matches, key=lambda pair: len(pair[0].parts)) if matches else (Path("/"), "unknown")


def model_options(model_id, role):
    prefix = {
        "int4": "EOLE_MODEL",
        "bf16": "EOLE_BF16_MODEL",
        "translate": "EOLE_TRANSLATION_MODEL",
        "ocr": "EOLE_OCR_MODEL",
    }[role]
    if role == "bf16" and qwen_variants() == ["bf16"]:
        prefix = "EOLE_MODEL"
    revision = os.environ.get(prefix + "_REVISION", MODEL_CATALOG[role]["revision"])
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(prefix + "_REVISION must be a full Hub commit SHA")
    identity = {"model": model_id, "revision": revision, "format": "eole"}
    return os.environ.get(prefix + "_PATH"), identity


def checkpoint_path(root, identity):
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    return root / "models" / key


def completed_checkpoint(model, identity):
    marker = model / "deployment.json"
    return marker.is_file() and valid_model(model) and json.loads(marker.read_text()) == identity


def preparation_space(root):
    # Download directly into the final filesystem: no upstream weight cache or
    # conversion output duplicates. Existing/partial files already consume disk;
    # conservatively reserve a complete missing checkpoint plus cache headroom.
    needed = 8 * GIB
    selected = [(v, QWEN_MODELS[v]) for v in qwen_variants()] + list(WORKSPACE_MODELS.items())
    for role, model_id in selected:
        explicit, identity = model_options(model_id, role)
        if not explicit and not completed_checkpoint(checkpoint_path(root, identity), identity):
            needed += MODEL_CATALOG[role]["size_bytes"]
    if not completed_drafter(drafter_path(root)):
        needed += sum(file["size"] for file in DFLASH2["files"].values())
    return needed


def drafter_path(root):
    return checkpoint_path(root, DFLASH2)


def completed_drafter(path):
    marker = path / "deployment.json"
    return (
        marker.is_file()
        and json.loads(marker.read_text()) == DFLASH2
        and all(
            (path / name).is_file() and (path / name).stat().st_size == metadata["size"]
            for name, metadata in DFLASH2["files"].items()
        )
    )


def qwen_variants():
    choice = os.environ.get("EOLE_QWEN_VARIANT", "int4").lower()
    if choice not in {"int4", "bf16", "both"}:
        raise ValueError("EOLE_QWEN_VARIANT must be int4, bf16 or both")
    return ["int4", "bf16"] if choice == "both" else [choice]


def workspace_settings(source, models):
    settings = yaml.safe_load((source / "recipes/ui/serve-showcase.yaml").read_text())
    settings["models_root"] = str(next(iter(models.values())).parent)
    qwen, translation, ocr = settings["models"]
    entries = []
    for variant in qwen_variants():
        entry = json.loads(json.dumps(qwen))
        entry["id"] = "qwen3.8-27B" if variant == "int4" else "qwen3.8-27B-bf16"
        entry["path"] = str(models[variant])
        if entry.get("dflash2"):
            entry["dflash2"]["path"] = str(models.get("dflash2", models[variant] / "dflash2"))
        entries.append(entry)
    translation["path"] = str(models["translate"])
    ocr["path"] = str(models["ocr"])
    settings["models"] = [*entries, translation, ocr]
    # Prepare only the default chat model; other models still load on selection.
    # Keeping this engine alive reuses the just-compiled verifier immediately.
    entries[0]["preload"] = True
    for entry in entries:
        entry["warmup"] = True
    settings["workspace"]["models"].update(chat=entries[0]["id"], knowledge=entries[0]["id"])
    return settings


def valid_model(path):
    return (
        (path / "config.json").is_file()
        and (path / "tokenizer.json").is_file()
        and any(path.glob("model.*.safetensors"))
    )


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

    def storage(self):
        self.stage("storage", "Checking persistent storage before GPU preparation…")
        mountinfo = Path("/proc/self/mountinfo").read_text()
        paths = [
            self.root,
            *[
                Path(os.environ[key])
                for key in ("HF_HOME", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "EOLE_VERIFIER_CACHE_DIR")
                if os.environ.get(key)
            ],
        ]
        for path in paths:
            mount, filesystem = filesystem_mount(path, mountinfo)
            if mount == Path("/") or filesystem in {"overlay", "tmpfs", "ramfs", "unknown"}:
                raise StorageError(
                    f"No persistent volume is mounted for {path}. {STORAGE_HELP} "
                    "The 60 GB container disk is not the model volume; preparation has not started."
                )
        self.root.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(self.root).free
        needed = preparation_space(self.root)
        if free < needed:
            raise StorageError(
                f"Model storage has {free / GIB:.1f} GiB free; approximately "
                f"{needed / GIB:.1f} GiB is needed for remaining preparation. {STORAGE_HELP}"
            )
        LOG.info(
            "Persistent storage verified: %.1f GiB reported free; %.1f GiB estimated needed.", free / GIB, needed / GIB
        )

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
required = int(sys.argv[2])
assert torch.cuda.get_device_properties(0).total_memory >= required * 1024**3, (
    f'Choose at least {required + 2} GB advertised VRAM'
)
info = dict(torch=torch.__version__, python=sys.version, machine=platform.machine(),
            cuda=torch.version.cuda, nvcc=subprocess.check_output(['nvcc','--version'],text=True),
            architectures=sorted(set(torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count()))))
open(sys.argv[1],'w').write(json.dumps(info))
""",
            runtime_file,
            78 if "bf16" in qwen_variants() else 30,
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
        self.stage("kernels", "Checking bundled GPU kernels…")
        # Prefer the image's multi-architecture binary over a deployment cache.
        # This also avoids overwriting a good bundled binary with stale cache data.
        bundled = list((self.source / "eole").glob("_ops*.so"))
        binaries = list(cache.glob("_ops*.so")) if (cache / "complete").is_file() else []
        if not bundled and binaries:
            LOG.info("Reusing cached GPU extension for this runtime.")
            for path in binaries:
                shutil.copy2(path, self.source / "eole" / path.name)
        try:
            await self.command(sys.executable, "-c", probe)
        except RuntimeError:
            LOG.warning("GPU extension unavailable or incompatible; building for visible GPUs.")
            self.stage("kernels", "Building kernels for this GPU and PyTorch version…")
            for path in (self.source / "eole").glob("_ops*.so"):
                path.unlink()
            # The image architecture list is a build argument, not a runtime ENV;
            # fallback builds therefore target the devices actually visible here.
            await self.command("/usr/local/bin/eole-build-kernels")
            await self.command(sys.executable, "-c", probe)
        cache.mkdir(parents=True, exist_ok=True)
        for path in (self.source / "eole").glob("_ops*.so"):
            shutil.copy2(path, cache / path.name)
        (cache / "complete").touch()

    async def model(self, model_id=MODEL_ID, role="int4"):
        explicit, identity = model_options(model_id, role)
        if explicit:
            model = Path(explicit).resolve()
            if not valid_model(model):
                raise ValueError("Existing model path must contain a converted checkpoint and tokenizer")
            return model
        model = checkpoint_path(self.root, identity)
        if completed_checkpoint(model, identity):
            self.stage("download", f"Using {model_id} saved on your volume.")
            return model
        staging = model.with_name(model.name + ".partial")
        staging.mkdir(parents=True, exist_ok=True)
        helper = RECIPE / "checkpoints.py"
        self.stage("download", f"Downloading pre-converted {model_id}. The first download can take several minutes…")
        await self.command(sys.executable, helper, "download", model_id, identity["revision"], staging)
        self.stage("verify", f"Verifying downloaded files for {model_id}…")
        await self.command(sys.executable, helper, "verify", model_id, identity["revision"], staging)
        if not valid_model(staging):
            raise RuntimeError("Downloaded checkpoint is incomplete; see the private Pod logs")
        (staging / "deployment.json").write_text(json.dumps(identity))
        if model.exists():
            raise RuntimeError("An incomplete destination exists; preserve/remove it manually before retrying")
        staging.rename(model)
        return model

    async def drafter(self):
        # One immutable drafter is shared by INT4 and BF16 targets on this volume.
        path = drafter_path(self.root)
        if completed_drafter(path):
            self.stage("download", "Using the DFlash2 drafter saved on your volume.")
            return path
        staging = path.with_name(path.name + ".partial")
        staging.mkdir(parents=True, exist_ok=True)
        helper = RECIPE / "checkpoints.py"
        self.stage("download", "Downloading the pinned DFlash2 drafter (3.85 GB)…")
        await self.command(sys.executable, helper, "download-drafter", DFLASH2["repo"], DFLASH2["revision"], staging)
        self.stage("verify", "Verifying DFlash2 config and weights…")
        await self.command(sys.executable, helper, "verify-drafter", DFLASH2["repo"], DFLASH2["revision"], staging)
        (staging / "deployment.json").write_text(json.dumps(DFLASH2))
        if path.exists():
            raise RuntimeError("An incomplete drafter destination exists; preserve/remove it manually before retrying")
        staging.rename(path)
        return path

    async def run(self):
        # One startup writer per volume; don't corrupt shared caches/models.
        import fcntl

        self.storage()
        with (self.root / "prepare.lock").open("w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("This deployment directory is in use by another Pod") from None
            await self.kernels()
            models = {}
            for variant in qwen_variants():
                models[variant] = await self.model(QWEN_MODELS[variant], variant)
            for role, model_id in WORKSPACE_MODELS.items():
                models[role] = await self.model(model_id, role)
            models["dflash2"] = await self.drafter()
            self.stage("load", "Loading Qwen and preparing inference: reuse the local verifier or compile it once…")
            config = self.root / "serve.yaml"
            settings = workspace_settings(self.source, models)
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
    os.environ.setdefault("EOLE_VERIFIER_CACHE_DIR", str(root / "verifiers"))
    os.environ.setdefault("EOLE_TORCH_COMPILE", "1")
    os.environ.setdefault("EOLE_COMPILE_MODE", "0")
    state = {"stage": "storage", "message": "Checking persistent storage…"}
    deployment = Deployment(root, source, state)

    @asynccontextmanager
    async def prepare():
        async def supervised():
            try:
                await deployment.run()
            except asyncio.CancelledError:
                raise
            except StorageError as exc:
                LOG.error("Deployment storage check failed: %s", exc)
                state.update(stage="failed", message=str(exc))
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
        public_origin=(
            f"https://{os.environ["RUNPOD_POD_ID"]}-5000.proxy.runpod.net" if os.environ.get("RUNPOD_POD_ID") else None
        ),
    )
    uvicorn.run(app, host="0.0.0.0", port=5000, access_log=False)


if __name__ == "__main__":
    main()
