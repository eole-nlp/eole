"""Cloud gateway and resumable preparation checks without renting a GPU."""

import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
from collections import namedtuple

from fastapi.testclient import TestClient
import httpx

from recipes.runpod.gateway import create_gateway, response_body
from recipes.runpod.auth import BrowserSessions, COOKIE_NAME, SESSION_SECONDS
from recipes.runpod.launch import (
    Deployment,
    kernel_fingerprint,
    workspace_settings,
    qwen_variants,
    filesystem_mount,
    preparation_space,
    StorageError,
    checkpoint_path,
    model_options,
    MODEL_CATALOG,
    QWEN_MODELS,
    WORKSPACE_MODELS,
)
from recipes.runpod.publish_template import template_spec

PASSWORD = "local-test-password-123"


class RawStream(httpx.AsyncByteStream):
    closed = False

    async def __aiter__(self):
        yield b"data: first\n\n"
        yield b"data: [DONE]\n\n"

    async def aclose(self):
        self.closed = True


class TestRunPodGateway(unittest.TestCase):
    def setUp(self):
        self.state = {"stage": "download", "message": "Downloading model"}
        self.requests = []
        self.stream = RawStream()

        async def upstream(request):
            self.requests.append((request, await request.aread()))
            if request.url.path == "/stream":
                return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=self.stream)
            return httpx.Response(201, stream=httpx.ByteStream(b"\x00\xffbinary"), headers={"x-upstream": "present"})

        app = create_gateway("eole", PASSWORD, self.state, transport=httpx.MockTransport(upstream))
        self.client = self.enterContext(TestClient(app, base_url="https://testserver"))
        self.client.auth = ("eole", PASSWORD)

    def test_all_paths_require_auth_before_preparation_and_after_ready(self):
        for stage in ["download", "ready", "failed"]:
            self.state["stage"] = stage
            for path in ["/_deployment/status", "/health", "/docs", "/ui/state", "/v1/models"]:
                with self.subTest(stage=stage, path=path):
                    response = self.client.get(path, auth=None)
                    self.assertEqual(response.status_code, 401)
                    self.assertIn("Bearer", response.headers["www-authenticate"])
        self.assertFalse(self.requests)

    def test_missing_or_unresolved_password_is_rejected(self):
        for password in ["", "short", "{{ RUNPOD_SECRET_eole_password }}"]:
            with self.assertRaises(ValueError):
                create_gateway("eole", password, self.state)

    def test_wrong_or_malformed_credentials_are_rejected(self):
        for header in ["Basic invalid", "Basic bm9jb2xvbg==", "Bearer wrong"]:
            self.assertEqual(
                self.client.get("/v1/models", headers={"authorization": header}, auth=None).status_code, 401
            )

    def test_browser_login_is_public_but_contains_no_workspace_state(self):
        response = self.client.get("/", auth=None)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Sign in to your workspace", response.text)
        self.assertNotIn("Downloading model", response.text)
        self.assertNotIn("www-authenticate", response.headers)
        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_browser_session_protects_ui_api_and_logout(self):
        self.client.auth = None
        bad = self.client.post("/_deployment/login", data={"username": "eole", "password": "wrong"})
        self.assertEqual(bad.status_code, 401)
        self.assertNotIn("set-cookie", bad.headers)
        login = self.client.post(
            "/_deployment/login", data={"username": "eole", "password": PASSWORD}, follow_redirects=False
        )
        self.assertEqual(login.status_code, 303)
        self.assertEqual(login.headers["location"], "/")
        cookie = login.headers["set-cookie"]
        for flag in ["HttpOnly", "Secure", "SameSite=strict", "Path=/", f"Max-Age={SESSION_SECONDS}"]:
            self.assertIn(flag, cookie)
        self.assertNotIn(PASSWORD, cookie)
        self.assertIn("Preparing your workspace", self.client.get("/").text)
        self.assertEqual(self.client.get("/_deployment/status").json(), self.state)
        self.client.post("/_deployment/logout", follow_redirects=False)
        self.assertEqual(self.client.get("/_deployment/status").status_code, 401)

    def test_invalid_session_and_cross_site_login_are_rejected(self):
        self.client.auth = None
        self.assertEqual(
            self.client.get("/_deployment/status", headers={"cookie": f"{COOKIE_NAME}=forged"}).status_code, 401
        )
        response = self.client.post(
            "/_deployment/login",
            data={"username": "eole", "password": PASSWORD},
            headers={"origin": "https://other.example"},
        )
        self.assertEqual(response.status_code, 403)
        oversized = self.client.post(
            "/_deployment/login", content=b"x" * 5000, headers={"content-type": "application/x-www-form-urlencoded"}
        )
        self.assertEqual(oversized.status_code, 413)

    def test_public_proxy_origin_login_with_rewritten_host(self):
        origin = "https://pod123-5000.proxy.runpod.net"
        app = create_gateway("eole", PASSWORD, self.state, public_origin=origin)
        with TestClient(app, base_url=origin) as client:
            data = {"username": "eole", "password": PASSWORD}
            headers = {"host": "internal-proxy:5000", "origin": origin, "accept": "application/json"}
            response = client.post("/_deployment/login", data=data, headers=headers)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"authenticated": True})
            self.assertEqual(client.get("/_deployment/status").status_code, 200)
            for bad in ["https://attacker.example", "https://pod123-5000.proxy.runpod.net.attacker.example"]:
                headers["origin"] = bad
                self.assertEqual(client.post("/_deployment/login", data=data, headers=headers).status_code, 403)

    def test_browser_sessions_expire_and_are_invalid_after_restart(self):
        sessions = BrowserSessions()
        token = sessions.issue()
        self.assertTrue(sessions.valid(token))
        self.assertFalse(sessions.valid(token + "x"))
        self.assertFalse(BrowserSessions().valid(token))
        with patch("recipes.runpod.auth.time.time", return_value=int(token.split(".")[0]) + 1):
            self.assertFalse(sessions.valid(token))

    def test_progress_and_failed_state_are_available_to_owner(self):
        self.assertIn("Preparing your workspace", self.client.get("/").text)
        self.assertEqual(self.client.get("/_deployment/status").json(), self.state)
        response = self.client.post("/v1/responses", json={})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.headers["retry-after"], "5")
        self.state.update(stage="failed", message="Check private logs")
        self.assertEqual(self.client.get("/_deployment/status").json()["stage"], "failed")

    def test_authenticated_binary_upload_proxy_and_query(self):
        self.state["stage"] = "ready"
        response = self.client.post(
            "/ui/images?mode=test",
            content=b"\x00\xffupload",
            headers={"cookie": "secret-cookie", "connection": "x-hop", "x-hop": "strip"},
        )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.content, b"\x00\xffbinary")
        self.assertEqual(response.headers["x-upstream"], "present")
        self.assertEqual(response.headers["cache-control"], "no-store")
        request, body = self.requests[-1]
        self.assertEqual(body, b"\x00\xffupload")
        self.assertEqual(str(request.url), "http://127.0.0.1:5001/ui/images?mode=test")
        for header in ["authorization", "cookie", "x-hop"]:
            self.assertNotIn(header, request.headers)

    def test_api_bearer_and_sse_are_forwarded_without_buffering_or_rewriting(self):
        self.state["stage"] = "ready"
        response = self.client.get("/stream", auth=None, headers={"authorization": f"Bearer {PASSWORD}"})
        self.assertEqual(response.content, b"data: first\n\ndata: [DONE]\n\n")
        self.assertEqual(response.headers["content-type"], "text/event-stream")
        self.assertTrue(self.stream.closed)

    def test_cross_origin_mutation_is_rejected(self):
        self.state["stage"] = "ready"
        self.assertEqual(
            self.client.post("/unload_model", headers={"origin": "https://other.example"}).status_code, 403
        )
        self.assertEqual(self.client.post("/unload_model", headers={"origin": "http://testserver"}).status_code, 201)

    def test_backend_connection_failure_is_not_exposed(self):
        self.state["stage"] = "ready"
        with patch.object(self.client.app.state.client, "send", side_effect=httpx.ConnectError("private detail")):
            response = self.client.get("/")
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("private detail", response.text)


class TestRunPodPreparation(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.deployment = Deployment(self.root, self.root, {})

    def checkpoint(self, path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(json.dumps({"tokenizer": str(path / "tokenizer.json")}))
        (path / "tokenizer.json").write_text("{}")
        (path / "model.00.safetensors").touch()

    def test_mountinfo_recognizes_bind_mounts_escaped_paths_and_nested_mounts(self):
        mounts = """1 0 0:1 / / rw - overlay overlay rw
2 1 0:1 /models /workspace rw - ext4 /dev/test rw
3 1 0:2 / /workspace/nested rw - nfs server:/models rw
4 1 0:3 / /with\\040space rw - ext4 /dev/test2 rw
"""
        self.assertEqual(filesystem_mount("/workspace/eole", mounts), (Path("/workspace"), "ext4"))
        self.assertEqual(
            filesystem_mount("/workspace/nested/eole", mounts),
            (Path("/workspace/nested"), "nfs"),
        )
        self.assertEqual(
            filesystem_mount("/with space/models", mounts),
            (Path("/with space"), "ext4"),
        )
        self.assertEqual(filesystem_mount("/workspace-other/eole", mounts)[0], Path("/"))

    async def test_missing_volume_stops_before_compilation_or_download(self):
        self.deployment.command = AsyncMock()
        with patch("pathlib.Path.read_text", return_value="1 0 0:1 / / rw - ext4 /dev/root rw"):
            with self.assertRaisesRegex(StorageError, "Add volume"):
                await self.deployment.run()
        self.deployment.command.assert_not_called()
        self.assertFalse((self.root / "prepare.lock").exists())

    def test_insufficient_space_and_volatile_cache_mount_are_rejected(self):
        mounts = f"1 0 0:1 / / rw - overlay overlay rw\n2 1 0:2 / {self.root} rw - ext4 /dev/test rw"
        usage = namedtuple("Usage", "total used free")
        with patch.dict("os.environ", {}, clear=True), patch("pathlib.Path.read_text", return_value=mounts):
            with patch("shutil.disk_usage", return_value=usage(30 * 1024**3, 0, 30 * 1024**3)):
                with self.assertRaisesRegex(StorageError, "GiB free"):
                    self.deployment.storage()
            with patch("shutil.disk_usage", return_value=usage(250 * 1024**3, 0, 250 * 1024**3)):
                self.deployment.storage()
            with patch.dict("os.environ", {"HF_HOME": "/container-cache"}):
                with self.assertRaisesRegex(StorageError, "No persistent volume"):
                    self.deployment.storage()

    def test_completed_models_reduce_storage_requirement_after_restart(self):
        with patch.dict("os.environ", {"EOLE_QWEN_VARIANT": "both"}, clear=True):
            cold = preparation_space(self.root)
            self.assertGreater(cold, 95 * 1024**3)
            self.assertLess(cold, 105 * 1024**3)
            self.assertLess(cold, 250 * 10**9)
            for role, model_id in [*QWEN_MODELS.items(), *WORKSPACE_MODELS.items()]:
                _, identity = model_options(model_id, role)
                model = checkpoint_path(self.root, identity)
                self.checkpoint(model)
                (model / "deployment.json").write_text(json.dumps(identity))
            self.assertEqual(preparation_space(self.root), 8 * 1024**3)

    async def test_download_is_atomic_cached_and_revision_pinned(self):
        calls = []

        async def command(*args):
            calls.append(args)
            if args[2] == "download":
                self.checkpoint(Path(args[-1]))

        self.deployment.command = command
        with patch.dict("os.environ", {}, clear=True):
            model = await self.deployment.model()
            self.assertTrue((model / "deployment.json").is_file())
            self.assertEqual([args[2] for args in calls], ["download", "verify"])
            self.assertEqual(calls[0][-2], MODEL_CATALOG["int4"]["revision"])
            self.assertEqual(await self.deployment.model(), model)
            self.assertEqual(len(calls), 2)
            with patch.dict("os.environ", {"EOLE_MODEL_REVISION": "a" * 40}):
                self.assertNotEqual(await self.deployment.model(), model)
            self.assertEqual(len(calls), 4)
            with patch.dict("os.environ", {"EOLE_MODEL_REVISION": "main"}):
                with self.assertRaisesRegex(ValueError, "full Hub commit"):
                    await self.deployment.model()

    async def test_interrupted_download_resumes_and_verification_failure_is_not_cached(
        self,
    ):
        async def command(*args):
            path = Path(args[-1])
            if args[2] == "download":
                if not (path / "config.json").exists():
                    (path / "config.json").write_text("{}")
                    raise RuntimeError("interrupted")
                self.assertTrue((path / "config.json").exists())
            else:
                raise RuntimeError("checksum mismatch")

        self.deployment.command = command
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                await self.deployment.model()
            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                await self.deployment.model()
        self.assertFalse(list(self.root.rglob("deployment.json")))
        self.assertTrue(list(self.root.rglob("*.partial")))

    async def test_explicit_existing_model_skips_download(self):
        path = self.root / "existing"
        self.checkpoint(path)
        self.deployment.command = AsyncMock()
        with patch.dict("os.environ", {"EOLE_MODEL_PATH": str(path)}):
            self.assertEqual(await self.deployment.model(), path)
        self.deployment.command.assert_not_called()

    async def test_cancelled_build_terminates_its_process(self):
        task = asyncio.create_task(self.deployment.command("sleep", "60"))
        for _ in range(100):
            if self.deployment.process:
                break
            await asyncio.sleep(0.01)
        process = self.deployment.process
        self.assertIsNotNone(process)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertIsNotNone(process.returncode)

    async def test_sse_heartbeats_during_warmup_and_stream_closes_on_disconnect(self):
        class SlowStream(httpx.AsyncByteStream):
            closed = False

            async def __aiter__(self):
                await asyncio.sleep(60)
                yield b"data: token\n\n"

            async def aclose(self):
                self.closed = True

        stream = SlowStream()
        response = httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"})
        with patch("recipes.runpod.gateway.SSE_HEARTBEAT_SECONDS", 0.01):
            body = response_body(response)
            self.assertEqual(await anext(body), b": eole keepalive\n\n")
            await body.aclose()
        self.assertTrue(stream.closed)

    def test_cache_key_changes_with_gpu_torch_and_source(self):
        (self.root / "setup.py").write_text("source-v1")
        key = kernel_fingerprint(self.root, {"torch": "2.14.1", "architectures": [[12, 0]]})
        self.assertNotEqual(
            key,
            kernel_fingerprint(self.root, {"torch": "2.14.1", "architectures": [[9, 0]]}),
        )
        self.assertNotEqual(
            key,
            kernel_fingerprint(self.root, {"torch": "2.13.0", "architectures": [[12, 0]]}),
        )
        (self.root / "setup.py").write_text("source-v2")
        self.assertNotEqual(
            key,
            kernel_fingerprint(self.root, {"torch": "2.14.1", "architectures": [[12, 0]]}),
        )

    async def test_kernel_cache_is_reused_and_invalid_binary_is_rebuilt(self):
        (self.root / "eole").mkdir()
        (self.root / "csrc").mkdir()
        (self.root / "setup.py").write_text("source")
        calls = []
        fail_probe = False

        async def command(*args):
            nonlocal fail_probe
            calls.append(args)
            if self.root / "runtime.json" in args:
                (self.root / "runtime.json").write_text(json.dumps({"torch": "2.14.1", "gpu": "12.0"}))
            elif args[0] == "/usr/local/bin/eole-build-kernels":
                (self.root / "eole" / "_ops.test.so").write_bytes(b"built")
            elif fail_probe or not list((self.root / "eole").glob("_ops*.so")):
                fail_probe = False
                raise RuntimeError("missing or bad binary")

        self.deployment.command = command
        await self.deployment.kernels()
        self.assertEqual(sum(args[0] == "/usr/local/bin/eole-build-kernels" for args in calls), 1)
        (self.root / "eole" / "_ops.test.so").unlink()
        calls.clear()
        await self.deployment.kernels()
        self.assertFalse(any(args[0] == "/usr/local/bin/eole-build-kernels" for args in calls))
        self.assertEqual((self.root / "eole" / "_ops.test.so").read_bytes(), b"built")
        calls.clear()
        fail_probe = True
        await self.deployment.kernels()
        self.assertEqual(sum(args[0] == "/usr/local/bin/eole-build-kernels" for args in calls), 1)

    async def test_bundled_kernel_is_probed_before_cache_without_building(self):
        (self.root / "eole").mkdir()
        (self.root / "eole" / "_ops.test.so").write_bytes(b"bundled")
        calls = []

        async def command(*args):
            calls.append(args)
            if self.root / "runtime.json" in args:
                (self.root / "runtime.json").write_text(json.dumps({"torch": "2.14.1", "gpu": "12.0"}))

        self.deployment.command = command
        await self.deployment.kernels()
        for cached in (self.root / "kernels").glob("*/_ops*.so"):
            cached.write_bytes(b"stale")
        calls.clear()
        await self.deployment.kernels()
        self.assertFalse(any(args[0] == "/usr/local/bin/eole-build-kernels" for args in calls))
        self.assertEqual((self.root / "eole" / "_ops.test.so").read_bytes(), b"bundled")

    def test_workspace_routes_models_and_enforces_single_resident_model(self):
        models = {role: self.root / role for role in ["int4", "bf16", "translate", "ocr"]}
        source = Path(__file__).resolve().parents[2]
        for choice, ids in [
            ("int4", ["qwen3.8-27B"]),
            ("bf16", ["qwen3.8-27B-bf16"]),
            ("both", ["qwen3.8-27B", "qwen3.8-27B-bf16"]),
        ]:
            with patch.dict("os.environ", {"EOLE_QWEN_VARIANT": choice}):
                settings = workspace_settings(source, models)
            self.assertEqual(settings["model_policy"], "single")
            self.assertEqual(
                [m["id"] for m in settings["models"]],
                ids + ["hy-mt2-7b", "deepseek-ocr"],
            )
            self.assertEqual([m["id"] for m in settings["models"] if m["preload"]], [ids[0]])
            self.assertTrue(all(m["warmup"] for m in settings["models"][: len(ids)]))
            self.assertTrue(
                all(m["config"]["vocabulary_quantization"] == "int4" for m in settings["models"][: len(ids)])
            )
            for entry in settings["models"][: len(ids)]:
                self.assertEqual(entry["dflash2"]["path"], str(Path(entry["path"]) / "dflash2"))
                self.assertNotIn("dflash2_model", entry["config"])
            self.assertEqual(settings["workspace"]["models"]["chat"], ids[0])
            self.assertEqual(settings["workspace"]["models"]["ocr"], "deepseek-ocr")
            self.assertEqual(settings["workspace"]["models"]["translate"], "hy-mt2-7b")
            self.assertNotIn("${", json.dumps(settings))
        with patch.dict("os.environ", {"EOLE_QWEN_VARIANT": "unknown"}):
            with self.assertRaises(ValueError):
                qwen_variants()

    async def test_all_models_have_independent_download_caches_and_tokenizers(self):
        calls = []

        async def command(*args):
            calls.append(args)
            if args[2] == "download":
                self.checkpoint(Path(args[-1]))

        self.deployment.command = command
        with patch.dict("os.environ", {"EOLE_QWEN_VARIANT": "both"}, clear=True):
            checkpoints = []
            for role, model_id in [*QWEN_MODELS.items(), *WORKSPACE_MODELS.items()]:
                model = await self.deployment.model(model_id, role)
                checkpoints.append(model)
                identity = json.loads((model / "deployment.json").read_text())
                self.assertEqual(identity["model"], model_id)
                self.assertEqual(identity["revision"], MODEL_CATALOG[role]["revision"])
                self.assertTrue((model / "tokenizer.json").is_file())
                self.assertEqual(await self.deployment.model(model_id, role), model)
            self.assertEqual(len(set(checkpoints)), 4)
            self.assertEqual(len(calls), 8)

    def test_catalog_pins_all_models_and_release_template_packages_them(self):
        self.assertEqual(set(MODEL_CATALOG), {"int4", "bf16", "translate", "ocr"})
        for model in MODEL_CATALOG.values():
            self.assertTrue(model["repo"].startswith("eole-nlp/"))
            self.assertRegex(model["revision"], r"^[0-9a-f]{40}$")
            self.assertGreater(model["size_bytes"], 1024**3)
        recipe = Path(__file__).resolve().parents[2] / "recipes/runpod"
        template = json.loads((recipe / "template.json").read_text())
        self.assertEqual(template["imageName"], "ghcr.io/eole-nlp/eole-runpod:0.6.5")
        self.assertIn('data-stage="verify"', (recipe / "preparing.html").read_text())
        self.assertNotIn('data-stage="convert"', (recipe / "preparing.html").read_text())

    def test_template_never_contains_a_shared_password_or_rents_a_pod(self):
        spec = template_spec("ghcr.io/eole-nlp/eole-runpod:dev-123456789abc")
        self.assertFalse(spec["isPublic"])
        self.assertFalse(spec["isServerless"])
        self.assertEqual(spec["ports"], ["5000/http"])
        self.assertEqual(spec["env"]["EOLE_PASSWORD"], "{{ RUNPOD_SECRET_eole_password }}")
        self.assertNotIn("5001", json.dumps(spec))
        with self.assertRaises(ValueError):
            template_spec("ghcr.io/eole-nlp/eole-runpod:latest")
