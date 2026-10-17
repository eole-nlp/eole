"""Cloud gateway and resumable preparation checks without renting a GPU."""

import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
import httpx

from recipes.runpod.gateway import create_gateway, response_body
from recipes.runpod.auth import BrowserSessions, COOKIE_NAME, SESSION_SECONDS
from recipes.runpod.launch import Deployment, kernel_fingerprint, relocate, workspace_settings, qwen_variants
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
        path.mkdir(parents=True)
        (path / "config.json").write_text(json.dumps({"tokenizer": str(path / "tokenizer.json")}))
        (path / "tokenizer.json").write_text("{}")
        (path / "model.00.safetensors").touch()

    async def test_conversion_is_atomic_and_cached_and_paths_are_relocated(self):
        calls = []

        async def command(*args):
            calls.append(args)
            if "--output" in args:
                self.checkpoint(Path(args[args.index("--output") + 1]))
            else:
                snapshot = self.root / "snapshot"
                snapshot.mkdir(exist_ok=True)
                (snapshot / "tokenizer.json").write_text("{}")
                (self.root / "snapshot.txt").write_text(str(snapshot))

        self.deployment.command = command
        with patch.dict("os.environ", {"EOLE_MODEL_REVISION": "pinned-commit"}):
            with patch.dict("os.environ") as env:
                env.pop("EOLE_MODEL_PATH", None)
                model = await self.deployment.model()
                self.assertTrue((model / "deployment.json").is_file())
                self.assertEqual(
                    json.loads((model / "config.json").read_text())["tokenizer"], str(model / "tokenizer.json")
                )
                self.assertIn("--tokenizer", calls[1])
                self.assertEqual(
                    json.loads((model / "config.json").read_text())["transforms_configs"]["huggingface_tokenize"][
                        "path"
                    ],
                    str(model / "tokenizer.json"),
                )
                self.assertEqual(len(calls), 2)
                self.assertEqual(await self.deployment.model(), model)
                self.assertEqual(len(calls), 2)
                env["EOLE_MODEL_REVISION"] = "another-commit"
                self.assertNotEqual(await self.deployment.model(), model)
                self.assertEqual(len(calls), 4)

    async def test_interrupted_conversion_is_not_marked_complete(self):
        async def command(*args):
            if "--output" in args:
                path = Path(args[args.index("--output") + 1])
                path.mkdir()
                (path / "config.json").write_text("{}")
                raise RuntimeError("interrupted")
            (self.root / "snapshot.txt").write_text("/cached/hf-snapshot")

        self.deployment.command = command
        with patch.dict("os.environ") as env:
            env.pop("EOLE_MODEL_PATH", None)
            with self.assertRaises(RuntimeError):
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
        self.assertNotEqual(key, kernel_fingerprint(self.root, {"torch": "2.14.1", "architectures": [[9, 0]]}))
        self.assertNotEqual(key, kernel_fingerprint(self.root, {"torch": "2.13.0", "architectures": [[12, 0]]}))
        (self.root / "setup.py").write_text("source-v2")
        self.assertNotEqual(key, kernel_fingerprint(self.root, {"torch": "2.14.1", "architectures": [[12, 0]]}))

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
            elif fail_probe:
                fail_probe = False
                raise RuntimeError("bad cached binary")

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
            self.assertEqual([m["id"] for m in settings["models"]], ids + ["hy-mt2-7b", "deepseek-ocr"])
            self.assertFalse(any(m["preload"] for m in settings["models"]))
            self.assertEqual(settings["workspace"]["models"]["chat"], ids[0])
            self.assertEqual(settings["workspace"]["models"]["ocr"], "deepseek-ocr")
            self.assertEqual(settings["workspace"]["models"]["translate"], "hy-mt2-7b")
            self.assertNotIn("${", json.dumps(settings))
        with patch.dict("os.environ", {"EOLE_QWEN_VARIANT": "unknown"}):
            with self.assertRaises(ValueError):
                qwen_variants()

    async def test_all_models_have_independent_conversion_caches_and_tokenizers(self):
        calls = []

        async def command(*args):
            calls.append(args)
            if "--output" in args:
                output = Path(args[args.index("--output") + 1])
                self.checkpoint(output)
                (output / "tokenizer.json").unlink()  # Real HF conversion references its source cache.
            else:
                snapshot = self.root / ("snapshot-" + str(len(calls)))
                snapshot.mkdir()
                (snapshot / "tokenizer.json").write_text("{}")
                Path(args[-1]).write_text(str(snapshot))

        self.deployment.command = command
        with patch.dict("os.environ", {"EOLE_QWEN_VARIANT": "both"}, clear=True):
            checkpoints = []
            for role, model_id in [
                ("int4", "RedHatAI/Qwen3.8-27B-INT4"),
                ("bf16", "Qwen/Qwen3.8-27B"),
                ("translate", "tencent/Hy-MT2-7B"),
                ("ocr", "deepseek-ai/DeepSeek-OCR"),
            ]:
                model = await self.deployment.model(model_id, role)
                checkpoints.append(model)
                self.assertEqual(json.loads((model / "deployment.json").read_text())["model"], model_id)
                self.assertTrue((model / "tokenizer.json").is_file())
                self.assertEqual(await self.deployment.model(model_id, role), model)
            self.assertEqual(len(set(checkpoints)), 4)
            self.assertEqual(len(calls), 8)

    def test_template_never_contains_a_shared_password_or_rents_a_pod(self):
        spec = template_spec("ghcr.io/eole-nlp/eole-runpod:dev-123456789abc")
        self.assertFalse(spec["isPublic"])
        self.assertFalse(spec["isServerless"])
        self.assertEqual(spec["ports"], ["5000/http"])
        self.assertEqual(spec["env"]["EOLE_PASSWORD"], "{{ RUNPOD_SECRET_eole_password }}")
        self.assertNotIn("5001", json.dumps(spec))
        with self.assertRaises(ValueError):
            template_spec("ghcr.io/eole-nlp/eole-runpod:latest")

    def test_path_relocation_does_not_change_other_paths(self):
        self.assertEqual(
            relocate(["/stage/tokenizer", "/stage-other/tokenizer"], Path("/stage"), Path("/final")),
            ["/final/tokenizer", "/stage-other/tokenizer"],
        )
