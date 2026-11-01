"""Workspace speculation selection, reload boundaries and safe request fallbacks."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from eole.server.model import Model as ServerModel
from eole.server.ui import register_ui
from eole.server.ui_runtime import Workspace
from eole.server.ui_speculation import capabilities
from eole.tests.test_server_ui import Model


class SpeculationSetup:
    def setup_model(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name)
        (self.folder / "config.json").write_text(
            json.dumps({"architectures": ["DFlash2DraftModel"], "dflash_config": {"block_size": 8}})
        )
        (self.folder / "model.safetensors").touch()
        self.model = Model()
        self.model.pre_config = dict(
            self_speculative_decoding=True, vocabulary_quantization="int4", batch_size=1, beam_size=1, top_k=1
        )
        self.model.speculation_defaults = dict(mode="mtp", mtp=True, path=str(self.folder), quantization="int8")
        self.model.engine.config = SimpleNamespace(
            world_size=1,
            self_speculative_decoding=True,
            compute_dtype="bf16",
            context_length=128,
            max_length=32,
            vocabulary_quantization="int4",
            dflash2_model=None,
            dflash2_quantization="none",
        )
        self.calls = []

        async def unload(model_id):
            self.calls.append("unload")
            self.model.loaded = False

        async def prepare(model_id, row):
            self.calls.append("prepare")
            self.model.loaded = True
            for name, value in self.model.pre_config.items():
                setattr(self.model.engine.config, name, value)

        self.server = SimpleNamespace(
            models={"test": self.model},
            unload_model=unload,
            prepare_model=prepare,
        )


class TestSpeculationReloads(SpeculationSetup, unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.setup_model()

    async def test_head_and_drafter_changes_share_one_reload(self):
        await Workspace(self.server).prepare(
            "test",
            self.model,
            {},
            vocabulary_quantization="int8",
            speculation="dflash2",
            dflash2_quantization="int8",
        )
        self.assertEqual(self.calls, ["unload", "prepare"])
        self.assertEqual(self.model.pre_config["dflash2_model"], str(self.folder))
        self.assertEqual(self.model.pre_config["vocabulary_quantization"], "int8")
        self.calls.clear()
        await Workspace(self.server).prepare("test", self.model, {}, speculation="dflash2")
        self.assertEqual(self.calls, ["prepare"])
        self.calls.clear()
        await Workspace(self.server).prepare("test", self.model, {}, speculation="dflash2", dflash2_quantization="none")
        self.assertEqual(self.calls, ["unload", "prepare"])

    async def test_exit_dflash2_and_preserve_choices(self):
        workspace = Workspace(self.server)
        await workspace.prepare("test", self.model, {}, speculation="dflash2")
        self.calls.clear()
        await workspace.prepare("test", self.model, {}, speculation="off")
        self.assertEqual(self.calls, ["unload", "prepare"])
        self.assertIsNone(self.model.engine.config.dflash2_model)
        self.assertTrue(capabilities(self.model)["dflash2_available"])
        self.calls.clear()
        await workspace.prepare("test", self.model, {}, speculation="mtp")
        self.assertEqual(self.calls, ["prepare"])

    async def test_text_workspace_keeps_loaded_drafter(self):
        workspace = Workspace(self.server)
        await workspace.prepare("test", self.model, {}, speculation="dflash2")
        self.calls.clear()
        await workspace.prepare("test", self.model, {})
        self.assertEqual(self.model.engine.config.dflash2_model, str(self.folder))
        self.assertEqual(self.calls, ["prepare"])

    async def test_invalid_drafter_metadata_and_compute_are_unavailable(self):
        self.model.engine.config.compute_dtype = "fp32"
        self.assertFalse(capabilities(self.model)["dflash2_available"])
        self.model.engine.config.compute_dtype = "bf16"
        for metadata in ([], {"dflash_config": None}, {"architectures": ["OtherModel"]}):
            (self.folder / "config.json").write_text(json.dumps(metadata))
            self.assertFalse(capabilities(self.model)["dflash2_available"])

    async def test_missing_checkpoint_is_rejected_before_unload(self):
        (self.folder / "model.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            await Workspace(self.server).prepare("test", self.model, {}, speculation="dflash2")
        self.assertEqual(self.calls, [])

    async def test_failed_switch_restores_config_for_lazy_retry(self):
        previous = self.model.pre_config
        original_prepare = self.server.prepare_model

        async def fail(model_id, row):
            raise RuntimeError("load failed")

        self.server.prepare_model = fail
        with self.assertRaisesRegex(RuntimeError, "load failed"):
            await Workspace(self.server).prepare("test", self.model, {}, speculation="dflash2")
        self.assertIs(self.model.pre_config, previous)
        self.assertFalse(self.model.loaded)
        self.server.prepare_model = original_prepare
        await Workspace(self.server).prepare("test", self.model, {})
        self.assertTrue(self.model.loaded)
        self.assertIsNone(self.model.engine.config.dflash2_model)

    async def test_incompatible_defaults_are_rejected_before_unload(self):
        for changes in ({"batch_size": 30}, {"beam_size": 5}, {"top_k": 0, "temperature": 1}):
            previous = self.model.pre_config.copy()
            self.model.pre_config.update(changes)
            with self.assertRaises(ValueError):
                await Workspace(self.server).prepare("test", self.model, {}, speculation="dflash2")
            self.assertEqual(self.calls, [])
            self.model.pre_config = previous
        self.model.pre_config.update(top_k=0, temperature=0)
        self.assertTrue(capabilities(self.model)["dflash2_available"])
        for key in ("batch_size", "beam_size", "top_k", "temperature"):
            self.model.pre_config.pop(key, None)
        self.assertFalse(capabilities(self.model)["dflash2_available"])


class TestSpeculationRoutes(SpeculationSetup, unittest.TestCase):
    def setUp(self):
        self.setup_model()
        app = FastAPI()
        register_ui(app, self.server)
        self.client = TestClient(app)
        self.payload = dict(model="test", max_tokens=16, messages=[dict(role="user", content="Hello")])

    def events(self, **changes):
        result = self.client.post("/ui/chat", json={**self.payload, **changes})
        self.assertEqual(result.status_code, 200)
        return [json.loads(line[6:]) for line in result.text.splitlines() if line.startswith("data: ")]

    def test_capabilities_and_invalid_settings(self):
        with patch("eole.server.ui.Hardware.read", return_value={"gpus": []}):
            options = self.client.get("/ui/state").json()["models"][0]["speculation"]
        self.assertTrue(options["dflash2_available"])
        self.assertEqual(options["dflash2_drafts"], 7)
        for invalid in [dict(speculation="unknown"), dict(dflash2_quantization="int4")]:
            self.assertEqual(self.client.post("/ui/chat", json={**self.payload, **invalid}).status_code, 422)

    def test_dflash2_settings_and_creative_fallback(self):
        self.assertEqual(self.events(speculation="dflash2")[-1]["type"], "done")
        self.assertTrue(self.model.engine.settings["self_speculative_decoding"])
        self.assertEqual(self.model.engine.config.dflash2_model, str(self.folder))
        self.assertEqual(self.events(speculation="dflash2", preset="creative")[-1]["type"], "done")
        self.assertIsNone(self.model.engine.config.dflash2_model)
        self.assertFalse(self.model.engine.settings["self_speculative_decoding"])

    def test_image_history_uses_ordinary_engine(self):
        self.events(speculation="dflash2")
        image = dict(name="image.png", data="test-image-data")
        messages = [dict(role="user", content="Describe", images=[image])]
        with patch("eole.server.ui.chat_prompt", return_value=(dict(text="image prompt", images={}), "image prompt")):
            result = self.events(speculation="dflash2", messages=messages)
        self.assertEqual(result[-1]["type"], "done")
        self.assertIsNone(self.model.engine.config.dflash2_model)
        self.assertFalse(self.model.engine.settings["self_speculative_decoding"])


class TestModelSpeculationDefaults(unittest.TestCase):
    def test_metadata_does_not_enable_dflash2_at_startup(self):
        model = ServerModel(pre_config={"self_speculative_decoding": True}, dflash2_config={"path": "/models/drafter"})
        self.assertEqual(model.speculation_defaults["mode"], "mtp")
        self.assertNotIn("dflash2_model", model.pre_config)
        with self.assertRaises(ValueError):
            ServerModel(dflash2_config={"path": "/models/drafter", "quantization": "int4"})
