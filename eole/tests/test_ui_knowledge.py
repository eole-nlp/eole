"""Knowledge ownership, request lifecycle, citations and bounded document inputs."""

import asyncio
import base64
from importlib.util import find_spec
import json
from types import SimpleNamespace
import threading
import unittest

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from eole.server.ui import register_ui
from eole.server.ui_knowledge import Knowledge, Upload, Question, build_prompt, cpu_job, load_documents
from eole.tests import test_server_ui


class TestCollections(unittest.TestCase):
    def test_browser_renderer(self):
        import shutil
        import subprocess
        from pathlib import Path

        if not shutil.which("node"):
            self.skipTest("Optional Node runtime absent")
        subprocess.run(["node", "eole/tests/web/test_knowledge.cjs"], cwd=Path(__file__).parents[2], check=True)

    def setUp(self):
        self.store = Knowledge()
        self.token = self.store.new_session()
        self.other = self.store.new_session()
        self.collection = self.store.create(self.token, "Research")

    def test_session_isolation(self):
        self.assertEqual(self.store.list(self.other), [])
        for operation in (
            lambda: self.store.collection(self.other, self.collection),
            lambda: self.store.delete(self.other, self.collection),
            lambda: self.store.collection("missing", self.collection),
        ):
            with self.assertRaises(HTTPException):
                operation()
        self.assertEqual(self.store.list(self.token)[0]["name"], "Research")

    def test_expiry_and_capacity(self):
        self.store.sessions[self.token]["touched"] -= 3601
        with self.assertRaises(HTTPException):
            self.store.list(self.token)
        for _ in range(15):
            self.store.new_session()
        with self.assertRaises(ValueError):
            self.store.new_session()

    def test_collection_limit_and_disposal(self):
        for _ in range(3):
            self.store.create(self.token, "More")
        with self.assertRaises(ValueError):
            self.store.create(self.token, "Too many")
        self.store.dispose(self.token)
        with self.assertRaises(HTTPException):
            self.store.list(self.token)
        self.assertEqual(self.store.list(self.other), [])

    def test_remove_document_and_hide_internal_chunk_ids(self):
        collection = self.store.collection(self.token, self.collection)
        deleted = []
        collection.index = SimpleNamespace(delete=lambda ids: deleted.extend(ids))
        collection.documents["doc"] = dict(id="doc", name="notes.txt", chunks=2, chunk_ids=["a", "b"])
        self.assertNotIn("chunk_ids", self.store.list(self.token)[0]["documents"][0])
        self.store.delete(self.token, self.collection, "doc")
        self.assertEqual(deleted, ["a", "b"])
        self.assertEqual(collection.documents, {})


class TestKnowledgeRoutes(unittest.TestCase):
    def setUp(self):
        self.model = test_server_ui.Model()
        app = FastAPI()
        register_ui(app, SimpleNamespace(models={"test": self.model}))
        self.store = app.state.knowledge
        self.client = TestClient(app)
        self.token = self.client.post("/ui/knowledge/session").json()["token"]
        self.headers = {"X-Eole-Knowledge-Session": self.token}
        self.collection = self.client.post(
            "/ui/knowledge/collections", headers=self.headers, json={"name": "Notes"}
        ).json()["id"]
        self.sources = [
            dict(id=1, name="notes.txt", page=2, chunk=1, text="The telescope closes at 23:00.", distance=0.1)
        ]
        self.store.search = lambda *args: self.sources.copy()

    def events(self, response):
        self.assertEqual(response.status_code, 200)
        return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]

    def test_assets_and_optional_status(self):
        self.assertEqual(self.client.get("/ui/assets/knowledge.js").status_code, 200)
        self.assertEqual(self.client.get("/ui/assets/knowledge.css").status_code, 200)
        info = self.client.get("/ui/knowledge/info").json()
        self.assertEqual(info["device"], "cpu")
        self.assertEqual(info["embedding_model"], "google/embeddinggemma-2")
        self.assertTrue(info["temporary"])

    def test_ownership_enforced_before_stream(self):
        response = self.client.post(
            f"/ui/knowledge/collections/{self.collection}/ask", json={"model": "test", "question": "When?"}
        )
        self.assertEqual(response.status_code, 401)
        self.assertEqual(self.client.get("/ui/knowledge/collections", headers=self.headers).status_code, 200)
        self.assertEqual(self.client.get("/ui/knowledge/collections").status_code, 401)

    def test_sources_precede_answer_and_exact_worker_metrics(self):
        prompts = []
        self.model.apply_chat_template = lambda messages, **kwargs: prompts.append(messages) or messages[-1]["content"]
        response = self.client.post(
            f"/ui/knowledge/collections/{self.collection}/ask",
            headers=self.headers,
            json={"model": "test", "question": "When does it close?", "max_tokens": 16},
        )
        events = self.events(response)
        self.assertEqual([e["type"] for e in events], ["start", "sources", "delta", "delta", "done"])
        self.assertEqual(events[1]["sources"][0]["page"], 2)
        self.assertEqual(events[-1]["stats"]["output_tokens"], 3)
        self.assertGreaterEqual(events[-1]["stats"]["retrieval_seconds"], 0)
        self.assertIn("[1] notes.txt", prompts[0][-1]["content"])
        self.assertIn("untrusted", prompts[0][0]["content"])

    def test_context_overflow_fails_without_unbacked_answer(self):
        self.model.count_tokens = lambda text: 1000
        response = self.client.post(
            f"/ui/knowledge/collections/{self.collection}/ask",
            headers=self.headers,
            json={"model": "test", "question": "When?", "max_tokens": 16},
        )
        self.assertIn("context budget", self.events(response)[-1]["message"])

    def test_empty_collection_and_invalid_settings(self):
        def fail(*args):
            raise ValueError("Import a document before asking a question")

        self.store.search = fail
        response = self.client.post(
            f"/ui/knowledge/collections/{self.collection}/ask",
            headers=self.headers,
            json={"model": "test", "question": "When?", "max_tokens": 16},
        )
        self.assertEqual(self.events(response)[-1]["type"], "error")
        self.assertEqual(
            self.client.post(
                f"/ui/knowledge/collections/{self.collection}/ask",
                headers=self.headers,
                json={"model": "test", "question": "", "max_tokens": 16},
            ).status_code,
            422,
        )

    def test_index_stream_and_failure_are_reported(self):
        self.store.add = lambda *args: dict(id="doc", name="notes.txt", chunks=2)
        response = self.client.post(
            f"/ui/knowledge/collections/{self.collection}/import",
            headers=self.headers,
            json={"name": "notes.txt", "data": "aGVsbG8="},
        )
        result = self.events(response)
        self.assertEqual(result[-1]["document"]["chunks"], 2)
        self.assertGreaterEqual(result[-1]["indexing_seconds"], 0)

    def test_prompt_keeps_history_and_delimits_sources(self):
        self.model.apply_chat_template = lambda messages, **kwargs: messages
        request = Question(
            model="test", question="And Saturday?", history=[dict(role="user", content="Opening hours?")]
        )
        prompt = build_prompt(self.model, request, self.sources)
        self.assertEqual(prompt[1]["content"], "Opening hours?")
        self.assertIn("page 2", prompt[-1]["content"])
        self.assertIn("Do not invent citations", prompt[0]["content"])


class TestCPUDrain(unittest.IsolatedAsyncioTestCase):
    async def test_disconnect_sets_cancel_and_waits_for_worker(self):
        entered, stopped = threading.Event(), threading.Event()
        cancel = threading.Event()

        def run():
            entered.set()
            cancel.wait(timeout=2)
            stopped.set()

        task = asyncio.create_task(cpu_job(run, cancel=cancel))
        await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(cancel.is_set())
        self.assertTrue(stopped.is_set())


@unittest.skipUnless(find_spec("langchain_core") is not None, "Optional RAG dependencies are not installed")
class TestKnowledgeDocuments(unittest.TestCase):
    def upload(self, name, data):
        return Upload(name=name, data=base64.b64encode(data).decode())

    def test_text_validation(self):
        name, documents = load_documents(self.upload("../notes.md", b"# Hello\nSome content"))
        self.assertEqual(name, "notes.md")
        self.assertEqual(documents[0].metadata["source"], "notes.md")
        for name, data in (("empty.txt", b" "), ("file.exe", b"x"), ("large.txt", b"x" * 80001)):
            with self.assertRaises(ValueError):
                load_documents(self.upload(name, data))

    def test_pdf_page_provenance_and_scanned_rejection(self):
        try:
            import pymupdf
        except ImportError:
            self.skipTest("Optional PDF renderer absent")
        with pymupdf.open() as doc:
            page = doc.new_page()
            page.insert_text((50, 50), "First page source.")
            page = doc.new_page()
            page.insert_text((50, 50), "Second page source.")
            _, documents = load_documents(self.upload("test.pdf", doc.tobytes()))
        self.assertEqual([document.metadata["page"] for document in documents], [1, 2])
        with pymupdf.open() as doc:
            doc.new_page()
            with self.assertRaisesRegex(ValueError, "OCR"):
                load_documents(self.upload("scan.pdf", doc.tobytes()))

    def test_invalid_encoding(self):
        with self.assertRaises(ValueError):
            load_documents(Upload(name="bad.txt", data="!"))
