"""Translation workflows: native model routing, safe structure and real job lifecycle."""

import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from eole.predict.generator import GeneratorLM
from eole.server.ui import register_ui
from eole.server.ui_translation import Document, TranslationRequest, translation_prompt


class DocumentTests(unittest.TestCase):
    def test_blank_lines_leading_space_and_order(self):
        doc = Document("\n  One.\n\n\nTwo.\r\n\r\nThree.\n", "text")
        self.assertEqual([s.text for s in doc.segments], ["One.", "Two.", "Three."])
        doc.outputs = {0: "Un.", 1: "Deux.", 2: "Trois."}
        self.assertEqual(doc.render(), "\n  Un.\n\n\nDeux.\r\n\r\nTrois.\n")

    def test_json_only_string_values_and_safe_escaping(self):
        text = '{ "title": "Hello", "nested": ["World", 42, true], "empty": "" }'
        doc = Document(text, "json")
        self.assertEqual([s.text for s in doc.segments], ["Hello", "World"])
        doc.outputs = {0: 'Il dit "bonjour"', 1: "Monde"}
        parsed = json.loads(doc.render())
        self.assertEqual(parsed, dict(title='Il dit "bonjour"', nested=["Monde", 42, True], empty=""))
        self.assertTrue(doc.render().startswith('{ "title": '))

    def test_subtitle_metadata_is_never_generated(self):
        text = (
            "1\r\n00:00:01,000 --> 00:00:03,000\r\nHello\r\nworld\r\n\r\n2\r\n00:00:04,000 --> 00:00:05,000\r\nBye\r\n"
        )
        doc = Document(text, "srt")
        self.assertEqual([s.text for s in doc.segments], ["Hello\r\nworld", "Bye"])
        doc.outputs = {0: "Bonjour\nmonde", 1: "Au revoir"}
        self.assertIn("00:00:01,000 --> 00:00:03,000", doc.render())
        self.assertTrue(doc.render().endswith("Au revoir\r\n"))
        vtt = Document("WEBVTT\n\nNOTE skip this\n\ncue-1\n00:01.000 --> 00:03.000 align:start\nHello\n", "vtt")
        self.assertEqual([s.text for s in vtt.segments], ["Hello"])

    def test_invalid_documents_and_long_input(self):
        for text, format in [(" ", "text"), ('{"x": NaN}', "json"), ("bad", "srt"), ("bad", "vtt")]:
            with self.assertRaises(ValueError):
                Document(text, format)
        doc = Document("word " * 2000, "text")
        self.assertGreater(len(doc.segments), 1)
        self.assertEqual(doc.render(), "word " * 2000)
        chinese = "欢迎使用本地翻译。" * 500
        doc = Document(chinese, "text")
        self.assertGreater(len(doc.segments), 1)
        self.assertEqual(doc.render(), chinese)

    def test_official_style_terminology_and_context_prompts(self):
        request = TranslationRequest(
            model="hy",
            text="workspace",
            source="English",
            target="French",
            tone="formal",
            context="Software documentation",
            glossary=[dict(source="workspace", target="espace")],
        )
        prompt = translation_prompt(request, request.text)
        self.assertIn('"workspace" translates to "espace"', prompt)
        self.assertIn("formal tone", prompt)
        self.assertIn("Software documentation", prompt)
        self.assertIn("source language is English", prompt)


class ImportTests(unittest.TestCase):
    def test_docx_text_and_archive_bounds(self):
        import base64
        from io import BytesIO
        import zipfile
        from eole.server.ui_documents import extract_document

        data = BytesIO()
        with zipfile.ZipFile(data, "w") as z:
            z.writestr(
                "word/document.xml",
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                "<w:body><w:p><w:r><w:t>Hello</w:t></w:r></w:p>"
                "<w:tbl><w:tr><w:tc><w:p><w:r><w:t>World</w:t></w:r></w:p></w:tc></w:tr></w:tbl>"
                "</w:body></w:document>",
            )
        self.assertEqual(extract_document("sample.docx", base64.b64encode(data.getvalue()).decode()), "Hello\n\nWorld")
        with self.assertRaises(ValueError):
            extract_document("sample.docx", "invalid_base64")
        with self.assertRaises(ValueError):
            extract_document("sample.exe", base64.b64encode(b"text").decode())


class TranslationRoutes(unittest.TestCase):
    def setUp(self):
        self.prompts = []
        owner = self

        class Engine:
            config = SimpleNamespace(
                world_size=1, compute_dtype="bf16", context_length=4096, max_length=256, self_speculative_decoding=False
            )
            predictor = Mock(spec=GeneratorLM)

            def infer_list_stream(self, prompt, settings, generation_stats, cancel_event):
                owner.prompts.append(prompt)
                yield owner.output
                if owner.cancel:
                    cancel_event.set()
                generation_stats.update(
                    output_tokens=4,
                    stopped_on_eos=not owner.truncated,
                    decode_seconds=0.2,
                    decode_tokens=3,
                    cancelled=cancel_event.is_set(),
                )

        self.output, self.truncated = "Bonjour", False
        self.cancel = False
        self.model = SimpleNamespace(
            loaded=True,
            pre_config={},
            engine=Engine(),
            translation_config={},
            count_tokens=lambda text: 20,
            get_model_limits=lambda: (256, 3840),
            apply_chat_template=lambda messages, **kwargs: messages[0]["content"],
        )
        app = FastAPI()
        register_ui(app, SimpleNamespace(models={"hy": self.model}))
        self.client = TestClient(app)
        self.payload = dict(model="hy", text="Hello\n\nWorld", source="English", target="French", max_tokens=64)

    def events(self, **overrides):
        response = self.client.post("/ui/translate", json={**self.payload, **overrides})
        self.assertEqual(response.status_code, 200, response.text)
        return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]

    def test_multi_segment_completion_and_shared_monitor(self):
        events = self.events()
        done = events[-1]
        self.assertTrue(done["complete"])
        self.assertEqual(done["text"], "Bonjour\n\nBonjour")
        self.assertEqual(len(self.prompts), 2)
        self.assertEqual(done["stats"]["output_tokens"], 8)
        self.assertEqual(done["stats"]["decode_tokens_per_second"], 15)
        state = self.client.get("/ui/state").json()
        self.assertEqual(state["active"], [])
        self.assertEqual(state["recent"][0]["task"], "translate")
        self.assertEqual(state["recent"][0]["status"], "complete")

    def test_limits_language_and_placeholder_errors(self):
        self.assertEqual(self.events(target="invented")[-1]["type"], "error")
        self.assertEqual(self.events(text="Hello ${name}")[-1]["type"], "error")
        self.assertEqual(self.events(max_tokens=257)[-1]["type"], "error")
        self.assertEqual(
            self.client.post("/ui/translate", json={**self.payload, "text": "invalid", "format": "json"}).status_code,
            422,
        )
        self.assertEqual(self.client.post("/ui/translate", json={**self.payload, "target": "English"}).status_code, 422)

    def test_output_limit_never_marks_partial_document_complete(self):
        self.output = "a" * 10
        self.truncated = True
        events = self.events(max_tokens=4)
        self.assertFalse(events[-1]["complete"])
        self.assertEqual(events[-1]["stats"]["completed_segments"], 0)
        self.assertTrue(any(e["type"] == "warning" for e in events))

    def test_cancelled_translation_keeps_unfinished_source_and_stops_job(self):
        self.cancel = True
        done = self.events()[-1]
        self.assertFalse(done["complete"])
        self.assertTrue(done["stats"]["cancelled"])
        self.assertEqual(done["text"], self.payload["text"])
        self.assertEqual(len(self.prompts), 1)
        self.assertEqual(self.client.get("/ui/state").json()["recent"][0]["status"], "cancelled")

    def test_encoder_decoder_uses_native_input_and_restores_settings(self):
        from concurrent.futures import ThreadPoolExecutor

        predictor = SimpleNamespace(beam_size=4)
        self.model.engine.predictor = predictor
        self.model.translation_config = dict(prompt_style="raw", source_language="English", target_language="French")
        self.model.apply_chat_template = lambda *a, **k: self.fail("Native MT must not use a chat template")

        def infer_list(inputs, settings):
            self.assertEqual(predictor.beam_size, 4)
            predictor.beam_size = settings["beam_size"]
            self.prompts.extend(inputs)
            return [], [], [["Bonjour"]]

        with ThreadPoolExecutor(max_workers=1) as pool:
            self.model.engine._thread_pool = pool
            self.model.engine.infer_list = infer_list
            done = self.events(text="Hello")[-1]
        self.assertTrue(done["complete"])
        self.assertEqual(self.prompts, ["Hello"])
        self.assertEqual(predictor.beam_size, 4)
        self.assertIsNone(done["stats"]["decode_tokens_per_second"])
        self.assertIsNone(done["stats"]["output_tokens"])
        self.assertEqual(self.events(tone="formal")[-1]["type"], "error")


if __name__ == "__main__":
    unittest.main()
