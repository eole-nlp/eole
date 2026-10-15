"""OCR upload bounds, grounding output and real request lifecycle without a GPU."""

import base64
from io import BytesIO
import json
from types import SimpleNamespace
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from eole.server.ui_ocr import OCRImport, import_pages, parse_ocr, register_ocr
from eole.server.ui_runtime import Workspace


def image_data():
    buffer = BytesIO()
    Image.new("RGB", (200, 100), "white").save(buffer, "PNG")
    return base64.b64encode(buffer.getvalue()).decode()


class OCRTests(unittest.TestCase):
    def test_image_and_invalid_upload(self):
        pages = import_pages(OCRImport(name="scan.png", data=image_data()))
        self.assertEqual((pages[0]["width"], pages[0]["height"]), (200, 100))
        with self.assertRaises(ValueError):
            import_pages(OCRImport(name="bad.pdf", data="not base64"))
        with self.assertRaises(Exception):
            import_pages(OCRImport(name="bad.png", data=base64.b64encode(b"bad").decode()))

    def test_grounding_boxes_and_markdown(self):
        result = parse_ocr("<|ref|>title<|/ref|><|det|>[[10, 20, 900, 100]]<|/det|>\n# Hello\n<script>bad</script>")
        self.assertEqual(result["boxes"][0], dict(label="title", x=1, y=2, width=89, height=8))
        self.assertIn("# Hello", result["markdown"])
        self.assertNotIn("<|det|>", result["markdown"])
        # Content is escaped by the browser renderer, not interpreted as HTML here.
        self.assertIn("<script>", result["markdown"])
        self.assertEqual(parse_ocr("<|ref|>x<|/ref|><|det|>[[900,0,5,1001]]<|/det|>")["boxes"], [])

    def test_partial_grounding_kept_only_in_raw_output(self):
        result = parse_ocr("Readable text\n<|ref|>image<|/ref|><|det|>[[10,")
        self.assertEqual(result["markdown"], "Readable text")
        self.assertEqual(result["boxes"], [])

    def test_pdf_pages_and_limits(self):
        try:
            import pymupdf
        except ImportError:
            self.skipTest("Optional PDF renderer not installed")
        with pymupdf.open() as doc:
            doc.new_page(width=300, height=600)
            doc.new_page()
            data = base64.b64encode(doc.tobytes()).decode()
        pages = import_pages(OCRImport(name="two.pdf", data=data))
        self.assertEqual(len(pages), 2)
        self.assertEqual((pages[0]["width"], pages[0]["height"]), (512, 1024))
        with pymupdf.open() as doc:
            for _ in range(51):
                doc.new_page()
            data = base64.b64encode(doc.tobytes()).decode()
        with self.assertRaisesRegex(ValueError, "50 pages"):
            import_pages(OCRImport(name="long.pdf", data=data))

    def client(self, fail=False, cancelled=False, incomplete=False):
        class Engine:
            config = SimpleNamespace(world_size=1, data_type="image")

            def infer_list_stream(
                self,
                prompt,
                settings,
                generation_stats,
                cancel_event,
                preserve_special_tokens=False,
            ):
                assert preserve_special_tokens
                self.prompt = prompt
                if fail:
                    raise RuntimeError("test failure")
                yield "# Hello｟newline｠World"
                if cancelled:
                    cancel_event.set()
                generation_stats.update(
                    output_tokens=settings["max_length"] if incomplete else 5,
                    stopped_on_eos=not incomplete,
                    decode_tokens_per_second=42,
                )

        engine = Engine()
        model = SimpleNamespace(
            model_id="deepseek-ocr",
            model_path="DeepSeek-OCR",
            loaded=True,
            engine=engine,
            get_model_limits=lambda: (8192, 8192),
        )
        app, workspace = FastAPI(), Workspace()
        register_ocr(
            app,
            SimpleNamespace(
                models={
                    "ocr": model,
                    "chat": SimpleNamespace(model_path="qwen", model_id="chat"),
                }
            ),
            workspace,
        )
        return TestClient(app), workspace, engine

    def test_page_inference_and_monitor(self):
        client, workspace, engine = self.client()
        response = client.post("/ui/ocr/page", json=dict(model="ocr", data=image_data()))
        frames = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        self.assertEqual([f["type"] for f in frames], ["start", "delta", "done"])
        self.assertEqual(frames[-1]["markdown"], "# Hello\nWorld")
        self.assertEqual(frames[-1]["stats"]["decode_tokens_per_second"], 42)
        self.assertIn("<|grounding|>", engine.prompt["text"])
        self.assertIsInstance(engine.prompt["images"]["image"], Image.Image)
        self.assertFalse(workspace.active)
        self.assertEqual(workspace.recent[0]["status"], "complete")
        self.assertEqual(workspace.recent[0]["task"], "ocr")
        self.assertEqual(
            client.post("/ui/ocr/page", json=dict(model="chat", data=image_data())).status_code,
            422,
        )
        self.assertEqual(
            client.post("/ui/ocr/page", json=dict(model="missing", data=image_data())).status_code,
            404,
        )

    def test_cancelled_and_truncated_pages_keep_partial_output(self):
        for cancelled, incomplete, expected in [
            (True, False, "cancelled"),
            (False, True, "incomplete"),
        ]:
            with self.subTest(expected=expected):
                client, workspace, _ = self.client(cancelled=cancelled, incomplete=incomplete)
                response = client.post("/ui/ocr/page", json=dict(model="ocr", data=image_data()))
                frames = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
                self.assertEqual(frames[-1]["cancelled"], cancelled)
                self.assertEqual(frames[-1]["incomplete"], incomplete)
                self.assertEqual(frames[-1]["markdown"], "# Hello\nWorld")
                self.assertFalse(workspace.active)
                self.assertEqual(workspace.recent[0]["status"], expected)

    def test_failure_releases_request(self):
        client, workspace, _ = self.client(fail=True)
        response = client.post("/ui/ocr/page", json=dict(model="ocr", data=image_data()))
        self.assertIn('"type": "error"', response.text)
        self.assertFalse(workspace.active)
        self.assertEqual(workspace.recent[0]["status"], "error")


class OCRStreamerTests(unittest.TestCase):
    def test_grounding_markers_preserved_without_changing_chat(self):
        from tokenizers import AddedToken, Tokenizer
        from tokenizers.models import WordLevel
        from eole.predict.streamer import GenerationStreamer
        from eole.transforms.tokenize_id import HuggingfaceTokenizer

        tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "<|ref|>": 1, "title": 2, "<|eos|>": 3}, unk_token="[UNK]"))
        tokenizer.add_special_tokens([AddedToken("<|ref|>", special=True), AddedToken("<|eos|>", special=True)])
        transform = object.__new__(HuggingfaceTokenizer)
        transform.tokenizer = tokenizer
        pipe = SimpleNamespace(transforms=[transform], apply_reverse=transform.apply_reverse)
        for preserve, expected in [(True, "<|ref|> title"), (False, "title")]:
            streamer = GenerationStreamer({}, pipe, eos_token_ids=(3,), preserve_special_tokens=preserve)
            for token in [1, 2, 3]:
                streamer.put([token])
            streamer.end()
            self.assertEqual("".join(streamer), expected)
            self.assertEqual(streamer.token_count, 3)

    def test_missing_tokenizer_is_explicit(self):
        from eole.predict.streamer import GenerationStreamer

        with self.assertRaisesRegex(ValueError, "HuggingFace tokenizer"):
            GenerationStreamer({}, preserve_special_tokens=True)


class OCRAdapterTests(unittest.TestCase):
    def test_model_prompts_and_capabilities(self):
        from eole.server.ui_ocr_models import ocr_adapter

        deepseek = ocr_adapter(SimpleNamespace(model_path="/models/DeepSeek-OCR"))
        hunyuan = ocr_adapter(SimpleNamespace(model_path="/models/HunyuanOCR"))
        self.assertTrue(deepseek.layout)
        self.assertFalse(hunyuan.layout)
        self.assertIn("<|grounding|>", deepseek.prompt)
        self.assertIn("{image}", hunyuan.prompt)
        self.assertIn("<｜hy_User｜>", hunyuan.prompt)
        self.assertNotIn("<|grounding|>", hunyuan.prompt)
        self.assertIsNone(ocr_adapter(SimpleNamespace(model_path="Hy-MT2-7B")))

    def test_hunyuan_control_tokens_and_tables(self):
        from eole.server.ui_ocr_models import HUNYUAN

        raw = (
            "<｜hy_place▁holder▁no▁112｜>A title<｜hy_place▁holder▁no▁113｜>"
            "<｜hy_place▁holder▁no▁110｜>0,0,100,100<｜hy_place▁holder▁no▁111｜>\n"
            '<table onclick="bad()"><tr><th colspan="2">Header</th></tr>'
            '<tr><td rowspan="-2">&lt;img src=x&gt;<script>bad()</script></td>'
            '<td>\\(x^2\\)<img src="https://example.invalid/x" onerror="bad()"></td></tr></table>'
            "<｜hy_Assistant｜>"
        )
        result = parse_ocr(raw, HUNYUAN)
        self.assertEqual(result["boxes"], [])
        self.assertIn("A title", result["markdown"])
        self.assertNotIn("hy_", result["markdown"])
        self.assertNotIn("0,0,100,100", result["markdown"])
        rows = result["tables"][0]["rows"]
        self.assertEqual(rows[0][0]["colspan"], 2)
        self.assertEqual(rows[1][0]["rowspan"], 1)
        self.assertEqual(rows[1][0]["text"], "<img src=x>")
        self.assertEqual(rows[1][1]["text"], "\\(x^2\\)")
        self.assertNotIn("onclick", str(rows))
        self.assertNotIn("https:", str(rows))

    def test_malformed_table_attributes_and_partial_control_token(self):
        from eole.server.ui_ocr_models import HUNYUAN

        result = parse_ocr('<table><tr><td colspan rowspan="999999">Cell</td></tr></table><｜hy_place', HUNYUAN)
        self.assertEqual(result["tables"][0]["rows"][0][0]["colspan"], 1)
        self.assertEqual(result["tables"][0]["rows"][0][0]["rowspan"], 1)
        self.assertNotIn("hy_place", result["markdown"])

    def test_hunyuan_page_uses_own_prompt_and_disables_overlay(self):
        # Keep the request path exercised through the actual async bridge.
        client, workspace, engine = OCRTests().client()
        from eole.server.ui_ocr_models import HUNYUAN
        from unittest.mock import patch

        with patch("eole.server.ui_ocr.ocr_adapter", return_value=HUNYUAN):
            response = client.post("/ui/ocr/page", json=dict(model="ocr", data=image_data()))
        frames = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        self.assertEqual(frames[-1]["type"], "done")
        self.assertFalse(frames[-1]["layout_supported"])
        self.assertEqual(engine.prompt["text"], HUNYUAN.prompt)
        self.assertEqual(workspace.recent[0]["status"], "complete")

    def test_preview_renderer(self):
        import shutil
        import subprocess
        from pathlib import Path

        node = shutil.which("node")
        if node is None:
            self.skipTest("Node is only needed for this browser renderer regression check")
        for script in ("test_ocr_render.cjs", "test_ocr_drop.cjs"):
            subprocess.run([node, str(Path(__file__).with_name("web") / script)], check=True, timeout=20)
