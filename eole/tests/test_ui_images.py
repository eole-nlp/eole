"""Chat image validation, template ordering and multimodal budget accounting."""

import base64
from io import BytesIO
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from eole.server.ui import UIMessage
from eole.server.ui_images import chat_prompt, image_capabilities
from eole.tests.test_server_ui import TestServerUI


def attachment():
    stream = BytesIO()
    Image.new("RGB", (64, 32), "red").save(stream, "PNG")
    return dict(name="red.png", data=base64.b64encode(stream.getvalue()).decode())


class TestChatImages(TestServerUI):
    def vision(self):
        self.model.engine.config.data_type = "image"
        self.model.engine.config.model = SimpleNamespace(
            adapter="qwen3_5vl",
            spatial_merge_size=2,
            encoder=SimpleNamespace(patch_size=16, image_size=768),
        )
        self.model.apply_chat_template = lambda messages, **kw: "\n".join(
            (
                message["content"]
                if isinstance(message["content"], str)
                else "".join(part["text"] for part in message["content"])
            )
            for message in messages
        )

    def tearDown(self):
        for key in ("data_type", "model"):
            if hasattr(self.model.engine.config, key):
                delattr(self.model.engine.config, key)

    def test_image_prompt_preserves_order_braces_and_prior_images(self):
        self.vision()
        messages = [
            UIMessage(role="user", content='Read {"a": 1}', images=[attachment()]),
            UIMessage(role="assistant", content="Red"),
            UIMessage(role="user", content="And now?"),
        ]
        prompt, budget = chat_prompt(self.model, messages, False)
        self.assertEqual(list(prompt["images"]), ["eole_image_0"])
        self.assertIn('{{"a": 1}}', prompt["text"])
        self.assertIn("<|vision_start|>", budget)
        self.assertNotIn("EOLEIMAGEPLACEHOLDER", budget)
        self.assertTrue(budget.endswith("And now?"))
        self.assertEqual(prompt["text"].format(eole_image_0="VISUAL"), 'VISUALRead {"a": 1}\nRed\nAnd now?')

    def test_request_rejects_nonuser_and_too_many_images(self):
        for messages in (
            [{"role": "assistant", "content": "", "images": [attachment()]}],
            [{"role": "user", "content": "", "images": [attachment()] * 3}] * 2,
        ):
            self.assertEqual(self.client.post("/ui/chat", json={**self.payload, "messages": messages}).status_code, 422)

    def test_unloaded_local_vision_capability(self):
        import json
        import os
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "config.json").write_text(json.dumps({"model": {"adapter": "qwen3_5vl"}}))
            with patch.dict(os.environ, {"EOLE_TEST_VISION_PATH": folder}):
                model = SimpleNamespace(
                    engine=None,
                    local_path=None,
                    model_path="${EOLE_TEST_VISION_PATH}",
                    pre_config={"data_type": "image"},
                )
                self.assertTrue(image_capabilities(model))
                model.pre_config = {}
                self.assertFalse(image_capabilities(model))

    def test_nonvision_model_rejects_images(self):
        result = self.events(
            {**self.payload, "messages": [{"role": "user", "content": "Describe", "images": [attachment()]}]}
        )
        self.assertEqual(result[-1]["type"], "error")
        self.assertIn("image-capable", result[-1]["message"])

    def test_invalid_image_and_total_limit(self):
        self.vision()
        with self.assertRaises(ValueError):
            chat_prompt(self.model, [UIMessage(role="user", content="", images=[dict(name="x", data="!")])], False)
        with self.assertRaises(ValueError):
            chat_prompt(self.model, [UIMessage(role="user", content="", images=[attachment()] * 3)] * 2, False)

    def test_budget_uses_expanded_tokens_and_stream_receives_native_input(self):
        self.vision()
        self.model.count_tokens = lambda text: 120 if "<|image_pad|>" in text else 1
        payload = {**self.payload, "messages": [{"role": "user", "content": "Describe", "images": [attachment()]}]}
        self.assertEqual(self.events(payload)[-1]["type"], "error")
        self.model.count_tokens = lambda text: 10
        seen = []
        original = self.model.engine.infer_list_stream

        def stream(prompt, **kwargs):
            seen.append(prompt)
            yield from original(prompt, **kwargs)

        self.model.engine.infer_list_stream = stream
        self.assertEqual(self.events(payload)[-1]["type"], "done")
        self.assertEqual(list(seen[0]["images"]), ["eole_image_0"])

    def test_state_capability_and_text_only_in_image_mode(self):
        self.vision()
        self.assertTrue(image_capabilities(self.model))
        with patch("eole.server.ui.Hardware.read", return_value={}):
            self.assertTrue(self.client.get("/ui/state").json()["models"][0]["images_supported"])
        native, budget = chat_prompt(self.model, [UIMessage(role="user", content="a {b}")], False)
        self.assertEqual(native, dict(text="a {{b}}", images={}))
        self.assertEqual(budget, "a {b}")


class TestVisionTextInputs(unittest.TestCase):
    def test_hf_image_token_count_uses_native_tokenizer(self):
        from eole.server.model import Model
        from eole.transforms.tokenize_id import HuggingfaceTokenizer

        tokenizer = object.__new__(HuggingfaceTokenizer)
        tokenizer.tokenize_string = lambda text, side: ([1] * text.count("<|image_pad|>"), [])
        model = SimpleNamespace(
            loaded=True, engine=SimpleNamespace(transform_pipe=SimpleNamespace(transforms=[tokenizer]))
        )
        self.assertEqual(Model.count_tokens(model, "<|image_pad|>" * 240), 240)

    def test_browser_image_helpers(self):
        import shutil
        import subprocess
        from pathlib import Path

        if not shutil.which("node"):
            self.skipTest("Node is optional; browser helpers tested when available")
        subprocess.run(["node", "eole/tests/web/test_chat_images.cjs"], cwd=Path(__file__).parents[2], check=True)

    def test_plain_strings_keep_literal_braces_in_image_corpus(self):
        from eole.inputters.text_corpus import ImageTextCorpus

        item = next(iter(ImageTextCorpus("infer", ['JSON {"a": 1}']).load()))
        self.assertEqual(item["text"].format(), 'JSON {"a": 1}')
        self.assertEqual(item["images"], {})

    def test_empty_images_keep_text_only_fast_paths(self):
        from eole.inputters.text_utils import tensorify

        vocabs = {"specials": {"pad_token": "pad"}, "src": {"pad": 0}}
        example = dict(src={"src_ids": [1, 2]}, images={}, cid="infer", cid_line_number=0)
        batch = tensorify(vocabs, [(example, 0)], "cpu")
        self.assertIsNone(batch["images"])

    def test_chunked_prefill_slices_vision_positions(self):
        import torch
        from eole.decoders.transformer import TransformerDecoder

        positions = torch.arange(30).reshape(1, 10, 3)
        observed = []

        def forward(emb, **kwargs):
            observed.append(kwargs["pos_ids_2d"])
            return emb, {}

        decoder = SimpleNamespace(sliding_window=0, prefill_chunk_size=4, _prefill_cache=None, _forward_eager=forward)
        emb = torch.zeros(1, 10, 8)
        result, _ = TransformerDecoder._forward_chunked_prefill(
            decoder, emb, tgt_pad_mask=torch.zeros(1, 1, 10, dtype=torch.bool), pos_ids_2d=positions
        )
        self.assertEqual(result.shape, emb.shape)
        self.assertEqual([part.shape[1] for part in observed], [4, 4, 2])
        self.assertTrue(torch.equal(torch.cat(observed, dim=1), positions))

    def test_xdrope_chunked_output_matches_unchunked(self):
        import torch
        from eole.decoders.transformer import TransformerDecoder
        from eole.modules.rope import apply_rotary_pos_emb_xdrope

        class Layer:
            self_attn = SimpleNamespace(xdrope_section=[1, 1, 1])
            layer_type = "full_attention"

            def __call__(self, emb, **kwargs):
                query = emb.unsqueeze(2)
                rotated, _ = apply_rotary_pos_emb_xdrope(
                    query, query, kwargs["position_embeddings"], kwargs["pos_ids_2d"], [1, 1, 1]
                )
                return rotated.squeeze(2), None

        table = torch.randn(64, 6)
        decoder = SimpleNamespace(
            sliding_window=0,
            prefill_chunk_size=4,
            _prefill_cache=None,
            with_cross_attn=False,
            cache_seqlens=torch.zeros(1, dtype=torch.int32),
            rope=SimpleNamespace(cos_sin=table),
            rope_local=None,
            has_linear_attn=False,
            has_sliding_attn=False,
            flash=True,
            transformer_layers=[Layer()],
            layer_norm=lambda emb: emb,
        )
        decoder._forward_eager = lambda emb, **kw: TransformerDecoder._forward_eager(decoder, emb, **kw)
        emb = torch.randn(1, 10, 6)
        positions = torch.arange(10).view(1, 10, 1).expand(1, 10, 3)
        kwargs = dict(tgt_pad_mask=torch.zeros(1, 1, 10, dtype=torch.bool), pos_ids_2d=positions)
        whole, _ = decoder._forward_eager(emb, **kwargs)
        decoder.cache_seqlens.zero_()
        chunked, _ = TransformerDecoder._forward_chunked_prefill(decoder, emb, **kwargs)
        torch.testing.assert_close(chunked, whole)
