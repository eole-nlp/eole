"""Opt-in real CPU embeddings and Chroma integration; never download in normal CI."""

import base64
import os
from pathlib import Path
import threading
import unittest

from eole.server.ui_knowledge import Knowledge, Upload


@unittest.skipUnless(os.environ.get("EOLE_TEST_RAG_MODEL"), "Set EOLE_TEST_RAG_MODEL to run real embedding integration")
class TestRealKnowledgeEmbedding(unittest.TestCase):
    def test_index_retrieve_delete_with_eole_attention_settings(self):
        import torch
        import numpy as np

        store = Knowledge({"embedding_model": os.environ["EOLE_TEST_RAG_MODEL"], "embedding_cpu": True})
        token = store.new_session()
        collection = store.create(token, "Observatory")
        raw = (Path(__file__).parents[2] / "recipes/rag/observatory-guide.md").read_bytes()
        old_math = torch.backends.cuda.math_sdp_enabled()
        try:
            # Eole's inference setup disables math SDPA globally. CPU embedding
            # attention must work without changing that shared backend setting.
            torch.backends.cuda.enable_math_sdp(False)
            record = store.add(
                token, collection, Upload(name="guide.md", data=base64.b64encode(raw).decode()), threading.Event()
            )
            self.assertEqual(str(store.embeddings.model.device), "cpu")
            self.assertEqual(next(store.embeddings.model.parameters()).dtype, torch.float32)
            vector = store.embeddings.embed_query("When does the observatory close on Saturdays?")
            self.assertEqual(len(vector), 768)
            self.assertTrue(np.isfinite(vector).all())
            self.assertFalse(torch.backends.cuda.math_sdp_enabled())
            results = store.search(token, collection, "How do I reserve a telescope?", threading.Event())
            self.assertTrue(any("visits@cedar.example" in result["text"] for result in results))
            if os.environ.get("EOLE_TEST_RAG_CUDA"):
                # Existing Chroma indexes retain the same adapter across device moves.
                adapter = store.embeddings
                gpu_results = store.search(
                    token, collection, "How do I reserve a telescope?", threading.Event(), "cuda:0"
                )
                self.assertEqual(str(adapter.model.device), "cuda:0")
                self.assertIs(store.embeddings, adapter)
                self.assertEqual(gpu_results[0]["text"], results[0]["text"])
                store.search(token, collection, "Hours?", threading.Event(), "cpu")
                self.assertEqual(str(adapter.model.device), "cpu")
            store.delete(token, collection, record["id"])
            with self.assertRaisesRegex(ValueError, "Import a document"):
                store.search(token, collection, "Hours?", threading.Event())
            store.dispose(token)
        finally:
            torch.backends.cuda.enable_math_sdp(old_math)
