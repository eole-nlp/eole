"""Portable Hub checkpoints must be complete and verified before activation."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from recipes.runpod.checkpoints import DFLASH2, download, main, manifest_files, verify, verify_files


class TestCheckpointDownload(unittest.TestCase):
    def test_drafter_download_uses_pinned_files_without_target_manifest(self):
        with patch("sys.argv", ["checkpoints", "download-drafter", DFLASH2["repo"], DFLASH2["revision"], "/tmp/draft"]):
            with patch("recipes.runpod.checkpoints.snapshot_download") as snapshot:
                main()
        self.assertEqual(snapshot.call_args.kwargs["revision"], DFLASH2["revision"])
        self.assertEqual(set(snapshot.call_args.kwargs["allow_patterns"]), {"config.json", "model.safetensors"})

    def test_drafter_checksums_reject_corrupt_files(self):
        files = {"config.json": self.manifest["files"]["config.json"]}
        verify_files(self.root, files)
        (self.root / "config.json").write_bytes(b"[]")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            verify_files(self.root, files)
        self.assertFalse((self.root / "config.json").exists())

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        contents = {
            "config.json": b"{}",
            "tokenizer.json": b"{}",
            "vocab.json": b"{}",
            "model.00.safetensors": b"weights",
            "model.safetensors.index.json": json.dumps({"weight_map": {"weight": "model.00.safetensors"}}).encode(),
        }
        self.manifest = {"format": "eole", "format_version": 1, "files": {}}
        for name, data in contents.items():
            (self.root / name).write_bytes(data)
            self.manifest["files"][name] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        self.write_manifest()

    def write_manifest(self):
        (self.root / "model-manifest.json").write_text(json.dumps(self.manifest))

    def test_pinned_download_uses_direct_directory_and_only_manifest_files(self):
        with patch("recipes.runpod.checkpoints.hf_hub_download", return_value=self.root / "model-manifest.json") as hf:
            with patch("recipes.runpod.checkpoints.snapshot_download") as snapshot:
                download("org/model", "a" * 40, self.root)
        self.assertEqual(hf.call_args.kwargs["revision"], "a" * 40)
        self.assertEqual(snapshot.call_args.kwargs["revision"], "a" * 40)
        self.assertEqual(snapshot.call_args.kwargs["local_dir"], self.root)
        self.assertEqual(
            set(snapshot.call_args.kwargs["allow_patterns"]), {*self.manifest["files"], "model-manifest.json"}
        )
        verify(self.root)

    def test_corruption_is_rejected_and_bad_file_removed_for_retry(self):
        (self.root / "model.00.safetensors").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            verify(self.root)
        self.assertFalse((self.root / "model.00.safetensors").exists())

    def test_missing_truncated_or_symlinked_files_are_rejected(self):
        weight = self.root / "model.00.safetensors"
        weight.write_bytes(b"x")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            verify(self.root)
        weight.unlink(missing_ok=True)
        weight.symlink_to(self.root / "config.json")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            verify(self.root)

    def test_path_traversal_and_incomplete_manifest_are_rejected(self):
        for name in ["../outside", "/tmp/outside", "deployment.json"]:
            with self.subTest(name=name):
                bad = json.loads(json.dumps(self.manifest))
                bad["files"][name] = {"size": 0, "sha256": "a" * 64}
                with self.assertRaises(ValueError):
                    manifest_files(bad)
        del self.manifest["files"]["tokenizer.json"]
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            manifest_files(self.manifest)

    def test_index_cannot_reference_unpublished_shards(self):
        data = json.dumps({"weight_map": {"w": "missing.safetensors"}}).encode()
        (self.root / "model.safetensors.index.json").write_bytes(data)
        self.manifest["files"]["model.safetensors.index.json"] = {
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "missing weights"):
            verify(self.root)
