"""Download pinned, pre-converted Eole models without a second weight cache."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

DFLASH2 = json.loads(Path(__file__).with_name("dflash2.json").read_text())


def manifest_files(manifest):
    if manifest.get("format") != "eole" or manifest.get("format_version") != 1:
        raise ValueError("Unsupported Eole checkpoint manifest")
    files = manifest["files"]
    for name, metadata in files.items():
        # Published packages are flat. Never let a manifest escape the model directory.
        if not name or Path(name).name != name or name in {".", "..", "model-manifest.json", "deployment.json"}:
            raise ValueError("Invalid checkpoint filename")
        if metadata["size"] < 0 or not re.fullmatch(r"[0-9a-f]{64}", metadata["sha256"]):
            raise ValueError("Invalid checkpoint checksum metadata")
    required = {"config.json", "tokenizer.json", "vocab.json", "model.safetensors.index.json"}
    if not required.issubset(files) or not any(n.startswith("model.") and n.endswith(".safetensors") for n in files):
        raise ValueError("Incomplete Eole checkpoint manifest")
    return files


def download(repo, revision, destination):
    # The only Hub cache entry is the small manifest. Large files go straight to
    # staging; local_dir metadata lets an interrupted transfer resume in place.
    manifest = json.loads(Path(hf_hub_download(repo, "model-manifest.json", revision=revision)).read_text())
    files = manifest_files(manifest)
    snapshot_download(repo, revision=revision, local_dir=destination, allow_patterns=[*files, "model-manifest.json"])


def verify(destination):
    manifest = json.loads((destination / "model-manifest.json").read_text())
    files = manifest_files(manifest)
    verify_files(destination, files)
    index = json.loads((destination / "model.safetensors.index.json").read_text())
    if not index.get("weight_map") or any(name not in files for name in index["weight_map"].values()):
        raise ValueError("Checkpoint index references missing weights")


def verify_files(destination, files):
    for name, metadata in files.items():
        path = destination / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or incomplete checkpoint file: {name}")
        if path.stat().st_size != metadata["size"]:
            path.unlink()
            raise ValueError(f"Missing or incomplete checkpoint file: {name}; restart to download it again")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != metadata["sha256"]:
            # Discard the corrupt file so a retry downloads it again; retain the
            # other completed transfers. No deployment marker has been written.
            path.unlink()
            raise ValueError(f"Checkpoint checksum mismatch: {name}; restart to download it again")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["download", "verify", "download-drafter", "verify-drafter"])
    parser.add_argument("repo")
    parser.add_argument("revision")
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.action.endswith("-drafter"):
        if (args.repo, args.revision) != (DFLASH2["repo"], DFLASH2["revision"]):
            raise ValueError("Drafter must match the pinned deployment checkpoint")
        if args.action == "download-drafter":
            snapshot_download(
                args.repo, revision=args.revision, local_dir=args.destination, allow_patterns=list(DFLASH2["files"])
            )
        else:
            verify_files(args.destination, DFLASH2["files"])
    elif args.action == "download":
        download(args.repo, args.revision, args.destination)
    else:
        verify(args.destination)


if __name__ == "__main__":
    main()
