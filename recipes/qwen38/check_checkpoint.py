"""Inspect an Eole MTP checkpoint without loading tensor data."""

import argparse
import json
from pathlib import Path


def inspect_checkpoint(path):
    from safetensors import safe_open

    root = Path(path)
    config = json.loads((root / "config.json").read_text())
    decoder = config.get("model", {}).get("decoder", {})
    heads = decoder.get("num_mtp_heads", 0)
    if not heads:
        raise ValueError("Checkpoint configuration has no MTP heads; reconvert a checkpoint retaining MTP weights.")
    keys = []
    # Eole loader consumes model.*.safetensors shards, not raw HF/companion files.
    for shard in sorted(root.glob("model.*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as tensors:
            keys.extend(key for key in tensors.keys() if key.startswith("mtp_heads."))
    for index in range(heads):
        if not any(key.startswith(f"mtp_heads.{index}.") for key in keys):
            raise ValueError(f"No saved tensors for MTP head {index}; do not benchmark an uninitialized head.")
    if not (root / "chat_template.jinja").exists() and not config.get("inference", {}).get("chat_template"):
        raise ValueError("No chat template; provide chat_template.jinja for API serving.")
    return {
        "path": str(root.resolve()),
        "mtp_heads": heads,
        "quant_type": config.get("training", {}).get("quant_type", ""),
        "mtp_tensors": sorted(keys),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path")
    args = parser.parse_args()
    print(json.dumps(inspect_checkpoint(args.model_path), indent=2))


if __name__ == "__main__":
    main()
