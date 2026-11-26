"""Print a matched server YAML for one Codex benchmark profile."""

import argparse
import os
from pathlib import Path

import yaml

SERVING = Path(__file__).resolve().parents[1] / "serving"
PROFILES = {
    "serialized": "serve.yaml",
    "greedy": "serve_continuous.yaml",
    "mtp": "serve_continuous_mtp.yaml",
    "dflash2": "serve_continuous.yaml",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=PROFILES)
    parser.add_argument("--prefix-cache-entries", type=int, default=0)
    args = parser.parse_args()
    if args.prefix_cache_entries < 0 or (args.prefix_cache_entries and args.profile == "serialized"):
        parser.error("positive prefix-cache-entries require a continuous profile")
    config = yaml.safe_load((SERVING / PROFILES[args.profile]).read_text())
    model = config["models"][0]
    model["config"].update(context_length=12288, max_length=2048)
    if "continuous" in model:
        # Match prefill chunks throughout.
        model["continuous"]["prefill_idle_chunk_size"] = 256
        model["continuous"]["prefix_cache_entries"] = args.prefix_cache_entries
    if args.profile == "dflash2":
        drafter = os.environ.get("QWEN38_DFLASH2")
        if not drafter or not (Path(drafter).expanduser() / "config.json").is_file():
            parser.error("set QWEN38_DFLASH2 to a local drafter directory")
        model["config"].update(dflash2_model=str(Path(drafter).expanduser().resolve()), dflash2_quantization="int8")
    print(yaml.safe_dump(config, sort_keys=False), end="")


if __name__ == "__main__":
    main()
