#!/usr/bin/env python
"""Write an isolated Codex configuration; never edit the user's normal setup."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:5000/v1")
    parser.add_argument("--model", default="qwen3.8-27B")
    parser.add_argument("--context-window", type=int, default=28672)
    args = parser.parse_args()
    if args.context_window <= 4096:
        parser.error("context-window must leave room for a 4096-token output")
    auto_compact_token_limit = args.context_window - 4096
    target = args.output_dir.expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    target.chmod(0o700)
    catalog_path, config_path = target / "models.json", target / "config.toml"
    if config_path.exists() or catalog_path.exists():
        parser.error("output directory already contains config.toml or models.json; choose a fresh directory")
    model = {
        "slug": args.model,
        "display_name": args.model + " (Eole)",
        "description": "Local Qwen served by Eole; experimental coding integration",
        "default_reasoning_level": "low",
        "supported_reasoning_levels": [{"effort": "low", "description": "Direct output; server disables thinking"}],
        "shell_type": "unified_exec",
        "visibility": "list",
        "supported_in_api": True,
        "priority": 0,
        "upgrade": None,
        "base_instructions": "You are a coding assistant. Use the provided tools to inspect, edit, and test files. "
        "Follow the user instructions and report actual tool results. "
        "Never claim a command succeeded without evidence.",
        "default_reasoning_summary": "none",
        "support_verbosity": False,
        "apply_patch_tool_type": "freeform",
        "truncation_policy": {"mode": "tokens", "limit": 10000},
        "context_window": args.context_window,
        "max_context_window": args.context_window,
        "effective_context_window_percent": 90,
        "experimental_supported_tools": [],
        "input_modalities": ["text"],
        "supports_search_tool": False,
        "supports_experimental_context": False,
        "use_responses_lite": False,
    }
    catalog_path.write_text(json.dumps({"models": [model]}, indent=2) + "\n")
    config_path.write_text(
        f"model = {json.dumps(args.model)}\n"
        'model_provider = "eole"\n'
        f"model_catalog_json = {json.dumps(str(catalog_path))}\n"
        'web_search = "disabled"\n'
        f"model_context_window = {args.context_window}\n"
        f"model_auto_compact_token_limit = {auto_compact_token_limit}\n"
        "\n[model_providers.eole]\n"
        'name = "Local Eole"\n'
        f'base_url = {json.dumps(args.base_url.rstrip("/"))}\n'
        'wire_api = "responses"\n'
        "requires_openai_auth = false\n"
        "supports_websockets = false\n"
    )
    print(f"Created {config_path} and {catalog_path}")
    print(f"Launch the CLI or Linux app with CODEX_HOME={target}")


if __name__ == "__main__":
    main()
