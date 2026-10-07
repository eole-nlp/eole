#!/usr/bin/env bash
# Build API reference Markdown and stage it with handbook/recipe pages.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(cd .. && pwd)${PYTHONPATH:+:$PYTHONPATH}"

python -m sphinx -E -a -W -b markdown source sphinx_markdown/markdown
python assemble_docs.py
cd docusaurus_tsx
corepack yarn clear
corepack yarn typecheck
corepack yarn build
