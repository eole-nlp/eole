#!/usr/bin/env bash
# Run inside a GPU container. Keep build artifacts in its own source tree.
set -euo pipefail
cd /eole
python -c 'import torch; assert torch.cuda.is_available(), "Run with --gpus all"'
MAX_JOBS="${MAX_JOBS:-2}" python -m pip install -e '.[ui,translation,ocr,rag]' --no-build-isolation
python -c 'from eole import _ops; print(_ops.__file__)'
