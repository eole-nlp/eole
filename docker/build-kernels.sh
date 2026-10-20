#!/usr/bin/env bash
# Explicit TORCH_CUDA_ARCH_LIST enables GPU-less image builds. Otherwise target
# all visible devices for the deployment fallback or a local development build.
set -euo pipefail
cd /eole
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    python -c 'import torch; assert torch.cuda.is_available(), "Run with --gpus all or set TORCH_CUDA_ARCH_LIST"'
fi
# Force recompilation: an existing incompatible artifact must not be reused.
MAX_JOBS="${MAX_JOBS:-2}" python setup.py build_ext --inplace --force
python -c 'import torch; from eole import _ops; print(_ops.__file__)'
