#!/usr/bin/env bash
# Install the vision wheel matching the selected CUDA/PyTorch runtime.
set -euo pipefail
constraint=$(mktemp)
trap 'rm -f "$constraint"' EXIT
python -c 'import torch; print("torch==" + torch.__version__)' > "$constraint"
cuda_tag=$(python -c 'import torch; print(torch.version.cuda.replace(".", ""))')
python -m pip install torchvision --constraint "$constraint" \
    --index-url "https://download.pytorch.org/whl/cu${cuda_tag}"
python -c 'import torch, torchvision; print("Vision runtime:", torch.__version__, torchvision.__version__)'
