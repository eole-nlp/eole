#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
if (( $# < 2 || $# > 3 )) || [[ ${3:-} != '' && ${3:-} != --push ]]; then
    echo "Usage: $0 TORCH_VERSION CUDA_VERSION [--push]" >&2; exit 2
fi
TORCH_VERSION=$1
CUDA_VERSION=$2
[[ $TORCH_VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ && $CUDA_VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "Expected numeric x.y.z versions" >&2; exit 2; }
IMAGE="ghcr.io/eole-nlp/eole-base:torch${TORCH_VERSION}-ubuntu24.04-cuda${CUDA_VERSION}"
# Other toolkit families should not install the default CUDA13.2 compiler.
NVCC_PACKAGE=${CUDA_NVCC_PACKAGE-cuda-nvcc-13-2=13.2.86-1}
if [[ $CUDA_VERSION != 13.2.* && ! -v CUDA_NVCC_PACKAGE ]]; then NVCC_PACKAGE=''; fi
docker build --progress=plain -t "$IMAGE" -f docker/Dockerfile-base \
    --build-arg "TORCH_VERSION=$TORCH_VERSION" --build-arg "CUDA_VERSION=$CUDA_VERSION" \
    --build-arg "CUDA_NVCC_PACKAGE=$NVCC_PACKAGE" .
if [[ ${3:-} == --push ]]; then docker push "$IMAGE"; fi
