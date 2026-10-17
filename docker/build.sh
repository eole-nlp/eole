#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
if (( $# < 3 || $# > 4 )) || [[ ${4:-} != '' && ${4:-} != --push ]]; then
    echo "Usage: $0 SOURCE_VERSION TORCH_VERSION CUDA_VERSION [--push]" >&2; exit 2
fi
EOLE_VERSION=$1
TORCH_VERSION=$2
CUDA_VERSION=$3
REVISION=$(git rev-parse HEAD)
if [[ $EOLE_VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    [[ $(git rev-parse "refs/tags/$EOLE_VERSION^{commit}") == "$REVISION" ]] || { echo "Checkout release tag $EOLE_VERSION first" >&2; exit 2; }
elif [[ $EOLE_VERSION != "dev-$(git rev-parse --short=12 HEAD)" ]]; then
    echo "Use a release tag or dev-$(git rev-parse --short=12 HEAD)" >&2; exit 2
fi
[[ $TORCH_VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ && $CUDA_VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "Expected numeric x.y.z versions" >&2; exit 2; }
# A release label must describe clean source, not locally modified files.
if [[ ${4:-} == --push ]] && ! git diff --quiet HEAD --; then
    echo "Commit tracked changes before publishing" >&2; exit 2
fi
IMAGE="ghcr.io/eole-nlp/eole:${EOLE_VERSION}-torch${TORCH_VERSION}-ubuntu24.04-cuda${CUDA_VERSION}"
docker build --progress=plain -t "$IMAGE" -f docker/Dockerfile \
    --build-arg "TORCH_VERSION=$TORCH_VERSION" --build-arg "CUDA_VERSION=$CUDA_VERSION" \
    --build-arg "EOLE_VERSION=$EOLE_VERSION" --build-arg "VCS_REF=$REVISION" .
if [[ ${4:-} == --push ]]; then docker push "$IMAGE"; fi
