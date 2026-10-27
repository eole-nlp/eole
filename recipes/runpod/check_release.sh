#!/usr/bin/env bash
# A numbered release must combine the same tagged Eole and launcher source.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
if (( $# != 2 )); then
    echo "Usage: $0 DEPLOY_TAG EOLE_IMAGE" >&2
    exit 2
fi
DEPLOY_TAG=$1
EOLE_IMAGE=$2
# Dev images and Docker-only revision tags keep the existing feature checks.
if [[ ! $DEPLOY_TAG =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then exit 0; fi
SOURCE_VERSION=$(python -c 'import runpy; print(runpy.run_path("eole/__init__.py")["__version__"])')
REVISION=$(git rev-parse HEAD)
TAG_REVISION=$(git rev-parse "refs/tags/$DEPLOY_TAG^{commit}" 2>/dev/null || true)
if [[ $SOURCE_VERSION != "$DEPLOY_TAG" || $TAG_REVISION != "$REVISION" ]]; then
    echo "Release $DEPLOY_TAG requires its exact Git tag and matching package version; use source_ref=$DEPLOY_TAG." >&2
    exit 2
fi
docker pull "$EOLE_IMAGE"
PARENT_VERSION=$(docker run --rm --entrypoint python "$EOLE_IMAGE" -c 'import eole; print(eole.__version__)')
PARENT_REVISION=$(docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$EOLE_IMAGE")
if [[ $PARENT_VERSION != "$DEPLOY_TAG" || $PARENT_REVISION != "$REVISION" ]]; then
    echo "Parent Eole image must be version $DEPLOY_TAG from commit $REVISION; got $PARENT_VERSION / $PARENT_REVISION. Rebuild the full Eole image from source_ref=$DEPLOY_TAG." >&2
    exit 2
fi
