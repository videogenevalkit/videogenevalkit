#!/usr/bin/env bash
# Build the videogenevalkit Docker image.
#
# Prerequisites on the build host:
#   * docker installed (>= 20.10)
#   * a videvalkit-env.tar.gz tarball next to this script's repo root
#     (produce it via:  conda-pack -p /path/to/working/env -o videvalkit-env.tar.gz)
#
# Usage:
#   bash scripts/build_image.sh                    # default tag: videogenevalkit/videogenevalkit:0.1.0
#   bash scripts/build_image.sh 0.2.0              # custom version tag
#   IMAGE_NAME=ghcr.io/videogenevalkit/videogenevalkit \
#       bash scripts/build_image.sh 0.1.0          # custom registry
#
# After build, push with:
#   docker push videogenevalkit/videogenevalkit:0.1.0

set -eu

VERSION="${1:-0.1.0}"
IMAGE_NAME="${IMAGE_NAME:-videogenevalkit/videogenevalkit}"
ENV_TARBALL="${ENV_TARBALL:-videvalkit-env.tar.gz}"

# Resolve repo root
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -f "$ENV_TARBALL" ]; then
  echo "ERROR: $ENV_TARBALL not found in repo root ($REPO_ROOT)"
  echo
  echo "Produce one of:"
  echo "  1. From a working conda env:"
  echo "     conda-pack -p /path/to/env -o $ENV_TARBALL --ignore-missing-files"
  echo "  2. From the videogenevalkit HuggingFace dataset:"
  echo "     hf download videogenevalkit/env-tarball videvalkit-env.tar.gz \\"
  echo "                 --local-dir $REPO_ROOT"
  exit 1
fi

echo "=== building $IMAGE_NAME:$VERSION ==="
echo "  tarball: $ENV_TARBALL ($(du -h $ENV_TARBALL | awk '{print $1}'))"
echo

docker build \
    --build-arg ENV_TARBALL="$ENV_TARBALL" \
    -t "$IMAGE_NAME:$VERSION" \
    -t "$IMAGE_NAME:latest" \
    .

echo
echo "=== built $IMAGE_NAME:$VERSION ==="
docker images "$IMAGE_NAME" | head -3

echo
echo "Smoke-test with:"
echo "  docker run --rm --gpus all $IMAGE_NAME:$VERSION list benchmarks"
echo
echo "Push to registry with:"
echo "  docker push $IMAGE_NAME:$VERSION"
echo "  docker push $IMAGE_NAME:latest"
