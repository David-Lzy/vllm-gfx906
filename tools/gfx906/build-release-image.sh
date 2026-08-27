#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
revision=$(git -C "$repo_root" rev-parse HEAD)
image=${1:-ghcr.io/david-lzy/vllm-gfx906:v0.28.0-gfx906.1}

git -C "$repo_root" diff --quiet
git -C "$repo_root" diff --cached --quiet

exec docker buildx build \
  --load \
  --file "$repo_root/docker/Dockerfile.gfx906-v028-release" \
  --build-arg "VLLM_RELEASE_REVISION=$revision" \
  --tag "$image" \
  "$repo_root"
