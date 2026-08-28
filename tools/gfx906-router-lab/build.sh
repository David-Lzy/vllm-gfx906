#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=/dev/null
source "$script_dir/router-source.env"

image=${1:-$DEFAULT_IMAGE}
work_dir=$(mktemp -d "${TMPDIR:-/tmp}/gfx906-router-lab.XXXXXX")
trap 'rm -rf "$work_dir"' EXIT

git clone --filter=blob:none --no-checkout "$ROUTER_REPOSITORY" "$work_dir/router"
git -C "$work_dir/router" fetch --depth 1 origin "$ROUTER_COMMIT"
git -C "$work_dir/router" checkout --detach FETCH_HEAD
git -C "$work_dir/router" apply --check "$script_dir"/patches/*.patch
git -C "$work_dir/router" apply "$script_dir"/patches/*.patch

DOCKER_BUILDKIT=1 docker build \
  --file "$work_dir/router/Dockerfile.router" \
  --label "org.opencontainers.image.source=$ROUTER_REPOSITORY" \
  --label "org.opencontainers.image.revision=$ROUTER_COMMIT" \
  --label "org.opencontainers.image.version=${ROUTER_VERSION}-queue-aware-phase165" \
  --tag "$image" \
  "$work_dir/router"

docker image inspect "$image" --format '{{.Id}}'
