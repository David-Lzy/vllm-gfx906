#!/usr/bin/env bash
set -euo pipefail

source_dir=${TRITON_SOURCE_DIR:?set TRITON_SOURCE_DIR to the patched Triton 3.7.1 checkout}
artifact_root=${TRITON_ARTIFACT_ROOT:?set TRITON_ARTIFACT_ROOT outside this repository}
image=${TRITON_BUILD_IMAGE:-aiinfos/vllm-gfx906-mobydick:v0.23.1rc0.x-rocm7.2.1-pytorch2.11.0}

wheel_dir="$artifact_root/triton-wheels/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$wheel_dir" "$artifact_root/ccache"

printf 'source=%s\nimage=%s\n' "$source_dir" "$image" >"$wheel_dir/manifest.txt"

docker run --rm \
  --env TRITON_BUILD_WITH_CCACHE=true \
  --env TRITON_BUILD_WITH_CLANG_LLD=true \
  --env MAX_JOBS=8 \
  --env CCACHE_DIR=/artifacts/ccache \
  --volume "$source_dir:/workspace/triton" \
  --volume "$artifact_root:/artifacts" \
  --volume "$wheel_dir:/wheel" \
  --volume "$(pwd):/workspace/phase19:ro" \
  --workdir /workspace/triton \
  "$image" \
  bash /workspace/phase19/tools/build-gfx906-phase19-triton37-wheel-container.sh \
  2>&1 | tee "$wheel_dir/triton37-wheel.log"

test -n "$(find "$wheel_dir" -maxdepth 1 -name 'triton-*.whl' -print -quit)"
printf 'wheel_dir=%s\n' "$wheel_dir"
