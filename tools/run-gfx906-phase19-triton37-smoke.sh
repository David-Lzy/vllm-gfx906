#!/usr/bin/env bash
set -euo pipefail

source_dir=${TRITON_SOURCE_DIR:?set TRITON_SOURCE_DIR to the Triton 3.7.1 checkout}
artifact_root=${TRITON_ARTIFACT_ROOT:?set TRITON_ARTIFACT_ROOT outside this repository}
image=${TRITON_BUILD_IMAGE:-aiinfos/vllm-gfx906-mobydick:v0.23.1rc0.x-rocm7.2.1-pytorch2.11.0}
gpu=${TRITON_GPU_INDEX:-2}

result_dir="$artifact_root/results/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$result_dir" "$artifact_root/ccache" "$artifact_root/triton-home"

printf 'source=%s\nimage=%s\ngpu=%s\n' "$source_dir" "$image" "$gpu" \
  >"$result_dir/manifest.txt"

docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  --env HIP_VISIBLE_DEVICES="$gpu" \
  --env TRITON_BUILD_WITH_CCACHE=true \
  --env TRITON_BUILD_WITH_CLANG_LLD=true \
  --env MAX_JOBS=8 \
  --env TRITON_HOME=/artifacts/triton-home \
  --env CCACHE_DIR=/artifacts/ccache \
  --volume "$source_dir:/workspace/triton" \
  --volume "$artifact_root:/artifacts" \
  --volume "$(pwd):/workspace/phase19:ro" \
  --workdir /workspace/triton \
  "$image" \
  bash /workspace/phase19/tools/run-gfx906-phase19-triton37-smoke-container.sh \
  2>&1 | tee "$result_dir/triton37-smoke.log"

printf 'result_dir=%s\n' "$result_dir"
