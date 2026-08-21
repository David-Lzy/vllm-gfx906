#!/usr/bin/env bash
set -euo pipefail

wheel_dir=${TRITON_WHEEL_DIR:?set TRITON_WHEEL_DIR to the directory containing the patched Triton wheel}
artifact_root=${TRITON_ARTIFACT_ROOT:?set TRITON_ARTIFACT_ROOT outside this repository}
image=${TRITON_BUILD_IMAGE:-aiinfos/vllm-gfx906-mobydick:v0.23.1rc0.x-rocm7.2.1-pytorch2.11.0}
gpu=${TRITON_GPU_INDEX:-2}

wheel=$(find "$wheel_dir" -maxdepth 1 -name 'triton-*.whl' -print -quit)
test -n "$wheel"

result_dir="$artifact_root/results/$(date -u +%Y%m%dT%H%M%SZ)-wheel"
mkdir -p "$result_dir"
printf 'wheel=%s\nimage=%s\ngpu=%s\n' "$wheel" "$image" "$gpu" >"$result_dir/manifest.txt"

docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  --env HIP_VISIBLE_DEVICES="$gpu" \
  --volume "$wheel_dir:/wheel:ro" \
  --volume "$(pwd):/workspace/phase19:ro" \
  --workdir /workspace \
  "$image" \
  bash /workspace/phase19/tools/run-gfx906-phase19-triton37-wheel-smoke-container.sh \
  2>&1 | tee "$result_dir/triton37-wheel-smoke.log"

printf 'result_dir=%s\n' "$result_dir"
