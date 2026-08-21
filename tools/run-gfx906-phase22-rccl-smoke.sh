#!/usr/bin/env bash
# Run the minimal TP2 RCCL smoke on development GPUs 2 and 3.
set -euo pipefail

readonly IMAGE="local/vllm-gfx906:v0.27.1-phase22-qwen38"
readonly ROOT="/mnt/disk2/vllm-gfx906-build/phase-22"
readonly RESULT_DIR="${ROOT}/results/$(date -u +%Y%m%dT%H%M%SZ)-rccl-smoke"

if ! curl --fail --silent --max-time 10 http://127.0.0.1:8002/health >/dev/null; then
    echo "Production health check failed; refusing to start an unrelated experiment." >&2
    exit 1
fi

mkdir -p "${RESULT_DIR}"

docker run --rm \
    --device /dev/kfd --device /dev/dri --group-add video --ipc host --shm-size 4g \
    -v "$(pwd)/tools/gfx906_tp2_rccl_smoke.py:/opt/gfx906_tp2_rccl_smoke.py:ro" \
    -e HIP_VISIBLE_DEVICES=2,3 \
    -e PYTORCH_ROCM_ARCH=gfx906 \
    -e ROCM_ARCH=gfx906 \
    -e ROCM_PATH=/opt/rocm \
    -e ROCBLAS_TENSILE_LIBPATH=/opt/rocm/lib/rocblas/library \
    -e TORCH_NCCL_ASYNC_ERROR_HANDLING=3 \
    --entrypoint /opt/vllm-venv/bin/torchrun "${IMAGE}" \
    --standalone --nproc-per-node=2 /opt/gfx906_tp2_rccl_smoke.py \
    2>&1 | tee "${RESULT_DIR}/rccl-smoke.log"

printf '%s\n' "${RESULT_DIR}"
