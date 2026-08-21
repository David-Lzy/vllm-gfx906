#!/usr/bin/env bash
# Start one isolated Qwen3.8 TP2 Phase 22 server on development GPUs 2 and 3.
set -euo pipefail

readonly MODE="${1:-no-mtp}"
readonly IMAGE="local/vllm-gfx906:v0.27.1-phase22-qwen38"
readonly ROOT="/mnt/disk2/vllm-gfx906-build/phase-22"
readonly MODEL_DIR="${ROOT}/hf-model"
readonly CACHE_DIR="${ROOT}/cache"
readonly LOG_DIR="${ROOT}/logs"
readonly PORT=18075
readonly CONTAINER="vllm-gfx906-phase22-qwen38-${MODE}"
readonly SERVED_MODEL="qwen38-phase22"

case "${MODE}" in
    no-mtp|mtp1) ;;
    *) echo "Usage: $0 {no-mtp|mtp1}" >&2; exit 2 ;;
esac

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
    echo "Missing Phase 22 model at ${MODEL_DIR}; run fetch-gfx906-phase22-qwen38.sh first." >&2
    exit 1
fi

if ! curl --fail --silent --max-time 10 http://127.0.0.1:8002/health >/dev/null; then
    echo "Production health check failed; refusing to start an unrelated experiment." >&2
    exit 1
fi

mkdir -p "${CACHE_DIR}" "${LOG_DIR}"
docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true

extra_command=""
if [[ "${MODE}" == "mtp1" ]]; then
    extra_command=" --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":1}'"
fi

docker run -d --name "${CONTAINER}" --network host \
    --device /dev/kfd --device /dev/dri --group-add video --ipc host --shm-size 64g \
    -v "${MODEL_DIR}:/model:ro" \
    -v "${CACHE_DIR}:/root/.cache/vllm" \
    -e HIP_VISIBLE_DEVICES=2,3 \
    -e ROCR_VISIBLE_DEVICES=2,3 \
    -e PYTORCH_ROCM_ARCH=gfx906 \
    -e VLLM_TARGET_DEVICE=rocm \
    -e VLLM_ROCM_GFX906_PREFER_EXLLAMA=1 \
    -e OMP_NUM_THREADS=12 \
    -e OPENBLAS_NUM_THREADS=12 \
    -e MKL_NUM_THREADS=12 \
    -e NUMEXPR_NUM_THREADS=12 \
    -e TOKENIZERS_PARALLELISM=false \
    -e TORCH_NCCL_ASYNC_ERROR_HANDLING=3 \
    --entrypoint /bin/bash "${IMAGE}" -lc \
    "exec /opt/vllm-venv/bin/vllm serve /model \\
      --host 127.0.0.1 --port ${PORT} --served-model-name ${SERVED_MODEL} \\
      --tensor-parallel-size 2 --dtype float16 --quantization awq \\
      --max-model-len 100000 --gpu-memory-utilization 0.88 \\
      --max-num-seqs 2 --max-num-batched-tokens 8192 \\
      --limit-mm-per-prompt '{\"image\":64,\"video\":0}' \\
      --mm-processor-kwargs '{\"max_pixels\":16777216}' \\
      --mm-processor-cache-type shm --mm-processor-cache-gb 16 \\
      --mm-shm-cache-max-object-size-mb 512 --mm-tensor-ipc torch_shm \\
      --renderer-num-workers 1 --enable-prefix-caching --enable-chunked-prefill \\
      --mamba-cache-mode align --skip-mm-profiling \\
      --reasoning-parser qwen3 --default-chat-template-kwargs '{\"enable_thinking\":false}' \\
      --disable-log-requests${extra_command}" \
    > "${LOG_DIR}/${MODE}-container-id.txt"

for _ in $(seq 1 180); do
    if curl --fail --silent --max-time 10 "http://127.0.0.1:${PORT}/health" >/dev/null; then
        echo "${CONTAINER} is healthy on port ${PORT}."
        exit 0
    fi
    sleep 10
done

docker logs --tail 300 "${CONTAINER}" >&2 || true
echo "${CONTAINER} did not become healthy within 30 minutes." >&2
exit 1
