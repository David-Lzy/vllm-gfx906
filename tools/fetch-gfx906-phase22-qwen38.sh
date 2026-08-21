#!/usr/bin/env bash
# Fetch the Phase 22 checkpoint into an isolated, disposable disk2 directory.
set -euo pipefail

readonly IMAGE="local/vllm-gfx906:v0.27.1-phase21-llmm1"
readonly MODEL_ID="cyankiwi/Qwen3.8-27B-AWQ-INT4"
readonly ROOT="/mnt/disk2/vllm-gfx906-build/phase-22"
readonly MODEL_DIR="${ROOT}/hf-model"
readonly RESULT_DIR="${ROOT}/results"
readonly MIN_FREE_GIB=50

available_gib() {
    df --output=avail -BG /mnt/disk2 | tail -1 | tr -dc '0-9'
}

free_gib="$(available_gib)"
if (( free_gib < MIN_FREE_GIB )); then
    echo "Refusing Qwen3.8 download: ${free_gib} GiB free, need ${MIN_FREE_GIB} GiB." >&2
    exit 1
fi

if [[ -e "${MODEL_DIR}/config.json" ]]; then
    echo "Model directory already populated: ${MODEL_DIR}" >&2
    exit 1
fi

mkdir -p "${MODEL_DIR}" "${RESULT_DIR}"
revision="$(curl --fail --silent --show-error \
    "https://huggingface.co/api/models/${MODEL_ID}" | jq -er '.sha')"

jq -n \
    --arg model "${MODEL_ID}" \
    --arg revision "${revision}" \
    --arg fetched_at "$(date --iso-8601=seconds)" \
    --argjson free_gib_before "${free_gib}" \
    '{model: $model, revision: $revision, fetched_at: $fetched_at,
      free_gib_before: $free_gib_before}' \
    > "${RESULT_DIR}/model-resolution.json"

docker run --rm \
    --entrypoint /opt/vllm-venv/bin/hf \
    -v "${MODEL_DIR}:/model-dir" \
    -e HF_XET_HIGH_PERFORMANCE=1 \
    -e HF_XET_NUM_CONCURRENT_RANGE_GETS=64 \
    -e HF_HUB_DOWNLOAD_TIMEOUT=60 \
    "${IMAGE}" \
    download "${MODEL_ID}" --revision "${revision}" --local-dir /model-dir

free_gib="$(available_gib)"
if (( free_gib < 25 )); then
    echo "Download left ${free_gib} GiB free; below Phase 22 25 GiB floor." >&2
    exit 1
fi

jq --argjson free_gib_after "${free_gib}" \
    '. + {free_gib_after: $free_gib_after}' \
    "${RESULT_DIR}/model-resolution.json" \
    > "${RESULT_DIR}/model-resolution.tmp"
mv "${RESULT_DIR}/model-resolution.tmp" "${RESULT_DIR}/model-resolution.json"
