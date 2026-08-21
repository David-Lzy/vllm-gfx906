#!/usr/bin/env bash
# Run the isolated Phase 28 Qwen3.6/Qwen3.8 TP2 attribution controls.
set -euo pipefail

readonly ACTION="${1:-}"
readonly MODEL_ID="${MODEL_ID:-}"
readonly MODEL_LABEL="${MODEL_LABEL:-}"
readonly MODE="${MODE:-no-mtp}"
readonly IMAGE="${IMAGE:-local/vllm-gfx906:v0.27.1-phase22-qwen38}"
readonly SPLITKV_IMAGE="${SPLITKV_IMAGE:-local/vllm-gfx906:v0.27.1-phase28-splitkv}"
readonly SPLITKV_GFX906="${SPLITKV_GFX906:-0}"
readonly DOWNLOAD_IMAGE="${DOWNLOAD_IMAGE:-local/vllm-gfx906:v0.27.1-phase21-llmm1}"
readonly ROOT="${ROOT:-/mnt/disk2/vllm-gfx906-build/phase-28}"
readonly PORT="${PORT:-18078}"
readonly FIXTURE="${FIXTURE:-/mnt/disk2/vllm-gfx906-build/phase-19/fixtures/phase19-gpu2-256.png}"
readonly SERVED_MODEL="phase28-${MODEL_LABEL}"
readonly MODEL_DIR="${ROOT}/models/${MODEL_LABEL}"
readonly CACHE_LABEL="${CACHE_LABEL:-${MODEL_LABEL}-${MODE}}"
readonly CACHE_DIR="${ROOT}/cache/${CACHE_LABEL}"
readonly LOG_DIR="${ROOT}/logs/${MODEL_LABEL}-${MODE}"
readonly CONTAINER="vllm-gfx906-phase28-${MODEL_LABEL}-${MODE}"
readonly MIN_FREE_BEFORE_GIB=50
readonly MIN_FREE_AFTER_GIB=25

usage() {
    cat >&2 <<'EOF'
Usage:
  MODEL_ID=<hf repo> MODEL_LABEL=<short name> tools/run-gfx906-phase28-qwen27b.sh \
    {build-image|build-splitkv-image|fetch|start|gates|bench|slope|stop|cleanup}

Environment:
  MODE=no-mtp|mtp1|mtp2|mtp4   default: no-mtp
  PORT=18078                   temporary localhost port
  ROOT=/mnt/disk2/.../phase-28 disposable phase storage
  SPLITKV_GFX906=1             opt in to the gfx906 split-KV candidate
  IMAGE=<tag>                  use SPLITKV_IMAGE after build-splitkv-image
  CACHE_LABEL=<name>           isolate a candidate's compile and vLLM cache
  CONTEXT_WORDS=32768          repeated long-context word count for slope

Only one model cache is allowed at a time. Run cleanup after recording a
candidate before fetching the next checkpoint. cleanup requires CONFIRM=delete.
EOF
    exit 2
}

require_model() {
    if [[ -z "${MODEL_ID}" || -z "${MODEL_LABEL}" ]]; then
        echo "MODEL_ID and MODEL_LABEL are required." >&2
        usage
    fi
}

available_gib() {
    df --output=avail -BG /mnt/disk2 | tail -1 | tr -dc '0-9'
}

production_is_healthy() {
    curl --fail --silent --max-time 10 http://127.0.0.1:8002/health >/dev/null
}

require_fixture() {
    if [[ ! -r "${FIXTURE}" ]]; then
        echo "Missing readable Phase 28 fixture: ${FIXTURE}" >&2
        exit 1
    fi
}

require_server() {
    if ! curl --fail --silent --max-time 10 "http://127.0.0.1:${PORT}/health" \
        >/dev/null; then
        echo "Phase 28 server is not healthy on port ${PORT}." >&2
        exit 1
    fi
}

wait_for_server() {
    for _ in $(seq 1 180); do
        if curl --fail --silent --max-time 10 "http://127.0.0.1:${PORT}/health" \
            >/dev/null; then
            echo "${CONTAINER} is healthy on port ${PORT}."
            return 0
        fi
        if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER}" 2>/dev/null || true)" \
            != "true" ]]; then
            docker logs --tail 300 "${CONTAINER}" >&2 || true
            echo "${CONTAINER} exited before becoming healthy." >&2
            return 1
        fi
        sleep 10
    done

    docker logs --tail 300 "${CONTAINER}" >&2 || true
    echo "${CONTAINER} did not become healthy within 30 minutes." >&2
    return 1
}

speculative_config() {
    case "${MODE}" in
        no-mtp) ;;
        mtp1) printf '%s' " --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":1}'" ;;
        mtp2) printf '%s' " --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":2}'" ;;
        mtp4) printf '%s' " --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":4}'" ;;
        *) echo "Unsupported MODE: ${MODE}" >&2; exit 2 ;;
    esac
}

start() {
    require_model
    if [[ "${SPLITKV_GFX906}" != "0" && "${SPLITKV_GFX906}" != "1" ]]; then
        echo "SPLITKV_GFX906 must be 0 or 1." >&2
        exit 2
    fi
    if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
        echo "Missing model at ${MODEL_DIR}; run fetch first." >&2
        exit 1
    fi
    if ! production_is_healthy; then
        echo "Production health check failed; refusing unrelated experiment." >&2
        exit 1
    fi

    mkdir -p "${CACHE_DIR}/triton-cache" "${LOG_DIR}"
    docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true

    local extra_command
    extra_command="$(speculative_config)"
    docker run -d --name "${CONTAINER}" --network host \
        --device /dev/kfd --device /dev/dri --group-add video --ipc host --shm-size 64g \
        -v "${MODEL_DIR}:/model:ro" \
        -v "${CACHE_DIR}:/root/.cache/vllm" \
        -v "${CACHE_DIR}/triton-cache:/root/.triton/cache" \
        -e HIP_VISIBLE_DEVICES=2,3 \
        -e PYTORCH_ROCM_ARCH=gfx906 \
        -e ROCM_ARCH=gfx906 \
        -e ROCM_PATH=/opt/rocm \
        -e ROCBLAS_TENSILE_LIBPATH=/opt/rocm/lib/rocblas/library \
        -e VLLM_TARGET_DEVICE=rocm \
        -e VLLM_CACHE_ROOT=/root/.cache/vllm \
        -e VLLM_ROCM_GFX906_PREFER_EXLLAMA=1 \
        -e VLLM_ROCM_ENABLE_GFX906_SPLITKV="${SPLITKV_GFX906}" \
        -e TRITON_CACHE_DIR=/root/.triton/cache \
        -e HF_HUB_OFFLINE=1 \
        -e TRANSFORMERS_OFFLINE=1 \
        -e OMP_NUM_THREADS=12 \
        -e OPENBLAS_NUM_THREADS=12 \
        -e MKL_NUM_THREADS=12 \
        -e NUMEXPR_NUM_THREADS=12 \
        -e TOKENIZERS_PARALLELISM=false \
        -e TORCH_NCCL_ASYNC_ERROR_HANDLING=3 \
        --entrypoint /bin/bash "${IMAGE}" -lc \
        "exec /opt/vllm-venv/bin/vllm serve /model \\
          --host 127.0.0.1 --port ${PORT} --served-model-name ${SERVED_MODEL} \\
          --tensor-parallel-size 2 --dtype float16 --trust-remote-code \\
          --max-model-len 100000 --gpu-memory-utilization 0.88 \\
          --max-num-seqs 2 --max-num-batched-tokens 8192 \\
          --limit-mm-per-prompt '{\"image\":64,\"video\":0}' \\
          --mm-processor-kwargs '{\"max_pixels\":16777216}' \\
          --mm-processor-cache-type shm --mm-processor-cache-gb 16 \\
          --mm-shm-cache-max-object-size-mb 512 --mm-tensor-ipc direct_rpc \\
          --mm-encoder-tp-mode data --renderer-num-workers 1 \\
          --enable-prefix-caching --enable-chunked-prefill --mamba-cache-mode align \\
          --skip-mm-profiling --reasoning-parser qwen3 \\
          --default-chat-template-kwargs '{\"enable_thinking\":false}'${extra_command}" \
        > "${LOG_DIR}/container-id.txt"
    wait_for_server
}

fetch() {
    require_model
    if [[ -f "${MODEL_DIR}/config.json" ]]; then
        echo "Model already populated: ${MODEL_DIR}" >&2
        exit 1
    fi
    if find "${ROOT}/models" -mindepth 2 -name config.json -print -quit 2>/dev/null | \
        grep -q .; then
        echo "Another Phase 28 model exists; record and clean it before fetching again." >&2
        exit 1
    fi

    local before after revision result_dir
    before="$(available_gib)"
    if (( before < MIN_FREE_BEFORE_GIB )); then
        echo "Refusing download: ${before} GiB free, need ${MIN_FREE_BEFORE_GIB} GiB." >&2
        exit 1
    fi
    result_dir="${ROOT}/results/$(date -u +%Y%m%dT%H%M%SZ)-${MODEL_LABEL}-resolution"
    mkdir -p "${MODEL_DIR}" "${result_dir}"
    revision="$(curl --fail --silent --show-error \
        "https://huggingface.co/api/models/${MODEL_ID}" | jq -er '.sha')"
    jq -n --arg model "${MODEL_ID}" --arg label "${MODEL_LABEL}" \
        --arg revision "${revision}" --arg fetched_at "$(date --iso-8601=seconds)" \
        --argjson free_gib_before "${before}" \
        '{model: $model, label: $label, revision: $revision, fetched_at: $fetched_at,
          free_gib_before: $free_gib_before}' > "${result_dir}/model-resolution.json"

    docker run --rm --entrypoint /opt/vllm-venv/bin/hf \
        -v "${MODEL_DIR}:/model-dir" \
        -e HF_XET_HIGH_PERFORMANCE=1 \
        -e HF_XET_NUM_CONCURRENT_RANGE_GETS=32 \
        -e HF_HUB_DOWNLOAD_TIMEOUT=60 \
        "${DOWNLOAD_IMAGE}" download "${MODEL_ID}" --revision "${revision}" \
        --local-dir /model-dir

    after="$(available_gib)"
    jq --argjson free_gib_after "${after}" '. + {free_gib_after: $free_gib_after}' \
        "${result_dir}/model-resolution.json" > "${result_dir}/model-resolution.tmp"
    mv "${result_dir}/model-resolution.tmp" "${result_dir}/model-resolution.json"
    if (( after < MIN_FREE_AFTER_GIB )); then
        echo "Download left ${after} GiB free; below ${MIN_FREE_AFTER_GIB} GiB floor." >&2
        exit 1
    fi
}

post_case() {
    local result_dir="$1"
    local name="$2"
    local payload="$3"
    local output="${result_dir}/${name}.json"
    local code
    code="$(curl --silent --show-error --output "${output}" --write-out '%{http_code}' \
        --max-time 900 -H 'content-type: application/json' --data "${payload}" \
        "http://127.0.0.1:${PORT}/v1/chat/completions")"
    [[ "${code}" == "200" ]]
    jq -e '.choices[0].message.content | strings | length > 0' "${output}" >/dev/null
    jq -n --arg case_name "${name}" --arg http_status "${code}" \
        --arg response_file "${output}" --arg completed_at "$(date --iso-8601=seconds)" \
        '{case: $case_name, http_status: $http_status, response_file: $response_file,
          completed_at: $completed_at}' >> "${result_dir}/gates.jsonl"
}

gates() {
    require_model
    require_fixture
    require_server
    local result_dir image_url payload
    result_dir="${ROOT}/results/$(date -u +%Y%m%dT%H%M%SZ)-${MODEL_LABEL}-${MODE}-gates"
    mkdir -p "${result_dir}"
    image_url="data:image/png;base64,$(base64 -w0 "${FIXTURE}")"

    post_case "${result_dir}" text "$(jq -nc --arg model "${SERVED_MODEL}" \
        '{model: $model, temperature: 0, max_tokens: 64,
          messages: [{role: "user", content: "Reply exactly: phase 28 text healthy."}]}')"
    payload="$(jq -nc --arg model "${SERVED_MODEL}" --arg image "${image_url}" \
        '{model: $model, temperature: 0, max_tokens: 64,
          messages: [{role: "user", content: [{type: "text", text: "Name the dominant image color in one word."},
            {type: "image_url", image_url: {url: $image}}]}]}')"
    post_case "${result_dir}" image_1 "${payload}"
    payload="$(jq -nc --arg model "${SERVED_MODEL}" --arg one "${image_url}" \
        --arg two "${image_url}" '{model: $model, temperature: 0, max_tokens: 64,
          messages: [{role: "user", content: [{type: "text", text: "Two identical images are supplied. State one dominant color."},
            {type: "image_url", image_url: {url: $one}}, {type: "image_url", image_url: {url: $two}}]}]}')"
    post_case "${result_dir}" image_2 "${payload}"
    for index in 1 2 3; do
        post_case "${result_dir}" "json_${index}" "$(jq -nc --arg model "${SERVED_MODEL}" \
            '{model: $model, temperature: 0, max_tokens: 32,
              response_format: {type: "json_object"},
              messages: [{role: "user", content: "Return exactly one JSON object with boolean key ok set to true."}]}')"
        jq -er '.choices[0].message.content | fromjson | select(.ok == true)' \
            "${result_dir}/json_${index}.json" >/dev/null
    done
    curl --fail --silent "http://127.0.0.1:${PORT}/v1/models" > "${result_dir}/models.json"
    curl --fail --silent "http://127.0.0.1:${PORT}/metrics" > "${result_dir}/metrics.prom"
    echo "${result_dir}"
}

bench() {
    require_model
    require_server
    local result_dir payload
    result_dir="${ROOT}/results/$(date -u +%Y%m%dT%H%M%SZ)-${MODEL_LABEL}-${MODE}-fixed128"
    mkdir -p "${result_dir}"
    payload="$(jq -nc --arg model "${SERVED_MODEL}" \
        '{model: $model, temperature: 0, min_tokens: 128, max_tokens: 128,
          messages: [{role: "user", content: "Write exactly 128 concise tokens about reliable GPU inference."}]}')"

    local sample output started ended elapsed tokens throughput
    for sample in warmup sample_1 sample_2 sample_3 sample_4 sample_5; do
        output="${result_dir}/${sample}.json"
        started="$(date +%s.%N)"
        curl --fail --silent --show-error --max-time 900 -H 'content-type: application/json' \
            --data "${payload}" "http://127.0.0.1:${PORT}/v1/chat/completions" > "${output}"
        ended="$(date +%s.%N)"
        elapsed="$(awk -v start="${started}" -v end="${ended}" 'BEGIN { printf "%.6f", end - start }')"
        tokens="$(jq -er '.usage.completion_tokens' "${output}")"
        throughput="$(awk -v tokens="${tokens}" -v elapsed="${elapsed}" \
            'BEGIN { printf "%.6f", tokens / elapsed }')"
        jq -n --arg sample "${sample}" --argjson elapsed_seconds "${elapsed}" \
            --argjson completion_tokens "${tokens}" --argjson completion_tok_s "${throughput}" \
            '{sample: $sample, elapsed_seconds: $elapsed_seconds,
              completion_tokens: $completion_tokens, completion_tok_s: $completion_tok_s}' \
            >> "${result_dir}/samples.jsonl"
    done
    jq -s 'map(select(.sample != "warmup") | .completion_tok_s) | sort |
        {count: length, median_tok_s: .[length / 2 | floor], samples_tok_s: .}' \
        "${result_dir}/samples.jsonl" > "${result_dir}/summary.json"
    curl --fail --silent "http://127.0.0.1:${PORT}/metrics" > "${result_dir}/metrics.prom"
    docker logs --tail 500 "${CONTAINER}" > "${result_dir}/server-tail.log" 2>&1 || true
    cat "${result_dir}/summary.json"
}

post_slope_case() {
    local result_dir="$1"
    local name="$2"
    local payload_file="$3"
    local output="${result_dir}/${name}.json"
    local started ended elapsed code
    started="$(date +%s.%N)"
    code="$(curl --silent --show-error --output "${output}" --write-out '%{http_code}' \
        --max-time 1800 -H 'content-type: application/json' \
        --data-binary "@${payload_file}" \
        "http://127.0.0.1:${PORT}/v1/chat/completions")"
    ended="$(date +%s.%N)"
    elapsed="$(awk -v start="${started}" -v end="${ended}" \
        'BEGIN { printf "%.6f", end - start }')"
    [[ "${code}" == "200" ]]
    jq -e '.choices[0].message.content | strings | length > 0' "${output}" >/dev/null
    jq -n --arg case_name "${name}" --argjson elapsed_seconds "${elapsed}" \
        --argjson http_status "${code}" --arg response_file "${output}" \
        --arg completed_at "$(date --iso-8601=seconds)" \
        '{case: $case_name, elapsed_seconds: $elapsed_seconds,
          http_status: $http_status, response_file: $response_file,
          usage: {prompt_tokens: null, completion_tokens: null},
          completed_at: $completed_at}' \
        > "${result_dir}/${name}.meta.json"
    jq --slurpfile response "${output}" \
        '.usage.prompt_tokens = ($response[0].usage.prompt_tokens // null) |
         .usage.completion_tokens = ($response[0].usage.completion_tokens // null)' \
        "${result_dir}/${name}.meta.json" > "${result_dir}/${name}.tmp"
    mv "${result_dir}/${name}.tmp" "${result_dir}/${name}.meta.json"
}

slope() {
    require_model
    require_server
    local context_words="${CONTEXT_WORDS:-32768}"
    if ! [[ "${context_words}" =~ ^[1-9][0-9]*$ ]]; then
        echo "CONTEXT_WORDS must be a positive integer." >&2
        exit 2
    fi

    local result_dir context_file prime_payload_file decode_payload_file
    local decode_tokens decode_elapsed
    result_dir="${ROOT}/results/$(date -u +%Y%m%dT%H%M%SZ)-${MODEL_LABEL}-${MODE}-longctx"
    mkdir -p "${result_dir}"
    context_file="${result_dir}/context.txt"
    awk -v count="${context_words}" \
        'BEGIN { for (i = 0; i < count; i++) printf " context" }' > "${context_file}"

    prime_payload_file="${result_dir}/prime.request.json"
    jq -n --arg model "${SERVED_MODEL}" --rawfile prompt "${context_file}" \
        '{model: $model, temperature: 0, min_tokens: 1, max_tokens: 1,
          messages: [{role: "user", content: $prompt}]}' > "${prime_payload_file}"
    decode_payload_file="${result_dir}/cached_decode.request.json"
    jq -n --arg model "${SERVED_MODEL}" --rawfile prompt "${context_file}" \
        '{model: $model, temperature: 0, min_tokens: 8, max_tokens: 8,
          messages: [{role: "user", content: $prompt}]}' > "${decode_payload_file}"
    post_slope_case "${result_dir}" prime "${prime_payload_file}"
    post_slope_case "${result_dir}" cached_decode "${decode_payload_file}"

    decode_tokens="$(jq -er '.usage.completion_tokens' "${result_dir}/cached_decode.meta.json")"
    decode_elapsed="$(jq -er '.elapsed_seconds' "${result_dir}/cached_decode.meta.json")"
    jq --argjson context_words "${context_words}" \
        --argjson completion_tokens "${decode_tokens}" \
        --argjson elapsed_seconds "${decode_elapsed}" \
        '{context_words: $context_words, completion_tokens: $completion_tokens,
          cached_decode_elapsed_seconds: $elapsed_seconds,
          cached_decode_tok_s: ($completion_tokens / $elapsed_seconds)}' \
        > "${result_dir}/summary.json"
    curl --fail --silent "http://127.0.0.1:${PORT}/metrics" > "${result_dir}/metrics.prom"
    cat "${result_dir}/summary.json"
}

stop() {
    require_model
    docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
}

cleanup() {
    require_model
    if [[ "${CONFIRM:-}" != "delete" ]]; then
        echo "Refusing cleanup without CONFIRM=delete." >&2
        exit 2
    fi
    if docker inspect -f '{{.State.Running}}' "${CONTAINER}" 2>/dev/null | grep -qx true; then
        echo "Stop the current candidate before cleanup." >&2
        exit 1
    fi
    rm -rf --one-file-system "${MODEL_DIR}" "${ROOT}/cache/${MODEL_LABEL}-"*
}

case "${ACTION}" in
    build-image)
        docker build -f docker/Dockerfile.gfx906-v027-phase22-qwen38 \
            -t "${IMAGE}" .
        ;;
    build-splitkv-image)
        docker build -f docker/Dockerfile.gfx906-v027-phase28-splitkv \
            -t "${SPLITKV_IMAGE}" .
        ;;
    fetch) fetch ;;
    start) start ;;
    gates) gates ;;
    bench) bench ;;
    slope) slope ;;
    stop) stop ;;
    cleanup) cleanup ;;
    *) usage ;;
esac
