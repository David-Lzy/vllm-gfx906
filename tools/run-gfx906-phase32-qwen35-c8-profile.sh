#!/usr/bin/env bash
# Profile the v0.27 Qwen3.5 9B C8 regression on development GPU2 only.
set -euo pipefail

readonly ACTION="${1:-}"
readonly IMAGE="${IMAGE:-local/vllm-gfx906:v0.27.1-phase28-gptq-wna16}"
readonly MODEL="${MODEL:-cyankiwi/Qwen3.5-9B-AWQ-4bit}"
readonly HF_CACHE_DIR="${HF_CACHE_DIR:-/mnt/disk2/hf_cache}"
readonly FIXTURE="${FIXTURE:-/mnt/disk2/vllm-gfx906-build/phase-19/fixtures/phase19-gpu2-256.png}"
readonly ROOT="${ROOT:-/mnt/disk2/vllm-gfx906-build/phase-32}"
readonly GPU="${GPU:-2}"
readonly PORT="${PORT:-18079}"
# Keep an explicitly empty value so older vLLM controls can use auto selection.
readonly LINEAR_BACKEND="${LINEAR_BACKEND-gfx906_gptq}"
readonly TORCH_PROFILER_RECORD_SHAPES="${TORCH_PROFILER_RECORD_SHAPES:-false}"
readonly CONTAINER="vllm-gfx906-phase32-qwen35-c8"
readonly CACHE_DIR="${ROOT}/cache"
readonly RESULT_DIR="${ROOT}/results/$(date -u +%Y%m%dT%H%M%SZ)-qwen35-c8"
readonly ENDPOINT="http://127.0.0.1:${PORT}"

usage() {
    cat >&2 <<'EOF'
Usage: tools/run-gfx906-phase32-qwen35-c8-profile.sh {run|start|profile|gates|stop|cleanup}

This is an isolated GPU2 experiment. It refuses to run when production port
8002 is unhealthy and never changes the production service.
EOF
}

case "${TORCH_PROFILER_RECORD_SHAPES}" in
    true|false) ;;
    *)
        printf 'TORCH_PROFILER_RECORD_SHAPES must be true or false.\n' >&2
        exit 2
        ;;
esac

require_server() {
    curl --fail --silent --show-error --max-time 10 "${ENDPOINT}/health" >/dev/null
}

require_production() {
    curl --fail --silent --show-error --max-time 10 http://127.0.0.1:8002/health >/dev/null
}

wait_for_server() {
    local deadline=$((SECONDS + 1800))
    while ! require_server; do
        if [[ "$(docker inspect --format '{{.State.Running}}' "${CONTAINER}" 2>/dev/null || true)" != "true" ]]; then
            docker logs --tail 300 "${CONTAINER}" >&2 || true
            return 1
        fi
        if (( SECONDS >= deadline )); then
            docker logs --tail 300 "${CONTAINER}" >&2 || true
            printf 'Timed out waiting for %s.\n' "${CONTAINER}" >&2
            return 1
        fi
        sleep 10
    done
}

payload_dir() {
    printf '%s/payloads' "${ROOT}"
}

write_payloads() {
    local dir image_url
    dir="$(payload_dir)"
    mkdir -p "${dir}"
    jq -n --arg model "${MODEL}" \
        '{model: $model, temperature: 0, min_tokens: 64, max_tokens: 64,
          messages: [{role: "user", content: "Write concise factual notes about reliable GPU inference until the requested length is reached."}]}' \
        > "${dir}/text.json"
    image_url="data:image/png;base64,$(base64 -w0 "${FIXTURE}")"
    for count in 1 2; do
        jq -nc --arg model "${MODEL}" --arg image "${image_url}" --argjson count "${count}" \
            '[range(0; $count) | {type: "image_url", image_url: {url: $image}}] as $images |
             {model: $model, temperature: 0, max_tokens: 32,
              messages: [{role: "user", content: ([{type: "text", text: "Describe the supplied image concisely."}] + $images)}]}' \
            > "${dir}/image${count}.json"
    done
    jq -n --arg model "${MODEL}" \
        '{model: $model, temperature: 0, max_tokens: 32,
          response_format: {type: "json_object"},
          messages: [{role: "user", content: "Return exactly one JSON object with boolean key ok set to true."}]}' \
        > "${dir}/json.json"
}

post_checked() {
    local payload="$1" output="$2"
    curl --fail --silent --show-error --connect-timeout 10 --max-time 900 \
        -H 'content-type: application/json' --data-binary "@${payload}" \
        "${ENDPOINT}/v1/chat/completions" > "${output}"
    jq -er '.choices[0].message.content | strings | select(length > 0)' "${output}" >/dev/null
}

start() {
    local linear_backend_args=()

    require_production
    [[ -d "${HF_CACHE_DIR}" && -f "${FIXTURE}" ]] || {
        printf 'HF cache or image fixture is unavailable.\n' >&2
        exit 2
    }
    docker image inspect "${IMAGE}" >/dev/null
    if [[ -n "${LINEAR_BACKEND}" ]]; then
        linear_backend_args+=(--linear-backend "${LINEAR_BACKEND}")
    fi
    docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
    mkdir -p "${CACHE_DIR}/triton-cache" "${CACHE_DIR}/torch-profiler"
    docker run --detach --name "${CONTAINER}" --network host \
        --device /dev/kfd --device /dev/dri --group-add video --cap-add SYS_NICE \
        --ipc host --shm-size 64g \
        --volume "${HF_CACHE_DIR}:/root/.cache/huggingface:ro" \
        --volume "${CACHE_DIR}:/root/.cache/vllm" \
        --volume "${CACHE_DIR}/triton-cache:/root/.triton/cache" \
        --env HIP_VISIBLE_DEVICES="${GPU}" \
        --env PYTORCH_ROCM_ARCH=gfx906 --env ROCM_ARCH=gfx906 --env ROCM_PATH=/opt/rocm \
        --env VLLM_TARGET_DEVICE=rocm --env VLLM_CACHE_ROOT=/root/.cache/vllm \
        --env VLLM_ENGINE_READY_TIMEOUT_S=1800 --env VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200 \
        --env VLLM_ROCM_GFX906_PREFER_EXLLAMA=1 --env VLLM_CUSTOM_SCOPES_FOR_PROFILING=1 \
        --env OMP_NUM_THREADS=12 --env OPENBLAS_NUM_THREADS=12 --env MKL_NUM_THREADS=12 \
        --env NUMEXPR_NUM_THREADS=12 --env TOKENIZERS_PARALLELISM=false \
        --env TORCH_NCCL_ASYNC_ERROR_HANDLING=3 --env TRITON_CACHE_DIR=/root/.triton/cache \
        --env TORCHINDUCTOR_CACHE_DIR=/root/.cache/vllm/torch_compile_cache/torchinductor \
        --entrypoint /bin/bash "${IMAGE}" -lc \
        'mkdir -p /root/.cache/vllm/torch_compile_cache /root/.cache/vllm/torch_compile_cache/torchinductor /root/.triton/cache; exec nice -n -5 vllm "$@"' \
        vllm-wrapper serve "${MODEL}" --host 127.0.0.1 --port "${PORT}" \
        --served-model-name "${MODEL}" --trust-remote-code --dtype float16 --kv-cache-dtype float16 \
        --tensor-parallel-size 1 "${linear_backend_args[@]}" --max-model-len 100000 \
        --gpu-memory-utilization 0.90 --max-num-seqs 8 --max-num-batched-tokens 32768 \
        --renderer-num-workers 1 --enable-prefix-caching --reasoning-parser qwen3 \
        --default-chat-template-kwargs '{"enable_thinking":false}' \
        --limit-mm-per-prompt '{"image":64,"video":0}' --skip-mm-profiling \
        --mm-processor-kwargs.min_pixels 25088 --mm-processor-kwargs.max_pixels 16777216 \
        --mm-encoder-tp-mode data --mm-tensor-ipc direct_rpc --mm-processor-cache-type shm \
        --mm-processor-cache-gb 16 --mm-shm-cache-max-object-size-mb 512 \
        --enable-chunked-prefill --long-prefill-token-threshold 8192 \
        --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"/root/.cache/vllm/torch-profiler\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":${TORCH_PROFILER_RECORD_SHAPES},\"torch_profiler_use_gzip\":false,\"torch_profiler_dump_cuda_time_total\":true,\"ignore_frontend\":true}"
    wait_for_server
    write_payloads
    curl --fail --silent "${ENDPOINT}/v1/models" | jq . > "${ROOT}/models.json"
}

capture_profile() {
    local label="$1" concurrent="$2" dir payload started ended elapsed pids=() index
    dir="${RESULT_DIR}/${label}"
    payload="$(payload_dir)/text.json"
    mkdir -p "${dir}/responses" "${dir}/traces"
    for index in 1 2; do
        post_checked "${payload}" "${dir}/warmup-${index}.json"
    done
    if (( concurrent > 1 )); then
        for ((index = 1; index <= concurrent; index++)); do
            post_checked "${payload}" "${dir}/warmup-c${concurrent}-${index}.json" &
            pids+=("$!")
        done
        for index in "${pids[@]}"; do wait "${index}"; done
        pids=()
    fi
    # Retain the cache root and clear only the previous trace contents. This
    # keeps phase storage scoped while avoiding a broad recursive removal.
    find "${CACHE_DIR}/torch-profiler" -mindepth 1 -depth -delete
    curl --fail --silent --show-error -X POST "${ENDPOINT}/start_profile" > "${dir}/start-profile.json"
    started="$(date +%s%N)"
    for ((index = 1; index <= concurrent; index++)); do
        post_checked "${payload}" "${dir}/responses/${index}.json" &
        pids+=("$!")
    done
    for index in "${pids[@]}"; do wait "${index}"; done
    ended="$(date +%s%N)"
    curl --fail --silent --show-error -X POST "${ENDPOINT}/stop_profile" > "${dir}/stop-profile.json"
    elapsed="$(awk -v start="${started}" -v end="${ended}" 'BEGIN {printf "%.6f", (end - start) / 1000000000}')"
    jq -n --arg label "${label}" --argjson concurrent "${concurrent}" --argjson elapsed_seconds "${elapsed}" \
        '{label: $label, concurrent: $concurrent, elapsed_seconds: $elapsed_seconds}' > "${dir}/metadata.json"
    sleep 5
    find "${CACHE_DIR}/torch-profiler" -type f -name '*.json' -print0 | \
        xargs -0 -r -I{} cp --parents {} "${dir}/traces"
}

summarize_traces() {
    local root="$1" trace summary
    for trace in $(find "${root}" -type f -name '*.pt.trace.json' | sort); do
        summary="${trace%.json}.top.json"
        jq '
          (.traceEvents // [])
          | map(select((.dur? // 0) > 0) | {name: (.name // "unknown"), cat: (.cat // ""), dur: .dur})
          | group_by(.name)
          | map({name: .[0].name, cat: .[0].cat, total_us: (map(.dur) | add), calls: length})
          | sort_by(-.total_us) | .[:80]
        ' "${trace}" > "${summary}"
    done
}

profile() {
    require_server
    mkdir -p "${RESULT_DIR}"
    capture_profile c1 1
    capture_profile c8 8
    summarize_traces "${RESULT_DIR}"
    curl --fail --silent "${ENDPOINT}/metrics" > "${RESULT_DIR}/metrics-after.prom"
    docker logs "${CONTAINER}" > "${RESULT_DIR}/server.log" 2>&1
}

gates() {
    local dir index
    require_server
    dir="${RESULT_DIR}/gates"
    mkdir -p "${dir}"
    post_checked "$(payload_dir)/text.json" "${dir}/text.json"
    post_checked "$(payload_dir)/image1.json" "${dir}/image1.json"
    post_checked "$(payload_dir)/image2.json" "${dir}/image2.json"
    for index in 1 2 3; do
        post_checked "$(payload_dir)/json.json" "${dir}/json-${index}.json"
        jq -er '.choices[0].message.content | fromjson | select(.ok == true)' "${dir}/json-${index}.json" >/dev/null
    done
    curl --fail --silent "${ENDPOINT}/metrics" > "${dir}/metrics-after.prom"
    docker logs "${CONTAINER}" > "${dir}/server.log" 2>&1
    if rg -n -i 'out of memory|xgrammar|failed to advance fsm|rccl.*fatal|nccl.*fatal|traceback' "${dir}/server.log"; then
        printf 'Fatal log signature found.\n' >&2
        exit 1
    fi
}

stop() {
    docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
}

cleanup() {
    stop
    [[ -d "${CACHE_DIR}" ]] && find "${CACHE_DIR}" -depth -delete
}

run() {
    trap stop EXIT
    start
    profile
    gates
}

case "${ACTION}" in
    start|profile|gates|stop|cleanup|run) "${ACTION}" ;;
    *) usage; exit 2 ;;
esac
