#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run the reversible Phase 136 Qwen3.8 TP4 SplitKV 16-versus-29 A/B.
set -euo pipefail

: "${ALLOW_PRODUCTION_PAUSE:?set to 1 after confirming production is idle}"
: "${BUILD_ROOT:?set the disk2 gfx906 build root}"
: "${PRODUCTION_WORKDIR:?set the selected production Compose directory}"
: "${PRODUCTION_COMPOSE_FILE:?set the selected production Compose file}"
: "${PRODUCTION_ENV_FILE:?set the selected production env file}"
: "${MODEL_DIR:?set the standard Qwen3.8 AWQ model directory}"
: "${FIXTURE:?set a readable 256-square image fixture}"

if [[ "$ALLOW_PRODUCTION_PAUSE" != "1" ]]; then
    echo "ALLOW_PRODUCTION_PAUSE must be 1" >&2
    exit 2
fi

readonly SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
readonly REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
readonly PHASE_ROOT="$BUILD_ROOT/phase-136-qwen38-tp4-splitkv29-rebase"
readonly RESULT_ROOT="$PHASE_ROOT/results/$(date -u +%Y%m%dT%H%M%SZ)"
readonly IMAGE=${IMAGE:-local/vllm-gfx906:v0.28.0-phase136-qwen38-splitkv29}
readonly SERVED_MODEL=qwen38-phase136-splitkv-tp4
readonly CONTROL_PORT=18136
readonly CANDIDATE_PORT=18137

ACTIVE_CONTAINER=
PRODUCTION_RESTORED=0
XMR_WAS_PAUSED=0

mkdir -p "$RESULT_ROOT"

require_file() {
    [[ -r "$1" ]] || { echo "missing readable file: $1" >&2; exit 2; }
}

capture_idle_preflight() {
    require_file "$MODEL_DIR/config.json"
    require_file "$FIXTURE"
    require_file "$PRODUCTION_COMPOSE_FILE"
    require_file "$PRODUCTION_ENV_FILE"
    docker image inspect "$IMAGE" >"$RESULT_ROOT/image-inspect.json"
    curl -fsS --max-time 5 http://127.0.0.1:8002/health \
        >"$RESULT_ROOT/production-health-before.txt"
    curl -fsS --max-time 5 http://127.0.0.1:8002/metrics \
        >"$RESULT_ROOT/production-metrics-before.prom"
    if rg -q 'vllm:num_requests_(running|waiting).* [1-9][0-9]*\\.0' \
        "$RESULT_ROOT/production-metrics-before.prom"; then
        echo "production has active or waiting requests" >&2
        exit 3
    fi
    if systemctl is-active --quiet someai-pexels-video-indexer.service; then
        echo "Pexels indexer is active; refusing all-GPU Phase 136" >&2
        exit 3
    fi
    if [[ -n "${XMR_PID:-}" ]]; then
        if ! kill -0 "$XMR_PID" 2>/dev/null; then
            echo "XMR_PID is not a running process: $XMR_PID" >&2
            exit 3
        fi
        ps -o pid,ppid,user,stat,lstart,cmd -p "$XMR_PID" \
            >"$RESULT_ROOT/xmr-before.txt"
        kill -STOP "$XMR_PID"
        XMR_WAS_PAUSED=1
        ps -o pid,ppid,user,stat,lstart,cmd -p "$XMR_PID" \
            >"$RESULT_ROOT/xmr-paused.txt"
    elif pgrep -af 'xmrig|xmr|monero' >"$RESULT_ROOT/xmr-unmanaged.txt"; then
        echo "XMR is active; rerun with its top-level XMR_PID for reversible pause" >&2
        exit 3
    fi
}

resume_xmr() {
    if [[ "$XMR_WAS_PAUSED" == "1" ]] && kill -0 "$XMR_PID" 2>/dev/null; then
        kill -CONT "$XMR_PID" || true
        ps -o pid,ppid,user,stat,lstart,cmd -p "$XMR_PID" \
            >"$RESULT_ROOT/xmr-restored.txt" || true
        XMR_WAS_PAUSED=0
    fi
}

restore_production() {
    local deadline
    if [[ -n "$ACTIVE_CONTAINER" ]]; then
        docker rm -f "$ACTIVE_CONTAINER" >/dev/null 2>&1 || true
        ACTIVE_CONTAINER=
    fi
    if [[ "$PRODUCTION_RESTORED" != "1" ]]; then
        (
            cd "$PRODUCTION_WORKDIR"
            docker compose --env-file "$PRODUCTION_ENV_FILE" \
                -f "$PRODUCTION_COMPOSE_FILE" up -d
        )
        deadline=$((SECONDS + 1800))
        until curl -fsS --max-time 5 http://127.0.0.1:8002/health \
            >"$RESULT_ROOT/production-health-after.txt"; do
            (( SECONDS < deadline )) || {
                echo "production did not recover within 30 minutes" >&2
                return 1
            }
            sleep 10
        done
        curl -fsS --max-time 5 http://127.0.0.1:8002/v1/models \
            >"$RESULT_ROOT/production-models-after.json"
        curl -fsS --max-time 5 http://127.0.0.1:8002/metrics \
            >"$RESULT_ROOT/production-metrics-after.prom"
        PRODUCTION_RESTORED=1
    fi
    resume_xmr
}

trap restore_production EXIT

wait_for_health() {
    local port=$1
    local variant=$2
    local deadline=$((SECONDS + 1800))
    until curl -fsS --max-time 5 "http://127.0.0.1:${port}/health" \
        >"$RESULT_ROOT/${variant}/health.txt"; do
        if [[ "$(docker inspect -f '{{.State.Running}}' "$ACTIVE_CONTAINER" \
            2>/dev/null || true)" != "true" ]]; then
            docker logs "$ACTIVE_CONTAINER" >"$RESULT_ROOT/${variant}/server.log" 2>&1 || true
            echo "${variant} exited before becoming healthy" >&2
            return 1
        fi
        (( SECONDS < deadline )) || {
            docker logs "$ACTIVE_CONTAINER" >"$RESULT_ROOT/${variant}/server.log" 2>&1 || true
            echo "${variant} did not become healthy within 30 minutes" >&2
            return 1
        }
        sleep 10
    done
}

post_checked() {
    local port=$1
    local name=$2
    local payload=$3
    local directory=$4
    local output="$directory/${name}.json"
    local status
    status="$(curl --silent --show-error --output "$output" --write-out '%{http_code}' \
        --max-time 900 -H 'content-type: application/json' --data-binary "$payload" \
        "http://127.0.0.1:${port}/v1/chat/completions")"
    [[ "$status" == "200" ]]
    jq -e '.choices[0].message.content | strings | length > 0' "$output" >/dev/null
}

post_file_checked() {
    local port=$1
    local name=$2
    local payload_file=$3
    local directory=$4
    local output="$directory/${name}.json"
    local status
    status="$(curl --silent --show-error --output "$output" --write-out '%{http_code}' \
        --max-time 900 -H 'content-type: application/json' --data-binary "@$payload_file" \
        "http://127.0.0.1:${port}/v1/chat/completions")"
    [[ "$status" == "200" ]]
    jq -e '.choices[0].message.content | strings | length > 0' "$output" >/dev/null
}

run_gates() {
    local port=$1
    local directory=$2
    local image_b64 text_payload image1_payload image2_payload json_payload content index
    image_b64="$(base64 -w0 "$FIXTURE")"
    text_payload="$(jq -nc --arg model "$SERVED_MODEL" \
        '{model:$model,temperature:0,max_tokens:32,chat_template_kwargs:{enable_thinking:false},messages:[{role:"user",content:"Reply exactly: phase 136 text smoke passed"}]}')"
    image1_payload="$(jq -nc --arg model "$SERVED_MODEL" --arg b64 "$image_b64" \
        '{model:$model,temperature:0,max_tokens:48,chat_template_kwargs:{enable_thinking:false},messages:[{role:"user",content:[{type:"text",text:"Describe this image in one sentence."},{type:"image_url",image_url:{url:("data:image/png;base64,"+$b64)}}]}]}')"
    image2_payload="$(jq -nc --arg model "$SERVED_MODEL" --arg b64 "$image_b64" \
        '{model:$model,temperature:0,max_tokens:48,chat_template_kwargs:{enable_thinking:false},messages:[{role:"user",content:[{type:"text",text:"Describe these two images in one sentence."},{type:"image_url",image_url:{url:("data:image/png;base64,"+$b64)}},{type:"image_url",image_url:{url:("data:image/png;base64,"+$b64)}}]}]}')"
    post_checked "$port" text "$text_payload" "$directory"
    post_checked "$port" image1 "$image1_payload" "$directory"
    post_checked "$port" image2 "$image2_payload" "$directory"
    for index in 1 2 3; do
        json_payload="$(jq -nc --arg model "$SERVED_MODEL" \
            '{model:$model,temperature:0,max_tokens:32,response_format:{type:"json_object"},chat_template_kwargs:{enable_thinking:false},messages:[{role:"user",content:"Return exactly one JSON object with string key status and value ok."}]}')"
        post_checked "$port" "json-${index}" "$json_payload" "$directory"
        content="$(jq -r '.choices[0].message.content' "$directory/json-${index}.json")"
        jq -e '.status == "ok"' <<<"$content" >/dev/null
    done
}

record_c1() {
    local port=$1
    local directory=$2
    local payload=$3
    local name=$4
    local output="$directory/${name}.json"
    local elapsed tokens rate
    elapsed="$(curl --fail --silent --show-error --output "$output" --write-out '%{time_total}' \
        --max-time 900 -H 'content-type: application/json' --data-binary "$payload" \
        "http://127.0.0.1:${port}/v1/chat/completions")"
    tokens="$(jq -er '.usage.completion_tokens' "$output")"
    [[ "$tokens" == "128" ]]
    rate="$(awk -v tokens="$tokens" -v elapsed="$elapsed" 'BEGIN {printf "%.6f", tokens / elapsed}')"
    jq -n --arg sample "$name" --argjson seconds "$elapsed" \
        --argjson completion_tokens "$tokens" --argjson completion_tok_s "$rate" \
        '{sample:$sample,seconds:$seconds,completion_tokens:$completion_tokens,completion_tok_s:$completion_tok_s}' \
        >>"$directory/c1.jsonl"
}

record_c8() {
    local port=$1
    local directory=$2
    local payload=$3
    local batch=$4
    local start elapsed total=0 index tokens
    local pids=()
    start="$(date +%s.%N)"
    for index in $(seq 1 8); do
        curl --fail --silent --show-error --max-time 900 -H 'content-type: application/json' \
            --data-binary "$payload" "http://127.0.0.1:${port}/v1/chat/completions" \
            >"$directory/c8_${batch}_${index}.json" &
        pids+=("$!")
    done
    for index in "${pids[@]}"; do
        wait "$index"
    done
    elapsed="$(awk -v start="$start" -v end="$(date +%s.%N)" 'BEGIN {printf "%.6f", end - start}')"
    for index in $(seq 1 8); do
        tokens="$(jq -er '.usage.completion_tokens' "$directory/c8_${batch}_${index}.json")"
        [[ "$tokens" == "128" ]]
        total=$((total + tokens))
    done
    jq -n --argjson batch "$batch" --argjson seconds "$elapsed" \
        --argjson completion_tokens "$total" \
        --argjson aggregate_completion_tok_s "$(awk -v t="$total" -v s="$elapsed" 'BEGIN {printf "%.6f", t / s}')" \
        '{batch:$batch,seconds:$seconds,completion_tokens:$completion_tokens,aggregate_completion_tok_s:$aggregate_completion_tok_s}' \
        >>"$directory/c8.jsonl"
}

run_benchmarks() {
    local port=$1
    local directory=$2
    local payload context prime index
    payload="$(jq -nc --arg model "$SERVED_MODEL" \
        '{model:$model,temperature:0,min_tokens:128,max_tokens:128,chat_template_kwargs:{enable_thinking:false},messages:[{role:"user",content:"Write exactly 128 concise tokens about reliable GPU inference."}]}')"
    record_c1 "$port" "$directory" "$payload" warmup
    for index in 1 2 3; do
        record_c1 "$port" "$directory" "$payload" "c1_${index}"
        record_c8 "$port" "$directory" "$payload" "$index"
    done
    jq -s 'map(select(.sample | startswith("c1_")) | .completion_tok_s) | sort | {samples_tok_s:.,median_tok_s:.[length / 2 | floor]}' \
        "$directory/c1.jsonl" >"$directory/c1-summary.json"
    jq -s '{samples:.,median_tok_s:(map(.aggregate_completion_tok_s)|sort|.[length / 2 | floor])}' \
        "$directory/c8.jsonl" >"$directory/c8-summary.json"

    context="$directory/long32k-context.txt"
    awk 'BEGIN { for (i = 0; i < 32768; ++i) printf " trace" }' >"$context"
    prime="$directory/long-prefix-prime.request.json"
    jq -nc --arg model "$SERVED_MODEL" --rawfile text "$context" \
        '{model:$model,temperature:0,max_tokens:1,chat_template_kwargs:{enable_thinking:false},messages:[{role:"user",content:$text}]}' >"$prime"
    post_file_checked "$port" long-prefix-prime "$prime" "$directory"
    jq -nc --arg model "$SERVED_MODEL" --rawfile text "$context" \
        '{model:$model,temperature:0,min_tokens:128,max_tokens:128,chat_template_kwargs:{enable_thinking:false},messages:[{role:"user",content:$text}]}' >"$directory/long32k.request.json"
    for index in 1 2 3; do
        record_c1 "$port" "$directory" "@$directory/long32k.request.json" "long32k_${index}"
        mv "$directory/long32k_${index}.json" "$directory/long32k-${index}.json"
    done
    jq -s 'map(select(.sample | startswith("long32k_")) | {seconds,completion_tokens,completion_tok_s}) | {samples:.,median_tok_s:(map(.completion_tok_s)|sort|.[length / 2 | floor])}' \
        "$directory/c1.jsonl" >"$directory/long32k-summary.json"
}

scan_variant() {
    local port=$1
    local directory=$2
    curl -fsS --max-time 5 "http://127.0.0.1:${port}/metrics" >"$directory/metrics.prom"
    docker logs "$ACTIVE_CONTAINER" >"$directory/server.log" 2>&1
    if rg -ni 'oom|out of memory|traceback|xgrammar|failed to advance fsm|rccl.*fatal|nccl.*fatal|ras event|illegal instruction' \
        "$directory/server.log" >"$directory/error-scan.txt"; then
        echo "fatal log signature in $(basename "$directory")" >&2
        return 1
    fi
    if rg -q 'vllm:num_requests_(running|waiting).* [1-9][0-9]*\\.0' "$directory/metrics.prom"; then
        echo "request queue did not drain for $(basename "$directory")" >&2
        return 1
    fi
}

run_variant() {
    local variant=$1
    local port=$2
    local max_splits=$3
    local forced_splits=$4
    local directory="$RESULT_ROOT/$variant"
    local cache_dir="$PHASE_ROOT/cache/$variant"
    local container="vllm-gfx906-phase136-$variant"
    local started elapsed
    mkdir -p "$directory" "$cache_dir/triton-cache"
    ACTIVE_CONTAINER="$container"
    started="$(date +%s%N)"
    docker run -d --name "$container" --network host --ipc host --shm-size 64g \
        --device /dev/kfd --device /dev/dri --group-add video \
        -e HIP_VISIBLE_DEVICES=0,1,2,3 \
        -e PYTORCH_ROCM_ARCH=gfx906 -e ROCM_ARCH=gfx906 -e ROCM_PATH=/opt/rocm \
        -e USE_ROCM=1 -e VLLM_TARGET_DEVICE=rocm -e VLLM_CACHE_ROOT=/root/.cache/vllm \
        -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 -e TORCH_NCCL_ASYNC_ERROR_HANDLING=3 \
        -e OMP_NUM_THREADS=12 -e OPENBLAS_NUM_THREADS=12 -e MKL_NUM_THREADS=12 \
        -e NUMEXPR_NUM_THREADS=12 -e TOKENIZERS_PARALLELISM=false \
        -e VLLM_ROCM_ENABLE_GFX906_SPLITKV=1 -e VLLM_ROCM_GFX906_SPLITKV_DEBUG=1 \
        -e VLLM_ROCM_GFX906_SPLITKV_QUERY_ROWS=8 \
        -e VLLM_ROCM_GFX906_SPLITKV_MAX_SPLITS="$max_splits" \
        -e VLLM_ROCM_GFX906_SPLITKV_FORCE_SPLITS="$forced_splits" \
        -e VLLM_ROCM_USE_AITER=0 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
        -e TRITON_CACHE_DIR=/root/.triton/cache \
        -v "$MODEL_DIR:/model:ro" -v "$cache_dir:/root/.cache/vllm" \
        -v "$cache_dir/triton-cache:/root/.triton/cache" \
        --entrypoint /bin/bash "$IMAGE" -lc \
        "exec /opt/vllm-venv/bin/vllm serve /model --host 127.0.0.1 --port $port \\
          --served-model-name $SERVED_MODEL --tensor-parallel-size 4 --dtype float16 \\
          --trust-remote-code --linear-backend gfx906_gptq --max-model-len 100000 \\
          --gpu-memory-utilization 0.88 --max-num-seqs 8 --max-num-batched-tokens 8192 \\
          --limit-mm-per-prompt '{\"image\":64,\"video\":0}' \\
          --mm-processor-kwargs '{\"max_pixels\":16777216}' \\
          --mm-processor-cache-type shm --mm-processor-cache-gb 16 \\
          --mm-shm-cache-max-object-size-mb 512 --mm-tensor-ipc direct_rpc \\
          --mm-encoder-tp-mode data --renderer-num-workers 1 --enable-prefix-caching \\
          --enable-chunked-prefill --mamba-cache-mode align --skip-mm-profiling \\
          --reasoning-parser qwen3 --default-chat-template-kwargs '{\"enable_thinking\":false}'" \
        >"$directory/container-id.txt"
    wait_for_health "$port" "$variant"
    elapsed="$(awk -v start="$started" -v end="$(date +%s%N)" 'BEGIN {printf "%.3f", (end-start)/1000000000}')"
    jq -n --arg variant "$variant" --arg image "$IMAGE" --arg model_dir "$MODEL_DIR" \
        --argjson max_splits "$max_splits" --argjson forced_splits "$forced_splits" \
        --argjson startup_seconds "$elapsed" \
        '{variant:$variant,image:$image,model_dir:$model_dir,tensor_parallel_size:4,max_model_len:100000,max_splits:$max_splits,forced_splits:$forced_splits,startup_seconds:$startup_seconds}' \
        >"$directory/runtime.json"
    run_gates "$port" "$directory"
    run_benchmarks "$port" "$directory"
    scan_variant "$port" "$directory"
    docker rm -f "$container" >/dev/null
    ACTIVE_CONTAINER=
}

capture_idle_preflight
docker build -f "$REPO_ROOT/docker/Dockerfile.gfx906-v028-phase136-qwen38-splitkv29" \
    -t "$IMAGE" "$REPO_ROOT" >"$RESULT_ROOT/image-build.log" 2>&1
(
    cd "$PRODUCTION_WORKDIR"
    docker compose --env-file "$PRODUCTION_ENV_FILE" -f "$PRODUCTION_COMPOSE_FILE" down
)
run_variant control "$CONTROL_PORT" 16 16
run_variant candidate "$CANDIDATE_PORT" 32 29
restore_production
trap - EXIT
printf '%s\\n' "$RESULT_ROOT"
