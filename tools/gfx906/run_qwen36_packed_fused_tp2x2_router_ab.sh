#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Compare packed Qwen3.6 TP2x2 Router control with its fused QK/RoPE overlay.
set -euo pipefail

: "${ALLOW_PRODUCTION_PAUSE:?set to 1 after confirming production is idle}"
: "${BUILD_ROOT:?set the disk2 gfx906 build root}"
: "${PRODUCTION_WORKDIR:?set the selected production Compose directory}"
: "${PRODUCTION_COMPOSE_FILE:?set the selected production Compose file}"
: "${PRODUCTION_ENV_FILE:?set the selected production env file}"
: "${FIXTURE:?set a readable image fixture}"

[[ "$ALLOW_PRODUCTION_PAUSE" == "1" ]] || { echo "ALLOW_PRODUCTION_PAUSE must be 1" >&2; exit 2; }

readonly SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
readonly REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
readonly PHASE_SLUG=phase-151-qwen36-packed-fused-tp2x2-router
readonly PHASE_ROOT="${PHASE_ROOT:-$BUILD_ROOT/$PHASE_SLUG}"
readonly RESULT_ROOT="$PHASE_ROOT/results/$(date -u +%Y%m%dT%H%M%SZ)"
readonly CONTROL_IMAGE="${CONTROL_IMAGE:-local/vllm-gfx906:v0.28.0-phase142-qwen-gdn-output-norm}"
readonly CANDIDATE_IMAGE="${CANDIDATE_IMAGE:-local/vllm-gfx906:v0.28.0-phase151-qwen36-packed-fused-tp2x2}"
readonly ROUTER_IMAGE="${ROUTER_IMAGE:-vllm/vllm-router:nightly-20260710-b93cbcb}"
readonly WHEEL_DIR="$BUILD_ROOT/phase-129-triton36-scf-pointer/wheel"
readonly WHEEL="$WHEEL_DIR/triton-3.6.0-cp312-cp312-linux_x86_64.whl"
readonly STANDARD_MODEL_DIR="${STANDARD_MODEL_DIR:-$BUILD_ROOT/phase-91-qwen36-tp4-mtp-parity/models/qwen36-awq-int4}"
readonly PACKED_MODEL_DIR="${PACKED_MODEL_DIR:-$BUILD_ROOT/phase-125-v028-qwen36-packed-int8-tp4/models/qwen36-embed-lmhead-int8}"
readonly FIXTURE_DIR=$(dirname -- "$FIXTURE")
readonly FIXTURE_NAME=$(basename -- "$FIXTURE")

declare -a ACTIVE_CONTAINERS=()
declare -a ACTIVE_NETWORKS=()
PRODUCTION_RESTORED=0

mkdir -p "$RESULT_ROOT"

require_file() { [[ -r "$1" ]] || { echo "missing readable file: $1" >&2; exit 2; }; }

capture_preflight() {
    require_file "$FIXTURE"
    require_file "$WHEEL"
    require_file "$PRODUCTION_COMPOSE_FILE"
    require_file "$PRODUCTION_ENV_FILE"
    require_file "$STANDARD_MODEL_DIR/config.json"
    require_file "$PACKED_MODEL_DIR/config.json"
    require_file "$PACKED_MODEL_DIR/model-00003-of-00004.safetensors"
    require_file "$PACKED_MODEL_DIR/model-00004-of-00004.safetensors"
    docker image inspect "$CONTROL_IMAGE" >"$RESULT_ROOT/control-image-inspect.json"
    docker image inspect "$ROUTER_IMAGE" >"$RESULT_ROOT/router-image-inspect.json"
    curl -fsS --max-time 5 http://127.0.0.1:8002/health >"$RESULT_ROOT/production-health-before.txt"
    curl -fsS --max-time 5 http://127.0.0.1:8002/v1/models >"$RESULT_ROOT/production-models-before.json"
    curl -fsS --max-time 5 http://127.0.0.1:8002/metrics >"$RESULT_ROOT/production-metrics-before.prom"
    if rg -q 'vllm:num_requests_(running|waiting).* [1-9][0-9]*\.0' "$RESULT_ROOT/production-metrics-before.prom"; then
        echo "production has active or waiting requests" >&2
        exit 3
    fi
    if systemctl is-active --quiet someai-pexels-video-indexer.service; then
        echo "Pexels indexer is active; refusing all-GPU phase" >&2
        exit 3
    fi
    if pgrep -af '[x]mrig|[x]mr|[m]onero' >"$RESULT_ROOT/xmr-unmanaged.txt"; then
        echo "XMR is active; refusing exclusive benchmark" >&2
        exit 3
    fi
    for port in 18160 18161; do
        if ss -ltn "sport = :$port" | tail -n +2 | grep -q .; then
            echo "temporary port is already occupied: $port" >&2
            exit 3
        fi
    done
    docker ps --format '{{.Names}} {{.Image}} {{.Status}}' >"$RESULT_ROOT/docker-before.txt"
    df -h /mnt/disk1 /mnt/disk2 / >"$RESULT_ROOT/disk-before.txt"
}

cleanup_variant() {
    local name
    for name in "${ACTIVE_CONTAINERS[@]}"; do
        docker logs "$name" >"$RESULT_ROOT/${name}.log" 2>&1 || true
        docker rm -f "$name" >/dev/null 2>&1 || true
    done
    ACTIVE_CONTAINERS=()
    for name in "${ACTIVE_NETWORKS[@]}"; do
        docker network rm "$name" >/dev/null 2>&1 || true
    done
    ACTIVE_NETWORKS=()
}

restore_production() {
    local deadline
    cleanup_variant
    if [[ "$PRODUCTION_RESTORED" == "1" ]]; then
        return
    fi
    (
        cd "$PRODUCTION_WORKDIR"
        docker compose --env-file "$PRODUCTION_ENV_FILE" -f "$PRODUCTION_COMPOSE_FILE" up -d
    )
    deadline=$((SECONDS + 1800))
    until curl -fsS --max-time 5 http://127.0.0.1:8002/health >"$RESULT_ROOT/production-health-after.txt" 2>/dev/null && \
        curl -fsS --max-time 5 http://127.0.0.1:8002/v1/models >"$RESULT_ROOT/production-models-after.json" 2>/dev/null; do
        (( SECONDS < deadline )) || { echo "production did not recover within 30 minutes" >&2; return 1; }
        sleep 10
    done
    curl -fsS --max-time 5 http://127.0.0.1:8002/metrics >"$RESULT_ROOT/production-metrics-after.prom"
    PRODUCTION_RESTORED=1
}

trap restore_production EXIT

wait_worker_health() {
    local name=$1 deadline=$((SECONDS + 1800))
    while (( SECONDS < deadline )); do
        if docker exec "$name" python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()" >/dev/null 2>&1; then
            return 0
        fi
        if ! docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null | grep -qx true; then
            docker logs "$name" >"$RESULT_ROOT/$name-startup.log" 2>&1 || true
            echo "$name exited before health" >&2
            return 1
        fi
        sleep 10
    done
    docker logs "$name" >"$RESULT_ROOT/$name-startup.log" 2>&1 || true
    echo "$name did not become healthy within 30 minutes" >&2
    return 1
}

wait_router_health() {
    local port=$1 name=$2 deadline=$((SECONDS + 1800))
    while (( SECONDS < deadline )); do
        if curl -fsS --max-time 5 "http://127.0.0.1:${port}/health" >"$RESULT_ROOT/$name-health.txt"; then
            return 0
        fi
        if ! docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null | grep -qx true; then
            docker logs "$name" >"$RESULT_ROOT/$name-startup.log" 2>&1 || true
            echo "$name exited before health" >&2
            return 1
        fi
        sleep 10
    done
    docker logs "$name" >"$RESULT_ROOT/$name-startup.log" 2>&1 || true
    echo "$name did not become healthy within 30 minutes" >&2
    return 1
}

server_command() {
    cat <<EOF
exec /opt/vllm-venv/bin/vllm serve /model --host 0.0.0.0 --port 8000 \\
  --served-model-name "$SERVED_MODEL" --tensor-parallel-size 2 --dtype float16 \\
  --kv-cache-dtype float16 --trust-remote-code --linear-backend gfx906_gptq \\
  --max-model-len 100000 --gpu-memory-utilization 0.88 --max-num-seqs 8 \\
  --max-num-batched-tokens 8192 --limit-mm-per-prompt '{"image":64,"video":0}' \\
  --mm-processor-kwargs '{"max_pixels":16777216}' --mm-processor-cache-type shm \\
  --mm-processor-cache-gb 16 --mm-shm-cache-max-object-size-mb 512 \\
  --mm-tensor-ipc direct_rpc --mm-encoder-tp-mode data --renderer-num-workers 1 \\
  --enable-prefix-caching --enable-chunked-prefill --mamba-cache-mode align \\
  --skip-mm-profiling --reasoning-parser qwen3 \\
  --default-chat-template-kwargs '{"enable_thinking":false}'
EOF
}

run_worker() {
    local name=$1 devices=$2 network=$3 alias=$4 image=$5 cache_dir=$6
    mkdir -p "$cache_dir/triton-cache"
    docker run -d --name "$name" --network "$network" --network-alias "$alias" --ipc host --shm-size 64g \
        --device /dev/kfd --device /dev/dri --group-add video \
        -e HIP_VISIBLE_DEVICES="$devices" -e PYTORCH_ROCM_ARCH=gfx906 -e ROCM_ARCH=gfx906 \
        -e ROCM_PATH=/opt/rocm -e USE_ROCM=1 -e VLLM_TARGET_DEVICE=rocm \
        -e VLLM_CACHE_ROOT=/root/.cache/vllm -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
        -e TORCH_NCCL_ASYNC_ERROR_HANDLING=3 -e OMP_NUM_THREADS=12 -e OPENBLAS_NUM_THREADS=12 \
        -e MKL_NUM_THREADS=12 -e NUMEXPR_NUM_THREADS=12 -e TOKENIZERS_PARALLELISM=false \
        -e VLLM_ROCM_ENABLE_GFX906_SPLITKV=1 -e VLLM_ROCM_GFX906_SPLITKV_DEBUG=0 \
        -e VLLM_ROCM_GFX906_SPLITKV_QUERY_ROWS=8 -e VLLM_ROCM_GFX906_SPLITKV_MAX_SPLITS=16 \
        -e VLLM_ROCM_GFX906_SPLITKV_FORCE_SPLITS=16 -e VLLM_ROCM_USE_AITER=0 \
        -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e TRITON_CACHE_DIR=/root/.triton/cache \
        -v "$PACKED_MODEL_DIR:/model:ro" -v "$STANDARD_MODEL_DIR:/source:ro" \
        -v "$cache_dir:/root/.cache/vllm" -v "$cache_dir/triton-cache:/root/.triton/cache" \
        --entrypoint /bin/bash "$image" -lc "$(server_command)" >"$RESULT_ROOT/$name-container-id.txt"
    ACTIVE_CONTAINERS+=("$name")
}

scan_variant() {
    local label=$1 port=$2 directory=$3
    local name
    curl -fsS --max-time 5 "http://127.0.0.1:${port}/metrics" >"$directory/router-metrics.prom" || true
    for name in "${ACTIVE_CONTAINERS[@]}"; do
        docker logs "$name" >"$directory/${name}.server.log" 2>&1 || true
        if [[ "$name" == *worker* ]]; then
            docker exec "$name" python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/metrics').read().decode(), end='')" >"$directory/${name}.metrics.prom" || true
        fi
    done
    if rg -ni 'oom|out of memory|traceback|xgrammar|failed to advance fsm|rccl.*fatal|nccl.*fatal|ras event|illegal instruction' "$directory"/*.log "$directory"/*.server.log >"$directory/fatal-signatures.txt"; then
        echo "fatal log signature in $label" >&2
        return 1
    fi
    if rg -q 'vllm:num_requests_(running|waiting).* [1-9][0-9]*\.0' "$directory"/*.metrics.prom "$directory/router-metrics.prom"; then
        echo "request queue did not drain for $label" >&2
        return 1
    fi
}

run_variant() {
    local key=$1 image=$2 port=$3
    local base="phase151-${key}-tp2x2" directory="$RESULT_ROOT/$key/tp2x2-router"
    local network="${base}-net" worker0="${base}-worker0" worker1="${base}-worker1" router="${base}-router"
    local count0 count1
    SERVED_MODEL="qwen36-${key}-phase151"
    export SERVED_MODEL
    mkdir -p "$directory"
    docker image inspect "$image" >"$directory/image-inspect.json"
    docker network create "$network" >"$directory/network-id.txt"
    ACTIVE_NETWORKS+=("$network")
    run_worker "$worker0" 0,1 "$network" worker0 "$image" "$PHASE_ROOT/cache/$key/worker0"
    run_worker "$worker1" 2,3 "$network" worker1 "$image" "$PHASE_ROOT/cache/$key/worker1"
    wait_worker_health "$worker0"
    wait_worker_health "$worker1"
    docker run -d --name "$router" --network "$network" -p "127.0.0.1:${port}:8000" "$ROUTER_IMAGE" \
        vllm-router --host 0.0.0.0 --port 8000 --worker-urls http://worker0:8000 http://worker1:8000 \
        --policy round_robin --worker-startup-timeout-secs 1800 --worker-startup-check-interval 5 \
        --max-payload-size 268435456 --max-concurrent-requests 32 --request-timeout-secs 900 \
        --retry-max-retries 1 --health-check-interval-secs 10 --health-check-timeout-secs 5 \
        --prometheus-host 0.0.0.0 --prometheus-port 29000 >"$directory/router-container-id.txt"
    ACTIVE_CONTAINERS+=("$router")
    wait_router_health "$port" "$router"
    docker run --rm --network host -v "$SCRIPT_DIR:/tools:ro" -v "$FIXTURE_DIR:/fixture:ro" -v "$directory:/results" \
        --entrypoint /opt/vllm-venv/bin/python "$image" /tools/benchmark_qwen36_packed_tp2x2.py \
        --endpoint "http://127.0.0.1:${port}/v1" --model "$SERVED_MODEL" --fixture "/fixture/$FIXTURE_NAME" \
        --asset-dir /results/assets --output /results/result.json --rounds 3 --timeout 900 >"$directory/benchmark-summary.json"
    docker logs "$worker0" >"$directory/${worker0}-precleanup.log" 2>&1 || true
    docker logs "$worker1" >"$directory/${worker1}-precleanup.log" 2>&1 || true
    count0=$(rg -c --no-filename 'POST /v1/chat/completions HTTP/1\.1" 200' "$directory/${worker0}-precleanup.log" || true)
    count1=$(rg -c --no-filename 'POST /v1/chat/completions HTTP/1\.1" 200' "$directory/${worker1}-precleanup.log" || true)
    count0=${count0:-0}
    count1=${count1:-0}
    jq -n --argjson worker0 "$count0" --argjson worker1 "$count1" \
        '{worker0_chat_200:$worker0,worker1_chat_200:$worker1,difference:(($worker0-$worker1)|if . < 0 then -. else . end)}' >"$directory/worker-distribution.json"
    if (( count0 < 10 || count1 < 10 || (count0 - count1 > 4) || (count1 - count0 > 4) )); then
        echo "Router did not distribute material traffic: worker0=$count0 worker1=$count1" >&2
        return 1
    fi
    scan_variant "$key" "$port" "$directory"
    cleanup_variant
}

cleanup_cache() {
    local cache_root="$PHASE_ROOT/cache"
    [[ "$cache_root" == "$BUILD_ROOT/$PHASE_SLUG/cache" ]] || return 0
    [[ -d "$cache_root" ]] || return 0
    find "$cache_root" -depth -delete
}

capture_preflight
docker build --build-context "triton-wheel=$WHEEL_DIR" --build-arg "BASE_IMAGE=$CONTROL_IMAGE" \
    -f "$REPO_ROOT/docker/Dockerfile.gfx906-v028-phase151-qwen36-fused" \
    -t "$CANDIDATE_IMAGE" "$REPO_ROOT" >"$RESULT_ROOT/candidate-image-build.log" 2>&1
docker image inspect "$CANDIDATE_IMAGE" >"$RESULT_ROOT/candidate-image-inspect.json"
(
    cd "$PRODUCTION_WORKDIR"
    docker compose --env-file "$PRODUCTION_ENV_FILE" -f "$PRODUCTION_COMPOSE_FILE" down
)
run_variant packed-control "$CONTROL_IMAGE" 18160
run_variant packed-fused "$CANDIDATE_IMAGE" 18161
restore_production
trap - EXIT
cleanup_cache
df -h /mnt/disk1 /mnt/disk2 / >"$RESULT_ROOT/disk-after.txt"
printf '%s\n' "$RESULT_ROOT"
