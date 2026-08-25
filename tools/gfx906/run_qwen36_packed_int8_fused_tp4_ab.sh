#!/usr/bin/env bash
# Run a reversible TP4 A/B: Phase-125 packed Qwen3.6 versus the fused overlay.
set -euo pipefail

: "${ALLOW_PRODUCTION_PAUSE:?set to 1 after an idle production check}"
: "${BUILD_ROOT:?set the disk2 vllm-gfx906 build root}"
: "${PRODUCTION_WORKDIR:?set the selected production Compose directory}"
: "${PRODUCTION_COMPOSE_FILE:?set the selected production Compose file}"
: "${PRODUCTION_ENV_FILE:?set the selected production env file}"

if [[ "$ALLOW_PRODUCTION_PAUSE" != "1" ]]; then
    echo "ALLOW_PRODUCTION_PAUSE must be 1" >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
PHASE_ROOT="$BUILD_ROOT/phase-135-qwen36-packed-int8-fused-tp4"
RESULT_ROOT="$PHASE_ROOT/results/$(date -u +%Y%m%dT%H%M%SZ)"
WHEEL_DIR="$BUILD_ROOT/phase-129-triton36-scf-pointer/wheel"
WHEEL="$WHEEL_DIR/triton-3.6.0-cp312-cp312-linux_x86_64.whl"
MODEL_DIR="$BUILD_ROOT/phase-125-v028-qwen36-packed-int8-tp4/models/qwen36-embed-lmhead-int8"
SOURCE_MODEL_DIR="$BUILD_ROOT/phase-91-qwen36-tp4-mtp-parity/models/qwen36-awq-int4"
BASE_IMAGE=${BASE_IMAGE:-local/vllm-gfx906:v0.28.0-phase123-qwen38-int8-semantic}
CANDIDATE_IMAGE=${CANDIDATE_IMAGE:-local/vllm-gfx906:v0.28.0-phase135-qwen36-packed-fused}
MODEL_ID=qwen36-phase135-packed-int8-tp4
CONTROL_PORT=18135
CANDIDATE_PORT=18136
PRODUCTION_RESTORED=0
ACTIVE_CONTAINER=

mkdir -p "$RESULT_ROOT"

require_file() {
    [[ -f "$1" ]] || { echo "missing file: $1" >&2; exit 2; }
}

require_file "$WHEEL"
require_file "$MODEL_DIR/config.json"
require_file "$SOURCE_MODEL_DIR/config.json"
require_file "$PRODUCTION_COMPOSE_FILE"
require_file "$PRODUCTION_ENV_FILE"

capture_idle_preflight() {
    curl -fsS --max-time 5 http://127.0.0.1:8002/health \
        >"$RESULT_ROOT/production-health-before.txt"
    curl -fsS --max-time 5 http://127.0.0.1:8002/metrics \
        >"$RESULT_ROOT/production-metrics-before.prom"
    if rg -q 'vllm:num_requests_(running|waiting).* [1-9][0-9]*\.0' \
        "$RESULT_ROOT/production-metrics-before.prom"; then
        echo "production is not idle" >&2
        exit 3
    fi
    if pgrep -af 'xmrig|xmr|monero' >"$RESULT_ROOT/xmr-preflight.txt"; then
        echo "unmanaged XMR process is active" >&2
        exit 3
    fi
    if docker ps -a --format '{{.Names}}\t{{.Status}}\t{{.Image}}' \
        | rg -i 'xmrig|xmr|monero' >"$RESULT_ROOT/xmr-containers-preflight.txt"; then
        echo "XMR container is active or stopped but not accounted for" >&2
        exit 3
    fi
    if systemctl list-units --all --type=service --no-legend \
        | rg -i 'xmrig|xmr|monero' >"$RESULT_ROOT/xmr-units-preflight.txt"; then
        echo "XMR service is present but not accounted for" >&2
        exit 3
    fi
}

restore_production() {
    if [[ "$PRODUCTION_RESTORED" == "1" ]]; then
        return
    fi
    if [[ -n "$ACTIVE_CONTAINER" ]]; then
        docker rm -f "$ACTIVE_CONTAINER" >/dev/null 2>&1 || true
        ACTIVE_CONTAINER=
    fi
    (
        cd "$PRODUCTION_WORKDIR"
        docker compose --env-file "$PRODUCTION_ENV_FILE" \
            -f "$PRODUCTION_COMPOSE_FILE" up -d
    )
    local deadline=$((SECONDS + 1800))
    until curl -fsS --max-time 5 http://127.0.0.1:8002/health \
        >"$RESULT_ROOT/production-health-after.txt"; do
        (( SECONDS < deadline )) || {
            echo "production failed to recover" >&2
            return 1
        }
        sleep 10
    done
    curl -fsS --max-time 5 http://127.0.0.1:8002/v1/models \
        >"$RESULT_ROOT/production-models-after.json"
    curl -fsS --max-time 5 http://127.0.0.1:8002/metrics \
        >"$RESULT_ROOT/production-metrics-after.prom"
    PRODUCTION_RESTORED=1
}

trap restore_production EXIT

wait_for_health() {
    local port=$1
    local deadline=$((SECONDS + 1800))
    until curl -fsS --max-time 5 "http://127.0.0.1:${port}/health" \
        >"$RESULT_ROOT/${ACTIVE_CONTAINER}-health.txt"; do
        if [[ "$(docker inspect -f '{{.State.Running}}' "$ACTIVE_CONTAINER" 2>/dev/null || true)" != "true" ]]; then
            docker logs "$ACTIVE_CONTAINER" >"$RESULT_ROOT/${ACTIVE_CONTAINER}-startup.log" 2>&1 || true
            docker inspect "$ACTIVE_CONTAINER" >"$RESULT_ROOT/${ACTIVE_CONTAINER}-inspect.json" 2>&1 || true
            echo "candidate exited before becoming healthy" >&2
            return 1
        fi
        (( SECONDS < deadline )) || {
            docker logs "$ACTIVE_CONTAINER" >"$RESULT_ROOT/${ACTIVE_CONTAINER}-startup.log" 2>&1 || true
            echo "candidate did not become healthy" >&2
            return 1
        }
        sleep 10
    done
}

run_variant() {
    local variant=$1
    local image=$2
    local port=$3
    local variant_dir="$RESULT_ROOT/$variant"
    local cache_dir="$PHASE_ROOT/cache/$variant"
    local triton_dir="$cache_dir/triton-cache"
    local container="vllm-gfx906-phase135-$variant"
    mkdir -p "$variant_dir" "$cache_dir" "$triton_dir"
    ACTIVE_CONTAINER=$container

    docker run -d --name "$container" --network host --ipc host \
        --shm-size 64g --device /dev/kfd --device /dev/dri --group-add video \
        -e HIP_VISIBLE_DEVICES=0,1,2,3 \
        -e PYTORCH_ROCM_ARCH=gfx906 -e ROCM_ARCH=gfx906 -e ROCM_PATH=/opt/rocm \
        -e USE_ROCM=1 -e VLLM_TARGET_DEVICE=rocm \
        -e OMP_NUM_THREADS=12 -e OPENBLAS_NUM_THREADS=12 -e MKL_NUM_THREADS=12 \
        -e NUMEXPR_NUM_THREADS=12 -e TOKENIZERS_PARALLELISM=false \
        -e TORCH_NCCL_ASYNC_ERROR_HANDLING=3 -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
        -e VLLM_ROCM_ENABLE_GFX906_SPLITKV=1 -e VLLM_ROCM_GFX906_SPLITKV_DEBUG=0 \
        -e VLLM_ROCM_GFX906_SPLITKV_QUERY_ROWS=8 -e VLLM_ROCM_USE_AITER=0 \
        -e TRITON_CACHE_DIR=/root/.triton/cache -e VLLM_CACHE_ROOT=/root/.cache/vllm \
        -v "$MODEL_DIR:/model:ro" -v "$SOURCE_MODEL_DIR:/source:ro" \
        -v "$cache_dir:/root/.cache/vllm" \
        -v "$triton_dir:/root/.triton/cache" --entrypoint /bin/bash "$image" \
        -lc "exec /opt/vllm-venv/bin/vllm serve /model \\
          --host 127.0.0.1 --port $port --served-model-name $MODEL_ID \\
          --tensor-parallel-size 4 --dtype float16 --trust-remote-code \\
          --linear-backend gfx906_gptq --max-model-len 100000 \\
          --gpu-memory-utilization 0.88 --max-num-seqs 8 --max-num-batched-tokens 8192 \\
          --limit-mm-per-prompt '{\"image\":64,\"video\":0}' \\
          --mm-processor-kwargs '{\"max_pixels\":16777216}' \\
          --mm-processor-cache-type shm --mm-processor-cache-gb 16 \\
          --mm-shm-cache-max-object-size-mb 512 --mm-tensor-ipc direct_rpc \\
          --mm-encoder-tp-mode data --renderer-num-workers 1 \\
          --enable-prefix-caching --enable-chunked-prefill --mamba-cache-mode align \\
          --skip-mm-profiling --reasoning-parser qwen3 \\
          --default-chat-template-kwargs '{\"enable_thinking\":false}'"
    wait_for_health "$port"
    docker run --rm --network host -v "$SCRIPT_DIR:/tools:ro" \
        --entrypoint /opt/vllm-venv/bin/python "$image" \
        /tools/smoke_qwen36_fused_tp4.py --base-url "http://127.0.0.1:$port" \
        --model "$MODEL_ID" >"$variant_dir/smoke.log" 2>&1
    docker run --rm --network host -v "$SCRIPT_DIR:/tools:ro" \
        --entrypoint /opt/vllm-venv/bin/python "$image" \
        /tools/benchmark_qwen36_fused_tp4.py --base-url "http://127.0.0.1:$port" \
        --model "$MODEL_ID" --concurrency 1 --warmup >"$variant_dir/warmup.json"
    for concurrency in 1 8; do
        for repetition in 1 2 3; do
            docker run --rm --network host -v "$SCRIPT_DIR:/tools:ro" \
                --entrypoint /opt/vllm-venv/bin/python "$image" \
                /tools/benchmark_qwen36_fused_tp4.py \
                --base-url "http://127.0.0.1:$port" --model "$MODEL_ID" \
                --concurrency "$concurrency" >"$variant_dir/c${concurrency}-${repetition}.json"
        done
    done
    curl -fsS --max-time 5 "http://127.0.0.1:$port/metrics" >"$variant_dir/metrics.prom"
    docker logs "$container" >"$variant_dir/server.log" 2>&1
    if rg -ni 'oom|traceback|xgrammar|failed to advance fsm|rccl.*fatal|nccl.*fatal' \
        "$variant_dir/server.log" >"$variant_dir/error-scan.txt"; then
        echo "fatal log signature in $variant" >&2
        return 1
    fi
    if rg -q 'vllm:num_requests_(running|waiting).* [1-9][0-9]*\.0' \
        "$variant_dir/metrics.prom"; then
        echo "queue did not drain for $variant" >&2
        return 1
    fi
    docker rm -f "$container" >/dev/null
    ACTIVE_CONTAINER=
}

capture_idle_preflight
docker build --build-context "triton-wheel=$WHEEL_DIR" --build-arg "BASE_IMAGE=$BASE_IMAGE" \
    -f "$REPO_ROOT/docker/Dockerfile.gfx906-v028-phase135-qwen36-fused" \
    -t "$CANDIDATE_IMAGE" "$REPO_ROOT" >"$RESULT_ROOT/image-build.log" 2>&1
(
    cd "$PRODUCTION_WORKDIR"
    docker compose --env-file "$PRODUCTION_ENV_FILE" -f "$PRODUCTION_COMPOSE_FILE" down
)
run_variant control "$BASE_IMAGE" "$CONTROL_PORT"
run_variant candidate "$CANDIDATE_IMAGE" "$CANDIDATE_PORT"
restore_production
trap - EXIT
echo "$RESULT_ROOT"
