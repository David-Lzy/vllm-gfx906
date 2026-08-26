#!/usr/bin/env bash
# Reversible Qwen3.5 9B TP1 gate for the gfx906 GDN output-norm overlay.
set -euo pipefail

: "${ALLOW_GPU2_EXPERIMENT:?set to 1 after confirming GPU2 is available}"
if [[ "$ALLOW_GPU2_EXPERIMENT" != "1" ]]; then
    echo "ALLOW_GPU2_EXPERIMENT must be 1" >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
BUILD_ROOT=${BUILD_ROOT:-/mnt/disk2/vllm-gfx906-build}
PHASE_ROOT="$BUILD_ROOT/phase-142-gfx906-qwen-gdn-output-norm-v028"
RESULT_ROOT="$PHASE_ROOT/results/$(date -u +%Y%m%dT%H%M%SZ)"
MODEL_REPOSITORY_DIR=${MODEL_REPOSITORY_DIR:-/mnt/disk2/hf_cache/hub/models--cyankiwi--Qwen3.5-9B-AWQ-4bit}
MODEL_REVISION=${MODEL_REVISION:-156edc4bbeb8d1910ee7be9196bafaf1bc052156}
SERVED_MODEL=${SERVED_MODEL:-cyankiwi/Qwen3.5-9B-AWQ-4bit}
CONTROL_IMAGE=${CONTROL_IMAGE:-local/vllm-gfx906:v0.28.0-phase138-qwen36-packed-fused-splitkv29}
CANDIDATE_IMAGE=${CANDIDATE_IMAGE:-local/vllm-gfx906:v0.28.0-phase142-qwen-gdn-output-norm}
GPU_INDEX=${GPU_INDEX:-2}
CONTROL_PORT=${CONTROL_PORT:-18142}
CANDIDATE_PORT=${CANDIDATE_PORT:-18143}
CONTROL_CACHE_DIR=${CONTROL_CACHE_DIR:-}
CANDIDATE_CACHE_DIR=${CANDIDATE_CACHE_DIR:-}
ACTIVE_CONTAINER=

mkdir -p "$RESULT_ROOT"

cleanup() {
    if [[ -n "$ACTIVE_CONTAINER" ]]; then
        docker rm -f "$ACTIVE_CONTAINER" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

require_file() {
    [[ -f "$1" ]] || { echo "missing required file: $1" >&2; exit 2; }
}

wait_for_health() {
    local port=$1
    local container=$2
    for _ in $(seq 1 120); do
        if [[ "$(docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null || true)" != "true" ]]; then
            echo "$container exited before becoming healthy" >&2
            return 1
        fi
        if curl -fsS --max-time 5 "http://127.0.0.1:$port/health" >/dev/null; then
            return
        fi
        sleep 10
    done
    echo "service on port $port did not become healthy" >&2
    return 1
}

capture_production_health() {
    curl -fsS --max-time 5 http://127.0.0.1:8002/health >"$RESULT_ROOT/production-health-$1.txt"
    curl -fsS --max-time 5 http://127.0.0.1:8002/metrics >"$RESULT_ROOT/production-metrics-$1.prom"
}

run_client() {
    local image=$1
    local port=$2
    local output=$3
    docker run --rm --network host -v "$SCRIPT_DIR:/tools:ro" --entrypoint /bin/bash "$image" -lc \
        "/opt/vllm-venv/bin/python /tools/smoke_qwen36_fused_tp4.py --base-url http://127.0.0.1:$port --model $SERVED_MODEL && \\
         /opt/vllm-venv/bin/python /tools/benchmark_qwen36_fused_tp4.py --base-url http://127.0.0.1:$port --model $SERVED_MODEL --concurrency 1 --warmup --ignore-eos && \\
         /opt/vllm-venv/bin/python /tools/benchmark_qwen36_fused_tp4.py --base-url http://127.0.0.1:$port --model $SERVED_MODEL --concurrency 1 --ignore-eos && \\
         /opt/vllm-venv/bin/python /tools/benchmark_qwen36_fused_tp4.py --base-url http://127.0.0.1:$port --model $SERVED_MODEL --concurrency 8 --ignore-eos" \
        >"$output" 2>&1
}

run_variant() {
    local variant=$1
    local image=$2
    local port=$3
    local enabled=$4
    local cache_dir="$RESULT_ROOT/$variant/cache"
    local container="phase142-gdn-$variant"
    local candidate_env=()
    if [[ "$enabled" == "1" ]]; then
        candidate_env=(-e VLLM_ROCM_ENABLE_GFX906_QWEN_GDN_OUTPUT_NORM=1)
    fi
    if [[ "$variant" == "control" && -n "$CONTROL_CACHE_DIR" ]]; then
        cache_dir="$CONTROL_CACHE_DIR"
    fi
    if [[ "$variant" == "candidate" && -n "$CANDIDATE_CACHE_DIR" ]]; then
        cache_dir="$CANDIDATE_CACHE_DIR"
    fi
    mkdir -p "$cache_dir/triton"
    printf '%s\n' "$cache_dir" >"$RESULT_ROOT/$variant-cache-dir.txt"
    local started
    started=$(date +%s)
    ACTIVE_CONTAINER=$container
    docker run -d --name "$container" --network host --ipc host \
        --device=/dev/kfd --device=/dev/dri --group-add video --security-opt label=disable \
        -e HIP_VISIBLE_DEVICES="$GPU_INDEX" -e HSA_OVERRIDE_GFX_VERSION=9.0.6 \
        -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_ROCM_USE_AITER=0 \
        "${candidate_env[@]}" \
        -e TRITON_CACHE_DIR=/root/.triton/cache -e OMP_NUM_THREADS=12 \
        -v "$MODEL_REPOSITORY_DIR:/model:ro" -v "$cache_dir/vllm:/root/.cache/vllm" \
        -v "$cache_dir/triton:/root/.triton/cache" \
        --entrypoint /bin/bash "$image" -lc \
        "exec /opt/vllm-venv/bin/vllm serve /model/snapshots/$MODEL_REVISION --host 127.0.0.1 --port $port \\
          --served-model-name $SERVED_MODEL --tensor-parallel-size 1 --dtype float16 \\
          --trust-remote-code --linear-backend gfx906_gptq --max-model-len 100000 \\
          --gpu-memory-utilization 0.90 --max-num-seqs 8 --max-num-batched-tokens 32768 \\
          --limit-mm-per-prompt '{\"image\":64,\"video\":0}' \\
          --mm-processor-kwargs '{\"min_pixels\":25088,\"max_pixels\":16777216}' \\
          --mm-processor-cache-type shm --mm-processor-cache-gb 16 \\
          --mm-shm-cache-max-object-size-mb 512 --mm-tensor-ipc direct_rpc \\
          --mm-encoder-tp-mode data --renderer-num-workers 1 --enable-prefix-caching \\
          --enable-chunked-prefill --long-prefill-token-threshold 8192 --skip-mm-profiling \\
          --reasoning-parser qwen3 --default-chat-template-kwargs '{\"enable_thinking\":false}'" \
        >"$RESULT_ROOT/$variant-container-id.txt"
    if ! wait_for_health "$port" "$container"; then
        docker logs "$container" >"$RESULT_ROOT/$variant-server.log" 2>&1 || true
        return 1
    fi
    printf '%s\n' "$(( $(date +%s) - started ))" >"$RESULT_ROOT/$variant-startup-seconds.txt"
    if ! run_client "$image" "$port" "$RESULT_ROOT/$variant-client.log"; then
        docker logs "$container" >"$RESULT_ROOT/$variant-server.log" 2>&1 || true
        return 1
    fi
    curl -fsS --max-time 10 "http://127.0.0.1:$port/metrics" >"$RESULT_ROOT/$variant-metrics.prom"
    docker logs "$container" >"$RESULT_ROOT/$variant-server.log" 2>&1
    if rg -ni 'oom|traceback|xgrammar|failed to advance fsm|rccl.*fatal|nccl.*fatal' \
        "$RESULT_ROOT/$variant-server.log" >"$RESULT_ROOT/$variant-error-scan.txt"; then
        echo "fatal log signature in $variant" >&2
        return 1
    fi
    if rg -q 'vllm:num_requests_(running|waiting).* [1-9][0-9]*\.0' \
        "$RESULT_ROOT/$variant-metrics.prom"; then
        echo "queue did not drain for $variant" >&2
        return 1
    fi
    docker rm -f "$container" >/dev/null
    ACTIVE_CONTAINER=
}

require_file "$MODEL_REPOSITORY_DIR/snapshots/$MODEL_REVISION/config.json"
capture_production_health before
if pgrep -af 'xmrig|xmr|monero' >"$RESULT_ROOT/xmr-preflight.txt"; then
    echo "XMR is active; stop it before the exclusive GPU2 test" >&2
    exit 3
fi
docker build -f "$REPO_ROOT/docker/Dockerfile.gfx906-v028-phase142-qwen-gdn-output-norm" \
    -t "$CANDIDATE_IMAGE" "$REPO_ROOT" >"$RESULT_ROOT/image-build.log" 2>&1
run_variant control "$CONTROL_IMAGE" "$CONTROL_PORT" 0
run_variant candidate "$CANDIDATE_IMAGE" "$CANDIDATE_PORT" 1
capture_production_health after
echo "$RESULT_ROOT"
