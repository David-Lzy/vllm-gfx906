#!/usr/bin/env bash
# Compare the retained production image and v0.27 Phase 21 on one development
# MI50 with identical serving settings. This never touches production port 8002.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
baseline_image="${BASELINE_IMAGE:-aiinfos/vllm-gfx906-mobydick:v0.23.1rc0.x-rocm7.2.1-pytorch2.11.0}"
candidate_image="${CANDIDATE_IMAGE:-local/vllm-gfx906:v0.27.1-phase21-llmm1}"
model="${MODEL:-cyankiwi/Qwen3.5-9B-AWQ-4bit}"
hf_cache_dir="${HF_CACHE_DIR:?Set HF_CACHE_DIR to the existing Qwen HF cache.}"
test_image="${TEST_IMAGE:?Set TEST_IMAGE to the existing 256px fixture.}"
gpu="${GPU:-2}"
host_port="${HOST_PORT:-18073}"
result_root="${RESULT_ROOT:-/mnt/disk2/vllm-gfx906-build/phase-23/results}"
run_id="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-small-mm-parity}"
phase_root="${PHASE_ROOT:-/mnt/disk2/vllm-gfx906-build/phase-23}"
baseline_summary_file="${BASELINE_SUMMARY_FILE:-}"
baseline_label="${BASELINE_LABEL:-v0.23}"
candidate_label="${CANDIDATE_LABEL:-v0.27}"
candidate_runner_label="${CANDIDATE_RUNNER_LABEL:-candidate_v027}"
baseline_linear_backend="${BASELINE_LINEAR_BACKEND:-}"
candidate_linear_backend="${CANDIDATE_LINEAR_BACKEND:-}"
candidate_disable_async_scheduling="${CANDIDATE_DISABLE_ASYNC_SCHEDULING:-0}"
candidate_estimate_cudagraphs="${CANDIDATE_ESTIMATE_CUDAGRAPHS:-}"
candidate_compilation_config="${CANDIDATE_COMPILATION_CONFIG:-}"
attention_config="${ATTENTION_CONFIG:-}"
compilation_config="${COMPILATION_CONFIG:-}"
aot_compile="${VLLM_USE_AOT_COMPILE:-}"
mega_aot_artifact="${VLLM_USE_MEGA_AOT_ARTIFACT:-}"
max_model_len="${MAX_MODEL_LEN:-100000}"
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.90}"
max_num_seqs="${MAX_NUM_SEQS:-8}"
max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-32768}"
result_dir="$result_root/$run_id"

for command in docker curl jq file; do
  command -v "$command" >/dev/null || {
    printf 'Required command not found: %s\n' "$command" >&2
    exit 2
  }
done

if [[ ! -d "$hf_cache_dir" || ! -f "$test_image" ]]; then
  printf 'HF_CACHE_DIR or TEST_IMAGE is unavailable.\n' >&2
  exit 2
fi
if [[ "$candidate_disable_async_scheduling" != "0" && "$candidate_disable_async_scheduling" != "1" ]]; then
  printf 'CANDIDATE_DISABLE_ASYNC_SCHEDULING must be 0 or 1.\n' >&2
  exit 2
fi
if [[ -n "$candidate_estimate_cudagraphs" && "$candidate_estimate_cudagraphs" != "0" && "$candidate_estimate_cudagraphs" != "1" ]]; then
  printf 'CANDIDATE_ESTIMATE_CUDAGRAPHS must be empty, 0, or 1.\n' >&2
  exit 2
fi
if [[ -n "$candidate_compilation_config" ]] && ! jq -e . >/dev/null <<<"$candidate_compilation_config"; then
  printf 'CANDIDATE_COMPILATION_CONFIG must be valid JSON.\n' >&2
  exit 2
fi
if ! curl --fail --silent --show-error --max-time 5 http://127.0.0.1:8002/health >/dev/null; then
  printf 'Production health check failed; refusing to begin isolated comparison.\n' >&2
  exit 2
fi
if ! docker image inspect "$baseline_image" >/dev/null || ! docker image inspect "$candidate_image" >/dev/null; then
  printf 'A required local image is unavailable.\n' >&2
  exit 2
fi

mkdir -p "$result_dir"
jq -n \
  --arg baseline_image "$baseline_image" \
  --arg candidate_image "$candidate_image" \
  --arg model "$model" \
  --arg baseline_label "$baseline_label" \
  --arg candidate_label "$candidate_label" \
  --arg candidate_runner_label "$candidate_runner_label" \
  --arg baseline_linear_backend "$baseline_linear_backend" \
  --arg candidate_linear_backend "$candidate_linear_backend" \
  --arg candidate_disable_async_scheduling "$candidate_disable_async_scheduling" \
  --arg candidate_estimate_cudagraphs "$candidate_estimate_cudagraphs" \
  --arg candidate_compilation_config "$candidate_compilation_config" \
  --arg attention_config "$attention_config" \
  --arg compilation_config "$compilation_config" \
  --arg aot_compile "$aot_compile" \
  --arg mega_aot_artifact "$mega_aot_artifact" \
  --argjson gpu "$gpu" \
  --argjson host_port "$host_port" \
  --argjson max_model_len "$max_model_len" \
  --argjson gpu_memory_utilization "$gpu_memory_utilization" \
  --argjson max_num_seqs "$max_num_seqs" \
  --argjson max_num_batched_tokens "$max_num_batched_tokens" \
  '{baseline_image: $baseline_image, candidate_image: $candidate_image,
    model: $model, baseline_label: $baseline_label, candidate_label: $candidate_label,
    candidate_runner_label: $candidate_runner_label,
    baseline_linear_backend: $baseline_linear_backend,
    candidate_linear_backend: $candidate_linear_backend,
    candidate_disable_async_scheduling: $candidate_disable_async_scheduling,
    candidate_estimate_cudagraphs: $candidate_estimate_cudagraphs,
    candidate_compilation_config: $candidate_compilation_config,
    attention_config: $attention_config,
    compilation_config: $compilation_config, aot_compile: $aot_compile,
    mega_aot_artifact: $mega_aot_artifact,
    gpu: $gpu, host_port: $host_port,
    max_model_len: $max_model_len, gpu_memory_utilization: $gpu_memory_utilization,
    max_num_seqs: $max_num_seqs, max_num_batched_tokens: $max_num_batched_tokens}' \
  >"$result_dir/metadata.json"

container_name=""
cleanup() {
  if [[ -n "$container_name" ]]; then
    docker rm -f "$container_name" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

run_candidate() {
  local label="$1"
  local image="$2"
  local linear_backend="$3"
  local candidate_dir="$result_dir/$label"
  local cache_dir="$phase_root/cache/$run_id/$label"
  local triton_cache_dir="$cache_dir/triton-cache"
  local endpoint="http://127.0.0.1:$host_port"
  local deadline
  local -a attention_args=()
  local -a compilation_args=()
  local -a candidate_compilation_args=()
  local -a aot_compile_env=()
  local -a mega_aot_artifact_env=()
  local -a linear_args=()
  local -a scheduler_args=()
  local -a graph_memory_env=()

  if [[ -n "$linear_backend" ]]; then
    case "$linear_backend" in
      auto|gfx906_gptq) linear_args=(--linear-backend "$linear_backend") ;;
      *)
        printf 'Unsupported linear backend for %s: %s\n' "$label" "$linear_backend" >&2
        exit 2
        ;;
    esac
  fi

  if [[ -n "$attention_config" ]]; then
    attention_args=(--attention-config "$attention_config")
  fi
  if [[ -n "$compilation_config" ]]; then
    compilation_args=(--compilation-config "$compilation_config")
  fi
  if [[ -n "$aot_compile" ]]; then
    aot_compile_env=(--env "VLLM_USE_AOT_COMPILE=$aot_compile")
  fi
  if [[ -n "$mega_aot_artifact" ]]; then
    mega_aot_artifact_env=(--env "VLLM_USE_MEGA_AOT_ARTIFACT=$mega_aot_artifact")
  fi
  if [[ "$label" == "$candidate_runner_label" ]]; then
    if [[ "$candidate_disable_async_scheduling" == "1" ]]; then
      scheduler_args=(--no-async-scheduling)
    fi
    if [[ -n "$candidate_estimate_cudagraphs" ]]; then
      graph_memory_env=(--env "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=$candidate_estimate_cudagraphs")
    fi
    if [[ -n "$candidate_compilation_config" ]]; then
      candidate_compilation_args=(--compilation-config "$candidate_compilation_config")
    fi
  fi

  mkdir -p "$candidate_dir" "$cache_dir" "$triton_cache_dir"
  container_name="vllm-gfx906-phase23-${label}-${run_id,,}"
  container_name="${container_name//[^a-z0-9_.-]/-}"
  if docker ps -a --format '{{.Names}}' | grep -Fxq "$container_name"; then
    printf 'Refusing to reuse existing container: %s\n' "$container_name" >&2
    exit 2
  fi

  printf 'Starting %s on GPU%s with %s\n' "$label" "$gpu" "$image"
  docker run --detach --rm \
    --name "$container_name" \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add video \
    --cap-add SYS_NICE \
    --ipc host \
    --shm-size 64g \
    --publish "127.0.0.1:${host_port}:8000" \
    --volume "${hf_cache_dir}:/root/.cache/huggingface:ro" \
    --volume "${cache_dir}:/root/.cache/vllm" \
    --volume "${triton_cache_dir}:/root/.triton/cache" \
    --env HIP_VISIBLE_DEVICES="$gpu" \
    --env PYTORCH_ROCM_ARCH=gfx906 \
    --env ROCM_ARCH=gfx906 \
    --env ROCM_PATH=/opt/rocm \
    --env VLLM_TARGET_DEVICE=rocm \
    --env VLLM_CACHE_ROOT=/root/.cache/vllm \
    --env VLLM_ENGINE_READY_TIMEOUT_S=1800 \
    --env VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200 \
    --env OMP_NUM_THREADS=12 \
    --env OPENBLAS_NUM_THREADS=12 \
    --env MKL_NUM_THREADS=12 \
    --env NUMEXPR_NUM_THREADS=12 \
    --env TOKENIZERS_PARALLELISM=false \
    --env TORCH_NCCL_ASYNC_ERROR_HANDLING=3 \
    --env FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
    --env FLASH_ATTENTION_TRITON_AMD_REF=TRUE \
    --env FLASH_ATTENTION_TRITON_AMD_AUTOTUNE=0 \
    --env VLLM_ROCM_GFX906_PREFER_EXLLAMA=1 \
    "${aot_compile_env[@]}" \
    "${mega_aot_artifact_env[@]}" \
    "${graph_memory_env[@]}" \
    --env PROCESS_NICE=-5 \
    --env TRITON_CACHE_DIR=/root/.triton/cache \
    --env TORCHINDUCTOR_CACHE_DIR=/root/.cache/vllm/torch_compile_cache/torchinductor \
    --entrypoint /bin/bash \
    "$image" \
    -lc 'mkdir -p /root/.cache/vllm/torch_compile_cache /root/.cache/vllm/torch_compile_cache/torchinductor /root/.triton/cache; exec nice -n "${PROCESS_NICE:-0}" vllm "$@"' \
    vllm-wrapper \
    serve "$model" --host 0.0.0.0 --port 8000 --served-model-name "$model" --trust-remote-code \
    --dtype float16 --kv-cache-dtype float16 --tensor-parallel-size 1 "${linear_args[@]}" \
    --max-model-len "$max_model_len" --gpu-memory-utilization "$gpu_memory_utilization" \
    --max-num-seqs "$max_num_seqs" --max-num-batched-tokens "$max_num_batched_tokens" \
    --renderer-num-workers 1 --enable-prefix-caching --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --limit-mm-per-prompt '{"image":64,"video":0}' --skip-mm-profiling \
    --mm-processor-kwargs.min_pixels 25088 --mm-processor-kwargs.max_pixels 16777216 \
    --mm-encoder-tp-mode data --mm-tensor-ipc direct_rpc \
    --mm-processor-cache-type shm --mm-processor-cache-gb 16 \
    --mm-shm-cache-max-object-size-mb 512 --enable-chunked-prefill \
    --long-prefill-token-threshold 8192 "${scheduler_args[@]}" "${attention_args[@]}" "${compilation_args[@]}" "${candidate_compilation_args[@]}" >"$candidate_dir/container-id.txt"

  deadline=$((SECONDS + 1800))
  until curl --fail --silent --show-error --max-time 3 "$endpoint/health" >"$candidate_dir/health.json" 2>>"$candidate_dir/server.log"; do
    docker logs "$container_name" >"$candidate_dir/server-current.log" 2>&1 || true
    if [[ "$(docker inspect --format '{{.State.Running}}' "$container_name" 2>/dev/null || true)" != "true" ]]; then
      cp "$candidate_dir/server-current.log" "$candidate_dir/server.log"
      printf '%s exited before health.\n' "$label" >&2
      exit 1
    fi
    if (( SECONDS >= deadline )); then
      docker logs "$container_name" >"$candidate_dir/server.log" 2>&1 || true
      printf '%s did not become healthy within 1800 seconds.\n' "$label" >&2
      exit 1
    fi
    sleep 10
  done
  docker logs "$container_name" >"$candidate_dir/server.log" 2>&1
  curl --fail --silent --show-error "$endpoint/v1/models" | jq . >"$candidate_dir/models.json"

  ENDPOINT="$endpoint" MODEL="$model" TEST_IMAGE="$test_image" RESULT_DIR="$candidate_dir/benchmark" \
    "$script_dir/bench-gfx906-small-mm-parity.sh"

  docker logs "$container_name" >"$candidate_dir/server.log" 2>&1
  if rg -n -i 'out of memory|xgrammar|failed to advance fsm|rccl.*fatal|nccl.*fatal|traceback|engine core initialization failed' \
    "$candidate_dir/server.log" >"$candidate_dir/error-scan.txt"; then
    printf '%s log has a fatal error signature.\n' "$label" >&2
    exit 1
  fi
  docker rm -f "$container_name" >/dev/null
  container_name=""
}

if [[ -n "$baseline_summary_file" ]]; then
  if [[ ! -f "$baseline_summary_file" ]]; then
    printf 'BASELINE_SUMMARY_FILE does not exist: %s\n' "$baseline_summary_file" >&2
    exit 2
  fi
  cp "$baseline_summary_file" "$result_dir/baseline-summary-reused.json"
  baseline_summary_file="$result_dir/baseline-summary-reused.json"
else
  run_candidate baseline_v023 "$baseline_image" "$baseline_linear_backend"
  baseline_summary_file="$result_dir/baseline_v023/benchmark/summary.json"
fi
run_candidate "$candidate_runner_label" "$candidate_image" "$candidate_linear_backend"

jq -n \
  --slurpfile baseline "$baseline_summary_file" \
  --slurpfile candidate "$result_dir/$candidate_runner_label/benchmark/summary.json" '
  def by_scenario($items): reduce $items[] as $item ({}; .[$item.scenario] = $item);
  (by_scenario($baseline[0])) as $old |
  (by_scenario($candidate[0])) as $new |
  ["text_c1", "text_c8", "image1_c1", "image2_c1"] as $scenarios |
  {scenarios: [$scenarios[] | {
      scenario: .,
      baseline_toks: $old[.].median_completion_tokens_per_second,
      candidate_toks: $new[.].median_completion_tokens_per_second,
      throughput_ratio: ($new[.].median_completion_tokens_per_second / $old[.].median_completion_tokens_per_second),
      baseline_latency_ms: $old[.].median_elapsed_ms,
      candidate_latency_ms: $new[.].median_elapsed_ms,
      latency_ratio: ($new[.].median_elapsed_ms / $old[.].median_elapsed_ms)
    }],
   release_floor: 0.95,
   release_candidate: (["text_c1", "text_c8", "image1_c1", "image2_c1"]
     | all(. as $scenario |
       (($new[$scenario].median_completion_tokens_per_second / $old[$scenario].median_completion_tokens_per_second) >= 0.95)))
  }' >"$result_dir/comparison.json"

{
  printf '# Small Multimodal Parity Comparison\n\n'
  printf 'Both candidates used GPU%s, the same checkpoint cache, and the production serving parameters.\n\n' "$gpu"
  printf '| Scenario | %s tok/s | %s tok/s | %s / %s | %s ms | %s ms |\n| --- | ---: | ---: | ---: | ---: | ---: |\n' \
    "$baseline_label" "$candidate_label" "$candidate_label" "$baseline_label" \
    "$baseline_label" "$candidate_label"
  jq -r '.scenarios[] | "| \(.scenario) | \(.baseline_toks) | \(.candidate_toks) | \(.throughput_ratio) | \(.baseline_latency_ms) | \(.candidate_latency_ms) |"' \
    "$result_dir/comparison.json"
  jq -r '"\nRelease floor (>=95% in every routine scenario): \(.release_candidate)."' "$result_dir/comparison.json"
} >"$result_dir/comparison.md"

printf 'PASS: comparison written to %s\n' "$result_dir/comparison.md"
