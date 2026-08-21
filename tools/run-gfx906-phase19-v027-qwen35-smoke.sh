#!/usr/bin/env bash
# Run the minimal Phase 19 Qwen3.5 multimodal compatibility gate on one dev GPU.
# No production port, model cache, or container is changed by this harness.
set -euo pipefail

image="${IMAGE:-local/vllm-gfx906:v0.27.1-phase19-triton37}"
model="${MODEL:?Set MODEL to the Hugging Face model id.}"
hf_cache_dir="${HF_CACHE_DIR:?Set HF_CACHE_DIR to a populated Hugging Face cache.}"
test_image="${TEST_IMAGE:?Set TEST_IMAGE to a 256px image file.}"
gpu="${GPU:-2}"
host_port="${HOST_PORT:-18027}"
max_model_len="${MAX_MODEL_LEN:-100000}"
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.90}"
max_num_seqs="${MAX_NUM_SEQS:-8}"
max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-32768}"
result_root="${RESULT_ROOT:-/mnt/disk2/vllm-gfx906-build/phase-19/results}"
run_id="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-qwen35-smoke}"
result_dir="${result_root}/${run_id}"
container_name="vllm-gfx906-phase19-qwen35-${run_id,,}"
container_name="${container_name//[^a-z0-9_.-]/-}"
vllm_cache_dir="${VLLM_CACHE_DIR:-/mnt/disk2/vllm-gfx906-build/phase-19/cache/${run_id}}"
triton_cache_dir="${TRITON_CACHE_DIR:-${vllm_cache_dir}/triton-cache}"
disabled_kernels="${VLLM_DISABLED_KERNELS:-}"
prefer_exllama="${VLLM_ROCM_GFX906_PREFER_EXLLAMA:-0}"
keep_container="${KEEP_CONTAINER:-0}"
docker_env_args=()

if [[ -n "$disabled_kernels" ]]; then
  docker_env_args+=(--env "VLLM_DISABLED_KERNELS=$disabled_kernels")
fi
if [[ "$prefer_exllama" != "0" ]]; then
  docker_env_args+=(--env "VLLM_ROCM_GFX906_PREFER_EXLLAMA=$prefer_exllama")
fi

mkdir -p "$result_dir" "$vllm_cache_dir" "$triton_cache_dir"

cleanup() {
  if [[ "$keep_container" != "1" ]]; then
    docker rm -f "$container_name" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

require_command() {
  command -v "$1" >/dev/null || {
    printf 'Required command not found: %s\n' "$1" >&2
    exit 2
  }
}

for command in docker curl jq file base64; do
  require_command "$command"
done

if [[ ! -f "$test_image" ]]; then
  printf 'TEST_IMAGE does not exist: %s\n' "$test_image" >&2
  exit 2
fi

if [[ ! -d "$hf_cache_dir" ]]; then
  printf 'HF_CACHE_DIR does not exist: %s\n' "$hf_cache_dir" >&2
  exit 2
fi

if docker ps -a --format '{{.Names}}' | grep -Fxq "$container_name"; then
  printf 'Refusing to reuse an existing container: %s\n' "$container_name" >&2
  exit 2
fi

mime_type="$(file --brief --mime-type "$test_image")"
case "$mime_type" in
  image/jpeg|image/png|image/webp) ;;
  *)
    printf 'TEST_IMAGE must be JPEG, PNG, or WebP; got %s\n' "$mime_type" >&2
    exit 2
    ;;
esac

image_url="data:${mime_type};base64,$(base64 -w0 "$test_image")"

cat >"$result_dir/metadata.json" <<EOF
{
  "image": "${image}",
  "model": "${model}",
  "gpu": "${gpu}",
  "host_port": ${host_port},
  "max_model_len": ${max_model_len},
  "gpu_memory_utilization": ${gpu_memory_utilization},
  "max_num_seqs": ${max_num_seqs},
  "max_num_batched_tokens": ${max_num_batched_tokens},
  "prefer_exllama": "${prefer_exllama}",
  "test_image": "${test_image}",
  "test_image_mime_type": "${mime_type}"
}
EOF

printf 'Starting %s on HIP_VISIBLE_DEVICES=%s. Logs: %s/server.log\n' \
  "$image" "$gpu" "$result_dir"
docker run --detach --rm \
  --name "$container_name" \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --ipc host \
  --publish "127.0.0.1:${host_port}:8000" \
  --volume "${hf_cache_dir}:/root/.cache/huggingface:ro" \
  --volume "${vllm_cache_dir}:/root/.cache/vllm" \
  --volume "${triton_cache_dir}:/root/.triton/cache" \
  --env HIP_VISIBLE_DEVICES="$gpu" \
  --env PYTORCH_ROCM_ARCH=gfx906 \
  --env ROCM_ARCH=gfx906 \
  --env ROCM_PATH=/opt/rocm \
  --env ROCBLAS_TENSILE_LIBPATH="${ROCBLAS_TENSILE_LIBPATH:-/opt/rocm/lib/rocblas/library}" \
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
  "${docker_env_args[@]}" \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env TRITON_CACHE_DIR=/root/.triton/cache \
  "$image" \
  serve "$model" \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name "$model" \
  --trust-remote-code \
  --dtype float16 \
  --kv-cache-dtype float16 \
  --tensor-parallel-size 1 \
  --max-model-len "$max_model_len" \
  --gpu-memory-utilization "$gpu_memory_utilization" \
  --max-num-seqs "$max_num_seqs" \
  --max-num-batched-tokens "$max_num_batched_tokens" \
  --renderer-num-workers 1 \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --limit-mm-per-prompt '{"image":64,"video":0}' \
  --skip-mm-profiling \
  --mm-processor-kwargs.min_pixels 25088 \
  --mm-processor-kwargs.max_pixels 16777216 \
  --mm-encoder-tp-mode data \
  --mm-tensor-ipc direct_rpc \
  --mm-processor-cache-type shm \
  --mm-processor-cache-gb 16 \
  --mm-shm-cache-max-object-size-mb 512 \
  --enable-chunked-prefill \
  --long-prefill-token-threshold 8192 \
  >"$result_dir/container-id.txt"

endpoint="http://127.0.0.1:${host_port}"
deadline=$((SECONDS + 1800))
while ! curl --fail --silent --show-error --max-time 3 "$endpoint/health" \
  >"$result_dir/health.json" 2>>"$result_dir/server.log"; do
  docker logs "$container_name" >"$result_dir/server-current.log" 2>&1 || true
  if [[ "$(docker inspect --format '{{.State.Running}}' "$container_name" 2>/dev/null || true)" != "true" ]]; then
    cp "$result_dir/server-current.log" "$result_dir/server.log"
    printf 'Server container exited before becoming healthy.\n' >&2
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    docker logs "$container_name" >"$result_dir/server.log" 2>&1 || true
    printf 'Server did not become healthy within 1800 seconds.\n' >&2
    exit 1
  fi
  sleep 10
done
docker logs "$container_name" >"$result_dir/server.log" 2>&1
curl --fail --silent --show-error "$endpoint/v1/models" | jq . >"$result_dir/models.json"

post_request() {
  local name="$1"
  local payload_file="$2"
  local response_file="$result_dir/${name}.json"
  curl --fail --silent --show-error \
    --connect-timeout 10 \
    --max-time 900 \
    --header 'Content-Type: application/json' \
    --data-binary "@${payload_file}" \
    "$endpoint/v1/chat/completions" | tee "$response_file" >/dev/null
  jq -e '.choices[0].message.content | strings | length > 0' "$response_file" >/dev/null
}

jq -n --arg model "$model" '{model: $model, temperature: 0, max_tokens: 32, messages: [{role: "user", content: "Reply with the word READY."}]}' \
  >"$result_dir/text-request.json"
post_request text "$result_dir/text-request.json"

for image_count in 1 2; do
  jq -n --arg model "$model" --arg url "$image_url" --argjson count "$image_count" '
    [range(0; $count) | {type: "image_url", image_url: {url: $url}}] as $images
    | {model: $model, temperature: 0, max_tokens: 64,
       messages: [{role: "user", content: ([{type: "text", text: "Describe the image concisely."}] + $images)}]}
  ' >"$result_dir/image-${image_count}-request.json"
  post_request "image-${image_count}" "$result_dir/image-${image_count}-request.json"
done

for attempt in 1 2 3; do
  jq -n --arg model "$model" '{model: $model, temperature: 0, max_tokens: 64,
    response_format: {type: "json_object"},
    messages: [{role: "user", content: "Return exactly one JSON object with boolean key ok set to true."}]}' \
    >"$result_dir/json-${attempt}-request.json"
  post_request "json-${attempt}" "$result_dir/json-${attempt}-request.json"
  jq -er '.choices[0].message.content | fromjson | .ok == true' \
    "$result_dir/json-${attempt}.json" >/dev/null
done

docker logs "$container_name" >"$result_dir/server.log" 2>&1
if rg -n -i 'out of memory|xgrammar|failed to advance fsm|rccl.*fatal|nccl.*fatal|engine core initialization failed' \
  "$result_dir/server.log" >"$result_dir/error-scan.txt"; then
  printf 'Server log error signature found.\n' >&2
  exit 1
fi

printf 'PASS: text, 1/2 image, and JSON 3/3 succeeded. Results: %s\n' "$result_dir"
if [[ "$keep_container" == "1" ]]; then
  printf 'Container retained for follow-up benchmarks: %s\n' "$container_name"
fi
