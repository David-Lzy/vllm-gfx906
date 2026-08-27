#!/usr/bin/env bash
set -euo pipefail

image=${VLLM_IMAGE:-ghcr.io/david-lzy/vllm-gfx906:v0.28.0-gfx906.1}
model=${MODEL_SOURCE:-cyankiwi/Qwen3.5-9B-AWQ-4bit}
hf_cache=${HF_CACHE_DIR:?Set HF_CACHE_DIR to a writable host directory}
vllm_cache=${VLLM_CACHE_DIR:?Set VLLM_CACHE_DIR to a writable host directory}

exec docker run --rm \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --ipc host \
  --shm-size 32g \
  -p 127.0.0.1:8002:8000 \
  -e HIP_VISIBLE_DEVICES=0 \
  -e PYTORCH_ROCM_ARCH=gfx906 \
  -e VLLM_TARGET_DEVICE=rocm \
  -e VLLM_ROCM_USE_AITER=0 \
  -v "$hf_cache:/root/.cache/huggingface" \
  -v "$vllm_cache:/root/.cache/vllm" \
  "$image" serve "$model" \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name "$model" \
  --dtype float16 \
  --kv-cache-dtype float16 \
  --tensor-parallel-size 1 \
  --linear-backend gfx906_gptq \
  --max-model-len 100000 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 32768 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --limit-mm-per-prompt '{"image":64,"video":0}'
