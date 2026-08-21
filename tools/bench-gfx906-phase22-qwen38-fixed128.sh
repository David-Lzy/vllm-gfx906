#!/usr/bin/env bash
# Record sequential fixed-128-token decode samples after the Phase 22 gate.
set -euo pipefail

readonly MODE="${1:-no-mtp}"
readonly PORT=18075
readonly MODEL="qwen38-phase22"
readonly ROOT="/mnt/disk2/vllm-gfx906-build/phase-22"
readonly RESULT_DIR="${ROOT}/results/$(date -u +%Y%m%dT%H%M%SZ)-${MODE}-fixed128"
mkdir -p "${RESULT_DIR}"

payload="$(jq -nc --arg model "${MODEL}" \
    '{model: $model, temperature: 0, min_tokens: 128, max_tokens: 128,
      messages: [{role: "user", content: "Write exactly 128 concise tokens about reliable GPU inference."}]}')"

request() {
    local name="$1"
    local output="${RESULT_DIR}/${name}.json"
    local started ended elapsed tokens throughput
    started="$(date +%s.%N)"
    curl --fail --silent --show-error --max-time 900 \
        -H 'content-type: application/json' --data "${payload}" \
        "http://127.0.0.1:${PORT}/v1/chat/completions" > "${output}"
    ended="$(date +%s.%N)"
    elapsed="$(awk -v start="${started}" -v end="${ended}" 'BEGIN { printf "%.6f", end - start }')"
    tokens="$(jq -er '.usage.completion_tokens' "${output}")"
    throughput="$(awk -v tokens="${tokens}" -v elapsed="${elapsed}" 'BEGIN { printf "%.6f", tokens / elapsed }')"
    jq -n --arg sample "${name}" --argjson elapsed_seconds "${elapsed}" \
        --argjson completion_tokens "${tokens}" --argjson completion_tok_s "${throughput}" \
        '{sample: $sample, elapsed_seconds: $elapsed_seconds,
          completion_tokens: $completion_tokens, completion_tok_s: $completion_tok_s}' \
        >> "${RESULT_DIR}/samples.jsonl"
}

request warmup
for index in 1 2 3 4 5; do
    request "sample_${index}"
done

jq -s '
  map(select(.sample != "warmup") | .completion_tok_s) | sort |
  {count: length, median_tok_s: .[length / 2 | floor], samples_tok_s: .}
' "${RESULT_DIR}/samples.jsonl" > "${RESULT_DIR}/summary.json"
curl --fail --silent "http://127.0.0.1:${PORT}/metrics" \
    > "${RESULT_DIR}/metrics.prom"
cat "${RESULT_DIR}/summary.json"
