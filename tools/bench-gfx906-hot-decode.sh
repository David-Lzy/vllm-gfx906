#!/usr/bin/env bash
# Measure warmed sequential text completion on an already healthy dev endpoint.
# The caller owns the container lifecycle; this script never touches production.
set -euo pipefail

endpoint="${ENDPOINT:-http://127.0.0.1:18029}"
model="${MODEL:-cyankiwi/Qwen3.5-9B-AWQ-4bit}"
result_dir="${RESULT_DIR:?Set RESULT_DIR to the retained experiment result directory.}"
warmups="${WARMUPS:-2}"
iterations="${ITERATIONS:-3}"
max_tokens="${MAX_TOKENS:-64}"
min_tokens="${MIN_TOKENS:-$max_tokens}"

for command in curl jq date awk; do
  command -v "$command" >/dev/null || {
    printf 'Required command not found: %s\n' "$command" >&2
    exit 2
  }
done

mkdir -p "$result_dir"
request_file="$result_dir/hot-decode-request.json"
result_file="$result_dir/hot-decode.jsonl"
metrics_file="$result_dir/metrics-after-hot-decode.prom"

jq -n --arg model "$model" --argjson max_tokens "$max_tokens" --argjson min_tokens "$min_tokens" \
  '{model: $model, temperature: 0, max_tokens: $max_tokens, min_tokens: $min_tokens,
    messages: [{role: "user", content: "Write a concise factual note about language-model inference. Continue until the requested length is reached."}]}' \
  >"$request_file"

request_once() {
  local phase="$1"
  local iteration="$2"
  local response_file="$result_dir/hot-decode-${phase}-${iteration}.json"
  local started_ns ended_ns elapsed_ms completion_tokens content

  started_ns="$(date +%s%N)"
  curl --fail --silent --show-error --connect-timeout 10 --max-time 900 \
    --header 'Content-Type: application/json' \
    --data-binary "@$request_file" \
    "$endpoint/v1/chat/completions" >"$response_file"
  ended_ns="$(date +%s%N)"
  content="$(jq -er '.choices[0].message.content | strings | select(length > 0)' "$response_file")"
  completion_tokens="$(jq -er '.usage.completion_tokens // 0' "$response_file")"
  elapsed_ms="$(awk -v start="$started_ns" -v end="$ended_ns" 'BEGIN { printf "%.6f", (end - start) / 1000000 }')"

  jq -n \
    --arg phase "$phase" \
    --argjson iteration "$iteration" \
    --argjson elapsed_ms "$elapsed_ms" \
    --argjson completion_tokens "$completion_tokens" \
    --arg content "$content" \
    '{phase: $phase, iteration: $iteration, elapsed_ms: $elapsed_ms,
      completion_tokens: $completion_tokens, content: $content,
      completion_tokens_per_second: (if $elapsed_ms > 0 then $completion_tokens / ($elapsed_ms / 1000) else null end)}' \
      >>"$result_file"
}

: >"$result_file"
for ((iteration = 1; iteration <= warmups; iteration++)); do
  request_once warmup "$iteration"
done
for ((iteration = 1; iteration <= iterations; iteration++)); do
  request_once measured "$iteration"
done

curl --fail --silent --show-error "$endpoint/metrics" >"$metrics_file"
jq -s '{warmups: [.[] | select(.phase == "warmup")], measured: [.[] | select(.phase == "measured")]}' \
  "$result_file" >"$result_dir/hot-decode-summary.json"
printf 'PASS: wrote %s\n' "$result_dir/hot-decode-summary.json"
