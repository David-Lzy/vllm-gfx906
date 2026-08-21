#!/usr/bin/env bash
# Benchmark one warmed OpenAI-compatible Qwen worker with the routine gfx906
# release gate: text, one/two 256px images, C8 text throughput, and JSON.
# The caller owns the server lifecycle; this script never targets production.
set -euo pipefail

endpoint="${ENDPOINT:?Set ENDPOINT to the temporary worker endpoint.}"
model="${MODEL:?Set MODEL to the served model name.}"
test_image="${TEST_IMAGE:?Set TEST_IMAGE to a 256px PNG/JPEG/WebP fixture.}"
result_dir="${RESULT_DIR:?Set RESULT_DIR to a disk2 result directory.}"
warmups="${WARMUPS:-2}"
iterations="${ITERATIONS:-5}"
max_tokens="${MAX_TOKENS:-64}"
min_tokens="${MIN_TOKENS:-$max_tokens}"
c8_requests="${C8_REQUESTS:-8}"
c8_warmups="${C8_WARMUPS:-1}"

for command in base64 curl date file jq awk; do
  command -v "$command" >/dev/null || {
    printf 'Required command not found: %s\n' "$command" >&2
    exit 2
  }
done

if [[ ! -f "$test_image" ]]; then
  printf 'TEST_IMAGE does not exist: %s\n' "$test_image" >&2
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

endpoint="${endpoint%/}"
mkdir -p "$result_dir/payloads" "$result_dir/responses"
image_url="data:${mime_type};base64,$(base64 -w0 "$test_image")"

jq -n \
  --arg endpoint "$endpoint" \
  --arg model "$model" \
  --arg test_image "$test_image" \
  --arg mime_type "$mime_type" \
  --argjson warmups "$warmups" \
  --argjson iterations "$iterations" \
  --argjson max_tokens "$max_tokens" \
  --argjson c8_requests "$c8_requests" \
  --argjson c8_warmups "$c8_warmups" \
  '{endpoint: $endpoint, model: $model, test_image: $test_image,
    test_image_mime_type: $mime_type, warmups: $warmups,
    iterations: $iterations, max_tokens: $max_tokens, c8_requests: $c8_requests,
    c8_warmups: $c8_warmups}' \
  >"$result_dir/metadata.json"

jq -n --arg model "$model" --argjson max_tokens "$max_tokens" --argjson min_tokens "$min_tokens" \
  '{model: $model, temperature: 0, max_tokens: $max_tokens, min_tokens: $min_tokens,
    messages: [{role: "user", content: "Write a concise factual note about language-model inference. Continue until the requested length is reached."}]}' \
  >"$result_dir/payloads/text.json"

for image_count in 1 2; do
  jq -cn \
    --arg model "$model" \
    --arg url "$image_url" \
    --argjson count "$image_count" \
    --argjson max_tokens "$max_tokens" \
    --argjson min_tokens "$min_tokens" \
    '[range(0; $count) | {type: "image_url", image_url: {url: $url}}] as $images
      | {model: $model, temperature: 0, max_tokens: $max_tokens, min_tokens: $min_tokens,
         messages: [{role: "user", content: ([{type: "text", text: "Describe the image concisely."}] + $images)}]}' \
    >"$result_dir/payloads/image${image_count}.json"
done

jq -n --arg model "$model" \
  '{model: $model, temperature: 0, max_tokens: 64,
    response_format: {type: "json_object"},
    messages: [{role: "user", content: "Return exactly one JSON object with boolean key ok set to true."}]}' \
  >"$result_dir/payloads/json.json"

records="$result_dir/records.jsonl"
: >"$records"

post_once() {
  local scenario="$1"
  local phase="$2"
  local iteration="$3"
  local payload="$4"
  local response="$result_dir/responses/${scenario}-${phase}-${iteration}.json"
  local started_ns ended_ns elapsed_ms completion_tokens content

  started_ns="$(date +%s%N)"
  curl --fail --silent --show-error --connect-timeout 10 --max-time 900 \
    --header 'Content-Type: application/json' \
    --data-binary "@$payload" \
    "$endpoint/v1/chat/completions" >"$response"
  ended_ns="$(date +%s%N)"
  content="$(jq -er '.choices[0].message.content | strings | select(length > 0)' "$response")"
  completion_tokens="$(jq -er '.usage.completion_tokens // 0' "$response")"
  elapsed_ms="$(awk -v start="$started_ns" -v end="$ended_ns" 'BEGIN { printf "%.6f", (end - start) / 1000000 }')"

  jq -n \
    --arg scenario "$scenario" \
    --arg phase "$phase" \
    --argjson iteration "$iteration" \
    --argjson elapsed_ms "$elapsed_ms" \
    --argjson completion_tokens "$completion_tokens" \
    --arg content "$content" \
    '{scenario: $scenario, phase: $phase, iteration: $iteration,
      elapsed_ms: $elapsed_ms, completion_tokens: $completion_tokens,
      content: $content,
      completion_tokens_per_second: (if $elapsed_ms > 0 then $completion_tokens / ($elapsed_ms / 1000) else null end)}' \
    >>"$records"
}

run_c1() {
  local scenario="$1"
  local payload="$2"
  local iteration
  for ((iteration = 1; iteration <= warmups; iteration++)); do
    post_once "$scenario" warmup "$iteration" "$payload"
  done
  for ((iteration = 1; iteration <= iterations; iteration++)); do
    post_once "$scenario" measured "$iteration" "$payload"
  done
}

run_c8() {
  local scenario="text_c8"
  local payload="$result_dir/payloads/text.json"
  local started_ns ended_ns elapsed_ms completed=0 failures=0
  local -a pids=()
  local index pid response tokens warmup
  local response_dir warmup_dir

  # First use of a concurrent shape can JIT a distinct kernel. Warm it before
  # timing so the C8 result represents steady-state throughput.
  for ((warmup = 1; warmup <= c8_warmups; warmup++)); do
    warmup_dir="$result_dir/responses/${scenario}-warmup-${warmup}"
    mkdir -p "$warmup_dir"
    pids=()
    for ((index = 1; index <= c8_requests; index++)); do
      curl --fail --silent --show-error --connect-timeout 10 --max-time 900 \
        --header 'Content-Type: application/json' \
        --data-binary "@$payload" \
        "$endpoint/v1/chat/completions" >"$warmup_dir/$index.json" &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do
      wait "$pid"
    done
    for ((index = 1; index <= c8_requests; index++)); do
      jq -er '.choices[0].message.content | strings | select(length > 0)' \
        "$warmup_dir/$index.json" >/dev/null
    done
  done

  response_dir="$result_dir/responses/$scenario"
  mkdir -p "$response_dir"

  started_ns="$(date +%s%N)"
  for ((index = 1; index <= c8_requests; index++)); do
    curl --fail --silent --show-error --connect-timeout 10 --max-time 900 \
      --header 'Content-Type: application/json' \
      --data-binary "@$payload" \
      "$endpoint/v1/chat/completions" >"$response_dir/$index.json" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failures=$((failures + 1))
    fi
  done
  ended_ns="$(date +%s%N)"

  for ((index = 1; index <= c8_requests; index++)); do
    response="$response_dir/$index.json"
    if [[ ! -s "$response" ]] || ! jq -er '.choices[0].message.content | strings | select(length > 0)' "$response" >/dev/null; then
      failures=$((failures + 1))
      continue
    fi
    tokens="$(jq -er '.usage.completion_tokens // 0' "$response")"
    completed=$((completed + tokens))
  done
  elapsed_ms="$(awk -v start="$started_ns" -v end="$ended_ns" 'BEGIN { printf "%.6f", (end - start) / 1000000 }')"
  if (( failures > 0 )); then
    printf 'C8 benchmark had %s failed responses.\n' "$failures" >&2
    exit 1
  fi

  jq -cn \
    --arg scenario "$scenario" \
    --argjson requests "$c8_requests" \
    --argjson elapsed_ms "$elapsed_ms" \
    --argjson completion_tokens "$completed" \
    '{scenario: $scenario, phase: "measured", requests: $requests,
      elapsed_ms: $elapsed_ms, completion_tokens: $completion_tokens,
      completion_tokens_per_second: (if $elapsed_ms > 0 then $completion_tokens / ($elapsed_ms / 1000) else null end)}' \
    >>"$records"
}

run_json() {
  local attempt response
  for ((attempt = 1; attempt <= 3; attempt++)); do
    response="$result_dir/responses/json-${attempt}.json"
    curl --fail --silent --show-error --connect-timeout 10 --max-time 900 \
      --header 'Content-Type: application/json' \
      --data-binary "@$result_dir/payloads/json.json" \
      "$endpoint/v1/chat/completions" >"$response"
    jq -er '.choices[0].message.content | fromjson | .ok == true' "$response" >/dev/null
  done
}

run_c1 text_c1 "$result_dir/payloads/text.json"
run_c1 image1_c1 "$result_dir/payloads/image1.json"
run_c1 image2_c1 "$result_dir/payloads/image2.json"
run_c8
run_json

curl --fail --silent --show-error "$endpoint/metrics" >"$result_dir/metrics-after.prom"
if awk '/vllm:num_requests_(running|waiting)/ && ($NF + 0) > 0 { found = 1 } END { exit found ? 0 : 1 }' \
  "$result_dir/metrics-after.prom"; then
  printf 'Worker still has running or waiting requests after benchmark.\n' >&2
  exit 1
fi

jq -s '
  [group_by(.scenario)[]
   | {scenario: .[0].scenario,
      samples: length,
      measured_samples: ([.[] | select(.phase == "measured")] | length),
      median_elapsed_ms: ([.[] | select(.phase == "measured") | .elapsed_ms] | sort | .[(length / 2 | floor)]),
      median_completion_tokens_per_second: ([.[] | select(.phase == "measured") | .completion_tokens_per_second] | sort | .[(length / 2 | floor)])}]
' "$records" >"$result_dir/summary.json"

{
  printf '# Small Multimodal Parity Benchmark\n\n'
  printf -- '- Endpoint: `%s`\n- Model: `%s`\n- Fixture: `%s`\n\n' "$endpoint" "$model" "$test_image"
  printf '| Scenario | Samples | Median latency ms | Completion tok/s |\n| --- | ---: | ---: | ---: |\n'
  jq -r '.[] | "| \(.scenario) | \(.measured_samples) | \(.median_elapsed_ms // 0 | tostring) | \(.median_completion_tokens_per_second // 0 | tostring) |"' \
    "$result_dir/summary.json"
  printf '\nJSON constrained output: 3/3 passed.\n'
} >"$result_dir/summary.md"

printf 'PASS: wrote %s\n' "$result_dir/summary.json"
