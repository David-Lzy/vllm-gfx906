#!/usr/bin/env bash
# Validate an already-running Phase 19 server without owning its lifecycle.
set -euo pipefail

endpoint="${ENDPOINT:-http://127.0.0.1:18028}"
model="${MODEL:-cyankiwi/Qwen3.5-9B-AWQ-4bit}"
test_image="${TEST_IMAGE:?Set TEST_IMAGE to a 256px image file.}"
result_dir="${RESULT_DIR:?Set RESULT_DIR to the persistent server result directory.}"
ready_timeout_seconds="${READY_TIMEOUT_SECONDS:-1800}"

require_command() {
  command -v "$1" >/dev/null || {
    printf 'Required command not found: %s\n' "$1" >&2
    exit 2
  }
}

for command in base64 curl file jq; do
  require_command "$command"
done

if [[ ! -f "$test_image" ]]; then
  printf 'TEST_IMAGE does not exist: %s\n' "$test_image" >&2
  exit 2
fi

mkdir -p "$result_dir"
mime_type="$(file --brief --mime-type "$test_image")"
case "$mime_type" in
  image/jpeg|image/png|image/webp) ;;
  *)
    printf 'TEST_IMAGE must be JPEG, PNG, or WebP; got %s\n' "$mime_type" >&2
    exit 2
    ;;
esac
image_url="data:${mime_type};base64,$(base64 -w0 "$test_image")"

deadline=$((SECONDS + ready_timeout_seconds))
until curl --fail --silent --show-error --max-time 3 "$endpoint/health" \
  >"$result_dir/health.json" 2>>"$result_dir/request-runner.log"; do
  if (( SECONDS >= deadline )); then
    printf 'Server did not become healthy within %s seconds.\n' \
      "$ready_timeout_seconds" >&2
    exit 1
  fi
  sleep 10
done

curl --fail --silent --show-error "$endpoint/v1/models" | jq . \
  >"$result_dir/models.json"

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
  jq -e '.choices[0].message.content | strings | length > 0' \
    "$response_file" >/dev/null
}

jq -n --arg model "$model" '{model: $model, temperature: 0, max_tokens: 32,
  messages: [{role: "user", content: "Reply with the word READY."}]}' \
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

curl --fail --silent --show-error "$endpoint/metrics" \
  >"$result_dir/metrics-after.prom"
printf 'PASS: text, 1/2 image, and JSON 3/3 succeeded. Results: %s\n' "$result_dir"
