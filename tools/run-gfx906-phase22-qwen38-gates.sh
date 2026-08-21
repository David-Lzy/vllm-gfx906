#!/usr/bin/env bash
# Run the bounded Phase 22 multimodal and JSON routine gate.
set -euo pipefail

readonly MODE="${1:-no-mtp}"
readonly PORT=18075
readonly MODEL="qwen38-phase22"
readonly ROOT="/mnt/disk2/vllm-gfx906-build/phase-22"
readonly RESULT_DIR="${ROOT}/results/$(date -u +%Y%m%dT%H%M%SZ)-${MODE}-gates"
mkdir -p "${RESULT_DIR}"

if ! curl --fail --silent --max-time 10 "http://127.0.0.1:${PORT}/health" >/dev/null; then
    echo "Phase 22 server is not healthy on port ${PORT}." >&2
    exit 1
fi

python3 - "${RESULT_DIR}" <<'PY'
import sys
from pathlib import Path
from PIL import Image, ImageDraw

root = Path(sys.argv[1])
for name, color, label in (("one", "#1967d2", "ONE"), ("two", "#d25b19", "TWO")):
    image = Image.new("RGB", (256, 256), color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((32, 32, 224, 224), outline="white", width=8)
    draw.text((88, 112), label, fill="white")
    image.save(root / f"{name}.jpg", quality=92)
PY

image_url() {
    local file="$1"
    printf 'data:image/jpeg;base64,%s' "$(base64 -w0 "${file}")"
}

post_case() {
    local name="$1"
    local payload="$2"
    local output="${RESULT_DIR}/${name}.json"
    local code
    code="$(curl --silent --show-error --output "${output}" --write-out '%{http_code}' \
        --max-time 900 -H 'content-type: application/json' \
        --data "${payload}" "http://127.0.0.1:${PORT}/v1/chat/completions")"
    [[ "${code}" == "200" ]]
    jq -e '.choices[0].message.content | strings | length > 0' "${output}" >/dev/null
    jq -n --arg case "${name}" --arg http_status "${code}" \
        --arg response_file "${output}" --arg completed_at "$(date --iso-8601=seconds)" \
        '{case: $case, http_status: $http_status, response_file: $response_file,
          completed_at: $completed_at}' >> "${RESULT_DIR}/gates.jsonl"
}

post_case text "$(jq -nc --arg model "${MODEL}" \
    '{model: $model, temperature: 0, max_tokens: 64,
      messages: [{role: "user", content: "Reply exactly: phase 22 text healthy."}]}')"

one_url="$(image_url "${RESULT_DIR}/one.jpg")"
two_url="$(image_url "${RESULT_DIR}/two.jpg")"
post_case image_1 "$(jq -nc --arg model "${MODEL}" --arg image "${one_url}" \
    '{model: $model, temperature: 0, max_tokens: 64,
      messages: [{role: "user", content: [{type: "text", text: "Name the dominant image color in one word."}, {type: "image_url", image_url: {url: $image}}]}]}')"
post_case image_2 "$(jq -nc --arg model "${MODEL}" --arg one "${one_url}" --arg two "${two_url}" \
    '{model: $model, temperature: 0, max_tokens: 64,
      messages: [{role: "user", content: [{type: "text", text: "State the two dominant colors in order, separated by a comma."}, {type: "image_url", image_url: {url: $one}}, {type: "image_url", image_url: {url: $two}}]}]}')"

for index in 1 2 3; do
    post_case "json_${index}" "$(jq -nc --arg model "${MODEL}" \
        '{model: $model, temperature: 0, max_tokens: 32,
          response_format: {type: "json_object"},
          messages: [{role: "user", content: "Return exactly one JSON object with boolean key ok set to true."}]}')"
    jq -er '.choices[0].message.content | fromjson | select(.ok == true)' \
        "${RESULT_DIR}/json_${index}.json" >/dev/null
done

curl --fail --silent "http://127.0.0.1:${PORT}/v1/models" \
    > "${RESULT_DIR}/models.json"
curl --fail --silent "http://127.0.0.1:${PORT}/metrics" \
    > "${RESULT_DIR}/metrics.prom"
echo "${RESULT_DIR}"
