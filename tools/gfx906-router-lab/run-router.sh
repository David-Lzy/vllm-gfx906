#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run-router.sh NAME IMAGE POLICY API_PORT METRICS_PORT

Starts an isolated Router on the existing worker network. This helper never
stops or edits the production Router.
EOF
}

if [[ $# -ne 5 ]]; then
  usage >&2
  exit 2
fi

name=$1
image=$2
policy=$3
api_port=$4
metrics_port=$5
network=${ROUTER_LAB_NETWORK:-qwen35-9b-router-net}
served_model=${ROUTER_LAB_SERVED_MODEL:-cyankiwi/Qwen3.5-9B-AWQ-4bit}

case "$policy" in
  round_robin|least_inflight|queue_aware) ;;
  *)
    echo "unsupported policy: $policy" >&2
    exit 2
    ;;
esac

if docker container inspect "$name" >/dev/null 2>&1; then
  docker rm -f "$name" >/dev/null
fi

args=(
  vllm-router
  --host 0.0.0.0
  --port 8000
  --worker-urls
  http://vllm-gpu0:8000
  http://vllm-gpu1:8000
  http://vllm-gpu2:8000
  http://vllm-gpu3:8000
  --policy "$policy"
  --worker-startup-timeout-secs 1800
  --worker-startup-check-interval 10
  --max-payload-size 1073741824
  --max-concurrent-requests 128
  --request-timeout-secs 7500
  --retry-max-retries 1
  --health-check-interval-secs 10
  --health-check-timeout-secs 5
  --prometheus-host 0.0.0.0
  --prometheus-port 29000
)

if [[ "$policy" == queue_aware ]]; then
  args+=(
    --queue-metrics-interval-ms 500
    --queue-metrics-timeout-ms 200
    --queue-metrics-stale-ms 2000
  )
fi

docker run --detach --rm \
  --name "$name" \
  --network "$network" \
  --publish "127.0.0.1:${api_port}:8000" \
  --publish "127.0.0.1:${metrics_port}:29000" \
  "$image" "${args[@]}" >/dev/null

deadline=$((SECONDS + 60))
until curl --fail --silent --show-error --max-time 3 \
  "http://127.0.0.1:${api_port}/health" >/dev/null; do
  if (( SECONDS >= deadline )); then
    docker logs --tail 100 "$name" >&2 || true
    exit 1
  fi
  sleep 1
done

curl --fail --silent --show-error --max-time 5 \
  "http://127.0.0.1:${api_port}/v1/models" |
  jq -e --arg model "$served_model" '.data[] | select(.id == $model)' >/dev/null

printf '%s\n' \
  "name=$name" \
  "image=$image" \
  "policy=$policy" \
  "api=http://127.0.0.1:${api_port}" \
  "metrics=http://127.0.0.1:${metrics_port}/metrics"
