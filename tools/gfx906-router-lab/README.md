# gfx906 Router lab

This directory contains the reproducible vLLM Router experiment used for the
Qwen3.5 TP1x4 queue-skew evaluation. It does not contain model weights,
business payloads, runtime caches, or production deployment files.

The source lock in `router-source.env` pins upstream Router `0.1.14` at commit
`b93cbcb62702a9e8bb1bc170598cfce20aaadbbf`. The patch series adds two opt-in
policies and pins the upstream Dockerfile's Rust/Python base images by digest:

- `least_inflight`: choose the healthy worker with the fewest requests tracked
  by this Router process, with round-robin tie breaking.
- `queue_aware`: poll worker Prometheus endpoints and minimize the tuple
  `(waiting, max(running + waiting, local_inflight))`.

Queue telemetry defaults to a 500 ms poll interval, a 200 ms request timeout,
and a 2,000 ms stale threshold. Partial or stale telemetry falls back to
`least_inflight`; a Router that has never observed any telemetry falls back to
`round_robin`.
Request accounting is released on normal completion, HTTP and transport
errors, timeouts, streaming completion, and client disconnect. Health checks
do not reset active request counters. Dispatch latency uses dedicated
sub-millisecond through 10 ms Prometheus buckets so the 2 ms overhead gate is
measured directly.

## Build

```bash
./tools/gfx906-router-lab/build.sh
```

The default image is local-only:
`local/vllm-router:0.1.14-queue-aware-phase165`. Phase 165 did not pass its
performance and replay gates, so that image was deleted and Phase 166 was not
created. Rebuilding it is for evidence reproduction only, not production use.

## CPU-only contract test

```bash
python3 tools/gfx906-router-lab/smoke_queue_router.py \
  --image local/vllm-router:0.1.14-queue-aware-phase165
```

The test starts four localhost mock workers and an isolated Router. It checks
queue selection, local in-flight lifecycle, streaming completion, client
disconnect, HTTP errors, telemetry fallback, and timeout cleanup. It never
connects to the production endpoint.

## Isolated Router and benchmark

Start one policy beside production:

```bash
./tools/gfx906-router-lab/run-router.sh \
  phase165-router-queue \
  local/vllm-router:0.1.14-queue-aware-phase165 \
  queue_aware 8003 29001
```

Run one deterministic homogeneous text/one-image/two-image control while
sampling all four workers every 500 ms:

```bash
python3 tools/gfx906-router-lab/benchmark_queue_router.py \
  --arm queue-aware-c16 \
  --endpoint http://127.0.0.1:8003 \
  --router-metrics http://127.0.0.1:29001/metrics \
  --router-container phase165-router-queue \
  --worker-metrics http://172.27.0.5:8000/metrics \
  --worker-metrics http://172.27.0.2:8000/metrics \
  --worker-metrics http://172.27.0.3:8000/metrics \
  --worker-metrics http://172.27.0.4:8000/metrics \
  --output-dir /path/outside/git/results \
  --concurrency 16 \
  --fixed-class image1
```

Worker addresses are examples and must be discovered again before every run.
Use `--replay /path/to/replay.jsonl` for the private Phase1 corpus. Replay
records may be direct OpenAI request bodies or objects with `id`, `class`, and
`body` fields. A frozen manifest may instead use `payload_relpath` and
`payload_sha256`; the loader confines paths to the manifest directory and
verifies every payload before sending its original JSON bytes. Streaming
replays are rejected, and the model ID is preserved unless `--model` is
explicitly supplied. Request bodies and model output are never copied into the
result files.

During an approved maintenance window, `run_phase165_ab.py` executes the
rotation-ordered four-policy matrix. The fixed matrix runs homogeneous text,
one-image, and two-image arms at C16 and C32; worker Prometheus histograms
provide TTFT bounds without changing the original non-streaming request
shape. Before touching a Router it verifies exactly 120 replay payloads,
including path confinement, request-index uniqueness, JSON shape, streaming
mode, and every payload SHA-256. Use `--preflight-only` while production is
still running, then the maintenance run repeats the same validation. The
small fixed controls retain a 900-second client timeout; the real replay uses
3,600 seconds so historically valid Phase1 long-tail requests are not counted
as Router failures. The runner deliberately refuses to run while the
production Router container is active and requires `--maintenance-ready`.
It does not stop production, start the fallback, or restore production; those
operations remain outside the benchmark process. `summarize_phase165_ab.py`
aggregates all three rounds and evaluates the Phase 166 promotion gates.

## Production boundary

Phase 165 was a side-by-side evaluation and is now archived as rejected. It did
not edit the production Compose file or replace port 8002. The current worker
model, model ID, worker arguments, and OpenAI request/response formats remain
unchanged.
