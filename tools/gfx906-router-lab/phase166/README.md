# Phase 166 global FIFO Router lab

This directory contains the reproducible, CPU-first experiment for strict
global FIFO admission in front of four independent vLLM workers. It does not
contain model weights, business request bodies, generated output, caches, or
production deployment files.

The source lock pins vLLM Router `0.1.14` at commit
`b93cbcb62702a9e8bb1bc170598cfce20aaadbbf`. The patch series adds an opt-in
`global_fifo` policy with these production-shaped defaults:

- eight dispatched requests per worker and 32 across four workers;
- 96 Router-waiting requests and 128 total admitted requests;
- 1 GiB of queued request bodies;
- a 43,200-second queue timeout;
- OpenAI-shaped `429` plus `Retry-After: 30` at queue limits;
- OpenAI-shaped `408` at queue timeout.

The Router-local lifecycle is authoritative. Worker metrics are observability
only and never decide admission. A lease is released by normal completion,
HTTP or transport failure, timeout, streaming completion, client disconnect,
or cancellation. A retry releases its first lease and rejoins the FIFO tail,
excluding the failed worker when another healthy worker is available.

A health transition blocks future dispatches to that worker, but does not
force-release an active lease: the backend request may still be executing.
That lease is released by its HTTP lifecycle or timeout. This prevents a brief
health-check failure and recovery from overcommitting the worker.

## Build and CPU contract

```bash
./tools/gfx906-router-lab/phase166/build.sh
.venv/bin/python tools/gfx906-router-lab/phase166/contract_global_fifo.py \
  --image local/vllm-router:0.1.14-global-fifo-phase166
```

The contract uses four localhost mock workers. It validates burst limits,
FIFO, cancellation, timeout, request and byte limits, streaming disconnect,
retry, worker loss/recovery, and final zero queue/slot invariants. It never
connects to production workers.

## Isolated GPU Router

After the CPU contract passes, one candidate may be attached beside the hot
workers during an approved maintenance window:

```bash
./tools/gfx906-router-lab/phase166/run-router.sh \
  phase166-global-fifo \
  local/vllm-router:0.1.14-global-fifo-phase166 \
  global_fifo 8003 29001
```

The helper only starts a sidecar Router. It never stops or replaces port
`8002`. Raw benchmark payloads and outputs must remain outside Git.

## A/B stages

The runner refuses to start while the production Router is running and
requires the explicit maintenance assertion. Run each stage separately and
evaluate its scorecard before advancing:

```bash
.venv/bin/python tools/gfx906-router-lab/phase166/run_phase166_ab.py \
  --stage quick \
  --maintenance-ready \
  --output-dir /mnt/disk2/vllm-gfx906-build/phase166-router/results/quick \
  --replay-subset /path/to/private/phase166-subset40.jsonl \
  --replay-full /path/to/private/manifest.jsonl

.venv/bin/python tools/gfx906-router-lab/phase166/summarize_phase166_ab.py \
  /mnt/disk2/vllm-gfx906-build/phase166-router/results/quick \
  --stage quick \
  --json /mnt/disk2/vllm-gfx906-build/phase166-router/results/quick-gates.json \
  --markdown /mnt/disk2/vllm-gfx906-build/phase166-router/results/quick-gates.md
```

Replace `quick` with `subset` and then `full` only when the prior command exits
successfully. The private replay paths are deliberately not fixed in tracked
files.

## Promotion boundary

Phase 166 always restores the pinned production round-robin Router. Even a
passing result only makes Phase 167 eligible; replacing production requires a
second explicit approval.
