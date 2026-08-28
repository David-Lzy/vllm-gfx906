# Phase 166: Qwen3.5 TP1x4 Global FIFO Admission Router

## Status

Implementation retained; production canary not authorized. The strict global
FIFO Router passed its complete CPU contract and the fixed GPU quick matrix,
but the real-workload control failed before any candidate arm ran. Production
was restored to the pinned round-robin Router, so Phase 167 was not created.

## Problem

The production topology has four independent TP1 workers with eight sequence
slots each. Round-robin balances request counts, not work. A request may range
from text-only to dozens of images and may generate anywhere from a few hundred
to 32,768 tokens. Once round-robin has sent a request to a worker, that worker
can build a local queue while another worker still has free sequence slots.

Phase 165 showed this imbalance but rejected a 500 ms metrics-backed scheduler:
the snapshots were too stale for short requests. Phase 166 therefore moves
admission to the Router and treats Router-owned request lifetimes as the
authoritative scheduling state. Worker metrics are observational only.

## Implementation

The patch is based on vLLM Router 0.1.14 commit
`b93cbcb62702a9e8bb1bc170598cfce20aaadbbf`. It adds the opt-in
`global_fifo` policy:

- generation requests enter a model-isolated FIFO;
- each healthy worker has eight synchronously reserved slots;
- the global dispatched limit is 32 and the Router waiting limit is 96;
- queued request bodies are capped at 1 GiB and wait for at most 43,200 seconds;
- available workers are ranked by Router-local active requests, with a
  round-robin tie break;
- an owned admission lease is released exactly once on normal completion,
  HTTP or transport error, timeout, stream completion, client disconnect, or
  cancellation;
- one backend retry releases the old lease, rejoins the FIFO tail, and avoids
  the failed worker when another healthy worker exists;
- generation requests bypass the legacy token-bucket queue while this policy
  is active, avoiding double admission;
- full or byte-limited queues return OpenAI-shaped HTTP 429 with
  `Retry-After: 30`; admission timeout returns HTTP 408.

Health loss blocks new assignments. It does not forcibly release an active
lease because the backend request may still be running; its normal HTTP
lifecycle or timeout owns that release. This avoids transient health recovery
overcommitting a worker.

The Router exports queue depth and bytes, oldest wait, wait latency, worker
slots, dispatches, retries, cancellation, timeout, rejection, invariant, and
worker-duration metrics. Control endpoints remain outside admission, and the
OpenAI request and response formats are unchanged.

## Reproducibility

The source lock, patch, build entry, deterministic CPU contract, benchmark
runner, and scorecard generator are under `tools/gfx906-router-lab/phase166/`.
Business payloads, responses, model weights, caches, and build output remain
outside Git.

Pinned artifacts:

- Router source commit: `b93cbcb62702a9e8bb1bc170598cfce20aaadbbf`;
- patch SHA-256:
  `ea3542c4fb3b3bd601d606bbb112ae5fdd819defdd0d772adbe72a4fd97926c8`;
- experiment image ID:
  `sha256:99a7cd86dda6472ee5271166582d92a6d73aaaf615951e5cefe7334a18620ac6`.

All 492 Router unit tests, 47 API endpoint tests, integration targets,
`cargo fmt`, and `cargo clippy` passed. The HTTP CPU contract covered bursts of
40, 64, and 128 requests, strict FIFO, cancellation, queue and byte limits,
timeout, stream disconnect, retry, worker loss, and recovery. No worker
exceeded eight slots; the 128-request burst reached exactly 32 dispatched and
96 queued requests, then returned to zero without an invariant violation.

## GPU Quick Matrix

The quick matrix compared the pinned official round-robin image, patched
round-robin, Phase 165 least-inflight, and global FIFO. Text, one-image, and
two-image requests ran at C16, C32, C40, and C64 with one warmup and three
rotation-ordered measured rounds.

All 12 global-FIFO scenarios passed:

- no request failure, duplicate backend request, residual slot, queue, or
  worker request;
- admission backlog with a free slot: 0%;
- idle-while-queued reduction versus official round-robin: 100%;
- worker-local queue-seconds reduction: 100%;
- worst fixed C16/C32 throughput change: -2.24%, inside the -3% gate;
- worst measured Router dispatch p99 addition: 0.95 ms, inside the 2 ms gate.

Selected median results:

| Scenario | Official RR req/s | Global FIFO req/s | Change | RR idle while queued | FIFO idle while queued | RR worker queue-s | FIFO worker queue-s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Text C16 | 22.8133 | 28.7555 | +26.05% | 0% | 0% | 0.00 | 0.00 |
| One image C16 | 8.8461 | 8.9363 | +1.02% | 0% | 0% | 0.00 | 0.00 |
| Two images C16 | 7.3372 | 8.0197 | +9.30% | 0% | 0% | 0.00 | 0.00 |
| Text C32 | 38.4687 | 39.3130 | +2.19% | 0% | 0% | 0.00 | 0.00 |
| One image C32 | 12.0834 | 11.8123 | -2.24% | 0% | 0% | 0.00 | 0.00 |
| Two images C32 | 9.4562 | 9.5645 | +1.15% | 0% | 0% | 0.00 | 0.00 |
| Text C40 | 31.1709 | 30.5236 | -2.08% | 20.00% | 0% | 4.62 | 0.00 |
| Text C64 | 37.8817 | 40.0336 | +5.68% | 8.33% | 0% | 21.67 | 0.00 |
| Two images C64 | 9.9942 | 10.1816 | +1.88% | 11.11% | 0% | 74.90 | 0.00 |

These results validate the admission mechanism and the user's proposed
"keep more work at the Router and immediately fill a released worker slot"
design. They do not by themselves authorize production use.

## Real-Workload Gate

The next gate used a frozen 40-request subset of the production-shaped replay
at C40. It retained original payloads, zero to 48 images, and 32,768-token
budgets. The first official round-robin warmup produced:

| Metric | Result |
| --- | ---: |
| Success / failure | 39 / 1 |
| Makespan | 7,500.13 s |
| p50 / p95 / p99 | 823.76 / 4,184.20 / 7,138.56 s |
| Completion throughput | 31.60 tok/s |
| Idle while queued | 1.47% |
| Worker queue-seconds | 641.13 / 664.98 / 721.79 / 422.77 |
| Maximum worker waiting | 3 / 3 / 3 / 2 |

The failed request returned HTTP 500 with an empty body at 7,500.11 seconds,
matching the selected production Router's backend request timeout. All worker
running/waiting counters subsequently returned to zero. Other successful
requests included one complete 32,768-token generation at 6,573.05 seconds.

This is a control-arm failure, not evidence that global FIFO failed the real
workload. It also means there is no valid global-FIFO real-replay result. The
predeclared safety gate requires zero failures, so the runner stopped before
patched round-robin, least-inflight, global FIFO, or the full 120-request replay.
Changing the control timeout after observing the result would create a new
experiment rather than complete this one.

## Restoration

The experiment Router was removed and the exact pinned production Router
digest was restored without restarting the four hot vLLM workers. The Server2
Ollama fallback kept the public endpoint available during the handoff and was
then stopped together with its tunnel. Health, model discovery, text, and
two-image smoke completed successfully. The benchmark left no worker request
behind; Phase1's existing launcher and repair supervisor resumed their own
work after the production Router became reachable, so no duplicate producer
was started.

Recent worker logs contained no OOM, RCCL/NCCL fatal, FSM, xgrammar, or
traceback attributable to the experiment. Production remains on the original
round-robin digest and unchanged TP1x4 workers.

Compact evidence hashes:

- quick gate JSON:
  `a3197e13d7e81a39cd9e81f1d8d925d7f6a6952f76e836a5af4947fd4066df01`;
- subset gate JSON:
  `75dec0053e7e00f553ae0fc352287ef482f0ff5dd6b976b182c145f84841039f`;
- official-control warmup summary:
  `2d6340f7f99e4bf45a8267e86d0681f613c9086c2efe2856b026ee9d8f759d39`;
- official-control request result JSONL:
  `76fdcf5277bc0d217eb554e01b5a8c454ea037e3e8fbbd06b9e7c75faaf8ebfe`;
- compressed 200 ms metric stream:
  `e9b38454675f16306d8d9ee03a82d655c229cb1f05f57c2169f16c6fcb702576`.

The 641 MiB raw metric stream was reduced losslessly to 1.43 MiB with zstd.
The reproducible source/build tree and an accidental duplicate subset were
removed after hashing, recovering approximately 8 GiB. Compact Phase 166
evidence remains about 105 MiB.

## Decision

Retain the global-FIFO implementation and its positive quick-matrix evidence,
but do not promote it. Phase 167 is not created because the required real-load
comparison is incomplete and the zero-failure gate did not pass. A follow-up
may repeat only the real-load gate with a predeclared, identical backend
timeout above 7,500 seconds for every arm; that requires a new maintenance
decision and must not reinterpret this run after the fact.
