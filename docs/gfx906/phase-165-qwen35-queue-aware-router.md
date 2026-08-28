# Phase 165: Qwen3.5 TP1x4 Queue-Aware Router

## Status

Completed and rejected for production. The Router implementation and CPU
contract passed, but the real-GPU A/B failed both a homogeneous-throughput gate
and the production replay timeout gate. Production stayed on the pinned
round-robin Router; Phase 166 was not created.

## Problem

The selected production topology uses four independent TP1 workers behind
vLLM Router 0.1.14. Round-robin distributes request counts evenly, but it does
not account for large differences in image count, prompt length, output budget,
or time already spent on a worker. Live 500 ms sampling found intervals where
one worker had queued requests while another worker still had free sequence
capacity.

A five-minute passive sample observed this condition in 600 of 600 usable
samples. One worker accumulated approximately 1,496 queue-seconds while the
other three accumulated none. This establishes a persistent scheduling problem,
but is not itself evidence that a new policy improves end-to-end performance.

## Implementation

The experiment is a patch series over vLLM Router 0.1.14 commit `b93cbcb`. It
adds two opt-in policies while preserving request forwarding, streaming,
health checks, circuit breaking, timeouts, and the OpenAI-compatible API:

- `least_inflight` selects the healthy worker with the smallest Router-local
  active request count and uses round-robin for ties.
- `queue_aware` polls each worker's `/metrics` endpoint every 500 ms, with a
  200 ms timeout and a 2,000 ms stale threshold. It minimizes
  `(waiting, effective_depth)`, where
  `effective_depth=max(running+waiting, local_inflight)`.

Stale or partially available telemetry falls back to `least_inflight`.
Complete lack of observed telemetry falls back to round-robin. Health checks no
longer reset active request counts. An owned response guard releases each local
count exactly once after normal completion, HTTP or transport error, timeout,
stream completion, or client disconnect.

The patch also exports per-worker observed running and waiting counts, local
in-flight and effective depth, telemetry age, policy selections, fallback
counts, worker request duration, and a dedicated Router dispatch histogram with
sub-2 ms buckets.

## Reproducibility

The source lock, patch series, build entry, CPU contract, benchmark runner, and
scorecard generator are under `tools/gfx906-router-lab/`. Model weights,
business requests, caches, and machine deployment files are not stored in Git.

Pinned artifacts:

- upstream Router commit: `b93cbcb62702a9e8bb1bc170598cfce20aaadbbf`;
- policy patch SHA-256:
  `afbe20b66526fbdbcb2e07807bed7147ca5dfcb40514d160090fbc65ee45739d`;
- dispatch-metric patch SHA-256:
  `feeb5065939f9046e40b8b5fac504a5197c500257030f5bf0eb50b9a993645e3`;
- local experiment image ID:
  `sha256:d53f5cbb43da2aedacd27911ecd9506579501c3114e01b795d650db7a40cc6e9`.

All 488 Router library tests and the workspace integration targets passed. The
final-image contract covered queue choice, health-check preservation, normal
and aborted streams, HTTP errors, timeout cleanup, client disconnect cleanup,
and telemetry fallback.

The benchmark initially raised before persisting a failed warmup. The runner
now writes warmup requests, 500 ms metric samples, and the summary before
enforcing the failure gate. A focused injected-failure check verified that all
three artifacts survive the exception path.

## Benchmark Contract

The measured matrix compares official round-robin, patched round-robin,
`least_inflight`, and `queue_aware`. It uses one complete warmup plus three
rotation-ordered measured rounds for each of these workloads:

- text, one 256-square image, and two 256-square images at C16 and C32;
- a frozen 120-request production-shaped replay at C32;
- synchronized Router and four-worker metrics every 500 ms.

The replay preserves exact request bytes and hashes outside Git. The benchmark
records request and completion throughput, makespan, p50/p95/p99, bounded TTFT,
queue-seconds, idle-while-queued, Router CPU/RSS, dispatch latency, errors,
duplicates, and residual counters.

Promotion eligibility requires all of the following:

- idle-while-queued improves by at least 50% and is no greater than 25%;
- replay p95 or p99 improves by at least 10%;
- replay throughput improves by at least 5% or makespan falls by at least 5%;
- no homogeneous workload loses more than 3% throughput;
- Router dispatch p99 adds no more than 2 ms;
- no failed, empty, duplicate, truncated, OOM, FSM, xgrammar, RCCL/NCCL fatal,
  or residual queued request is observed.

## Results

### Homogeneous controls

All four policies completed three rotation-ordered rounds for text, one-image,
and two-image requests at C16 and C32. There were no request failures, duplicate
backend requests, or residual worker/Router counters. Patched round-robin was
within 1.93% of the official image in every case, so the experimental build did
not introduce a material general forwarding regression.

The table reports median request throughput and the change from official
round-robin:

| Scenario | Official RR req/s | Patched RR | Least in-flight | Queue-aware |
| --- | ---: | ---: | ---: | ---: |
| Text C16 | 33.6239 | 33.6277 (+0.01%) | 33.6054 (-0.06%) | 30.1234 (-10.41%) |
| One image C16 | 9.8467 | 9.6569 (-1.93%) | 10.1767 (+3.35%) | 8.6863 (-11.78%) |
| Two images C16 | 7.9829 | 7.9628 (-0.25%) | 8.3537 (+4.65%) | 7.5161 (-5.85%) |
| Text C32 | 45.8334 | 45.6400 (-0.42%) | 45.5722 (-0.57%) | 45.5687 (-0.58%) |
| One image C32 | 12.5037 | 12.4946 (-0.07%) | 12.5158 (+0.10%) | 12.5680 (+0.51%) |
| Two images C32 | 10.1743 | 10.0087 (-1.63%) | 10.1270 (-0.47%) | 9.8978 (-2.72%) |

Queue-aware therefore failed the fixed-load gate before the production replay:
three C16 classes regressed by more than the allowed 3%. The 500 ms worker
snapshot is too stale for these short requests and can temporarily outweigh the
more current Router-local in-flight count. Least in-flight did not show this
failure and improved both C16 image controls, but it did not complete the core
production replay and is not a production candidate from this phase.

### Production replay

The exact 120-request replay retained its 32,768-token budgets and was run at
C32. Official round-robin's complete warmup produced the following result:

| Metric | Result |
| --- | ---: |
| Success / timeout | 104 / 16 |
| Makespan | 6,216.66 s |
| p50 / p95 / p99 | 464.16 / 3,600.08 / 3,600.10 s |
| Completion throughput | 45.94 tok/s |
| Idle while queued | 54.48% |
| Per-worker queue-seconds | 174.37 / 1.00 / 7,714.48 / 2,069.01 |
| Per-worker maximum waiting | 2 / 1 / 6 / 4 |

Round-robin dispatched exactly 30 requests to each worker, yet one worker
accumulated more than 7,700 queue-seconds while other workers had free sequence
capacity. This confirms that equal request counts are not equal work. It also
shows a second, independent limitation: 16 requests exceeded the one-hour
request timeout because the replay contains very large prompts and extremely
long generations. A Router can improve admission but cannot migrate or
parallelize a request after dispatch.

The timeout made this core workload ineligible, and the homogeneous regression
had already made the queue-aware candidate mathematically unable to pass. The
remaining replay arms were therefore not run merely to produce a losing
scorecard while production traffic waited.

### Restoration

The cleanup trap restored the exact production Router digest. All four vLLM
workers remained healthy and were not restarted. Health, model discovery, text,
and two-image smoke tests passed; all worker running/waiting counters returned
to zero; recent logs contained no OOM, RCCL/NCCL fatal, FSM, xgrammar, or
traceback match. The maintenance fallback was stopped, the benchmark-isolated
CPU workload was restored, and the paused Phase1 producer resumed from its
unchanged manifest.

After preserving the compact evidence, the rejected local Router image and
approximately 12 GiB of Rust/Cargo build output were deleted. The phase-local
evidence root is approximately 57 MiB; model weights and production caches were
not touched.

Compact local evidence hashes:

- frozen replay manifest: `3a443fae0ff20551c685d49f7dd15398aa5f309f48e8d8c0381eb16df83386ed`;
- fixed-load partial scorecard JSON:
  `38323fb6f6039e187db230cedf410c5a59aa673933c61b197c8c2dd6b07c661a`;
- reconstructed failed-warmup summary:
  `320e0889000bf50b45cdca7c7717ad3dca4f6d830fb951bd87b214dacfc71c1d`.

## Decision

Do not promote this queue-aware policy. Retain the implementation and evidence
as a reproducible negative result. A later, separately authorized Phase 166
tested strict Router-owned FIFO admission without the rejected 500 ms metrics
polling; see the Phase 166 report for its independent result.
