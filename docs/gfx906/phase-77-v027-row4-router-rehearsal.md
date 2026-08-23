# Phase 77: v0.27 Row-4 Router Rehearsal

## Goal

Validate the retained Phase 66 exact-M8 row-4 legacy-QGEMM selector in the
actual Qwen3.5 9B AWQ TP1x2 Router topology. Earlier Phase 66 evidence came
from a single-worker C8 path; this phase asks whether the narrow dispatch also
improves the two independent MI50 workers and Router combination used by a
future v0.27 canary.

## Method

The same test day used two isolated TP1 workers on development gfx906 devices
behind a private `power_of_two` Router. The production workers, production
Router, port 8002, Compose files, and model cache were not changed.

Both images used the same Qwen3.5 9B AWQ checkpoint and serving contract:

- 100K context, float16 KV cache, and 0.90 GPU-memory utilization;
- eight sequences and 32,768 batched tokens per worker;
- 64-image limit, no video, and a 16 GiB shared-memory multimodal cache; and
- chunked prefill, prefix caching, and the same text/one-image/two-image
  256-square fixture and JSON gate.

The control was the Phase 44 legacy-QGEMM image. The candidate was the Phase
66 image, whose only relevant difference is splitting exact `M=8` GPTQ blocks
into two four-row chunks while every other activation geometry keeps the
validated eight-row route. Each image had an independent compile/cache root.

## Results

All routine requests returned non-empty HTTP 200 results. Both variants passed
text, one-image, two-image, and JSON constrained output `3/3`.

| Router measurement | Phase 44 control | Phase 66 row-4 | Change |
| --- | ---: | ---: | ---: |
| C8 aggregate completion throughput | 248.86 tok/s | 252.84 tok/s | +1.6% |
| C16 aggregate completion throughput | 313.94 tok/s | 343.87 tok/s | +9.5% |
| C8 text C1 | 73.83 tok/s | 73.63 tok/s | -0.3% |
| C8 one-image C1 | 64.86 tok/s | 64.46 tok/s | -0.6% |
| C8 two-image C1 | 60.00 tok/s | 60.48 tok/s | +0.8% |
| C16 text C1 | 72.70 tok/s | 72.46 tok/s | -0.3% |
| C16 one-image C1 | 64.44 tok/s | 64.75 tok/s | +0.5% |
| C16 two-image C1 | 60.65 tok/s | 60.56 tok/s | -0.1% |

The candidate then completed a post-routine 40-batch C16 soak with fixed
256-token outputs: all 640 responses were HTTP 200 and non-empty. Its aggregate
completion throughput was 362.58 tok/s, with 11.34 s median and 14.73 s p95
batch latency. The benchmark summary retains the harness's historic
`text_c8` scenario label, but the actual runner invocation and 16 responses per
batch establish that this was a C16 test.

At the end of the soak both workers reported `running=0` and `waiting=0`.
Captured Router and worker logs contained no HTTP 5xx, OOM, traceback,
xgrammar/FSM, or RCCL/NCCL fatal signature. The private stack was removed
after artifact collection, and the retained v0.23 production endpoint was
healthy afterwards.

## Decision

**Retain the Phase 66 exact-M8 row-4 selector in the v0.27 gfx906 release
composition.** The same-day Router result is positive at C8 and materially
positive at C16 while all non-target C1 and multimodal gates remain within one
percent of the control.

This does not promote v0.27 to production. The previous real production-GPU
canary observed one C16/256 HTTP 500; a later isolated 120-batch reproduction
did not reproduce it, but it did not identify the original event. A new
production canary still requires separate approval and the evidence-preserving
soak harness.

Raw build, HTTP status, metrics, and log artifacts remain outside Git in the
Phase 77 build-result roots. No model weights, cache content, or machine-local
paths are stored in this document.
