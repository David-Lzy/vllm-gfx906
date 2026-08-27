# Phase 68: gfx906 QGEMM exact-M8 call attribution

## Scope

Phase 66 retained an exact-M8 row-4 dispatch because it improved the Qwen3.5
9B AWQ C8 decode benchmark by 2.85 percent across two paired runs. Its small
one-image C1 movement was initially treated conservatively because the call
shapes had not been attributed. This phase traced those shapes on development
GPU2 only. Production Router, workers, port 8002, model cache, and compose
files were not changed.

The development trace is on
`perf/gfx906-qgemm-m8-call-attribution`. It adds a default-off gfx906 legacy
QGEMM host trace, first in `3605bc2ffc`, with diagnostic updates in
`cef47f828c`, `b9941e83fd`, and `55529cddfe`. The trace image is disposable
and is not a performance candidate.

## Method

The service used the Phase 44 row-8 behavior, Qwen3.5 9B AWQ, 100K context,
eight sequences, 32K batched tokens, float16 KV cache, and the standard
single-MI50 development boundary.

Initial graph-mode samples established an important instrumentation limit:
the QGEMM host dispatcher runs while vLLM compiles and captures graphs, while
steady-state requests replay captured GPU graphs and do not re-enter that host
function. Those samples passed health and the request gates, but cannot
attribute request-time shapes.

The decisive sample used `--enforce-eager` only for diagnosis. It replicated
the Phase 66 text payload: 30 prompt tokens, `min_tokens=max_tokens=64`, one
text C1 request, one 256px image C1 request, then eight simultaneous text
requests. Eager mode changes observability, not the release setting.

## Request-time shape evidence

The marked text C1 and one-image C1 intervals recorded **zero** exact-M8
legacy-QGEMM calls. The marked C8 interval recorded only these five shapes:

| N | K | Groups | Calls in marked C8 interval |
| ---: | ---: | ---: | ---: |
| 10,240 | 4,096 | 128 | 496 |
| 12,288 | 4,096 | 128 | 1,488 |
| 24,576 | 4,096 | 128 | 1,984 |
| 4,096 | 12,288 | 384 | 1,984 |
| 4,096 | 4,096 | 128 | 1,984 |

All calls used 4-bit GPTQ v2. Every trace request returned a non-empty 200
response; the text requests generated 64 completion tokens, the image request
returned a correct short description, error scans were clean, and post-test
metrics recorded `running=0` and `waiting=0`.

## Disposition

**retained-targeted; default enabled for the Qwen3.5 9B AWQ gfx906 C8 release
profile.** No more specific shape filter is useful: exact `M=8` is already
the effective isolation boundary. The Phase 66 one-image C1 movement cannot
be caused by this dispatch because that workload does not issue exact-M8
QGEMM calls; treat it as benchmark variation unless a separate, reproducible
non-QGEMM cause is found.

The generic CMake option remains default-off so the fork does not silently
change unrelated models or hardware. The supported Qwen3.5 9B AWQ gfx906 C8
release profile should build with
`VLLM_GFX906_LEGACY_QGEMM_C8_ROWS_PER_BLOCK=4`. Promotion to the production
service still requires the separate canary approval and release gate.

Raw evidence remains outside Git under the configured local build root. The
diagnostic runs are named `phase68-m8-attribution`, `phase68-m8-attribution-
flush`, `phase68-m8-attribution-long`, and `phase68-m8-attribution-eager`.
