# Phase 67: gfx906 legacy QGEMM exact-M8 row-2 sweep

## Scope

This phase tested whether the Phase 66 exact-M8 row-4 dispatch could be
improved by splitting each M=8 legacy GPTQ launch into four row-2 chunks. The
candidate was isolated to gfx906 and exact M=8; every other shape continued to
use row-8 behavior. Production Router, workers, port 8002, model cache, and
compose files were not modified.

Commit `e5997a5ac3` on
`perf/gfx906-legacy-qgemm-c8-row2-sweep` expands the CMake selector to accept
`0`, `2`, or `4`. It built successfully on the existing Phase 44 legacy QGEMM
base image and was compared with the retained Phase 66 row-4 image on one
development MI50.

## Result

The service passed health, model discovery, text, one/two 256-square images,
JSON constrained output 3/3, and post-test `running=0`/`waiting=0`. Error
scans found no OOM, HTTP 5xx, traceback, xgrammar/FSM, RCCL, or NCCL fatal
signature. It nevertheless failed its sole performance gate:

| Scenario | Row-2 change versus row-4 |
| --- | ---: |
| Text C1 | -3.29% |
| Text C8 | -11.92% |
| One 256px image C1 | +1.90% |
| Two 256px images C1 | +1.18% |

The target C8 median was `236.87 -> 208.64 tok/s`. The extra launch/chunk
overhead dominates the reduced row workset on gfx906.

## Disposition

**Rejected.** The source branch and raw results remain available for
attribution, but row-2 is not carried into the retained build options and its
temporary image is removed after this report is recorded. Exact-M8 row-4
remains the retained C8-prioritized option.
