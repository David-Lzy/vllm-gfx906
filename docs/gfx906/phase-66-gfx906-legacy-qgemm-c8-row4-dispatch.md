# Phase 66: gfx906 legacy QGEMM exact-M8 row-4 dispatch

## Scope

Phase 65 established a repeatable C8 benefit from using four rows per legacy
QGEMM block, but the global row-4 build also moved non-C8 behavior. This phase
keeps the retained row-8 implementation as the common path and dispatches
row-4 only for exact `M=8` GPTQ blocks. It was run on one development MI50;
production Router, workers, port 8002, model cache, and compose files were not
modified.

The implementation is commit
`ce81ce80b12899f1b9a71eb185d4cc24540380f3` on
`perf/gfx906-legacy-qgemm-c8-row4-dispatch`. Its gfx906-only CMake option,
`VLLM_GFX906_LEGACY_QGEMM_C8_ROWS_PER_BLOCK`, accepts `0` or `4` and defaults
to `0`. With the option set to `4`, an M=8 launch runs two row-4 chunks; M=1
through M=7 retain normal row-8 chunking and capacity.

## Repeated service result

Two independent service A/B runs compared the Phase 44 legacy row-8 image with
the mixed-dispatch image. Both used isolated scratch compile/Triton caches, the
same Qwen3.5 9B AWQ checkpoint, 100K context, eight sequences, 32K batched
tokens, float16 KV cache, and the routine text plus one/two 256-square image
workload.

| Scenario | Run A change | Run B change | Mean change |
| --- | ---: | ---: | ---: |
| Text C1 | +0.26% | +2.18% | +1.22% |
| Text C8 | +3.07% | +2.64% | +2.85% |
| One 256px image C1 | -1.44% | -1.19% | -1.32% |
| Two 256px images C1 | +0.66% | -0.87% | -0.10% |

The C8 controls/candidates were `227.43 -> 234.41 tok/s` in Run A and
`226.73 -> 232.71 tok/s` in Run B. Every run passed health, model discovery,
routine text/image smoke, JSON constrained output 3/3, and post-test
`running=0`/`waiting=0`. Error scans were clear of OOM, HTTP 5xx, traceback,
xgrammar/FSM, RCCL, and NCCL fatal signatures.

## Disposition

**retained-targeted, default off.** A stable two-to-three percent C8 gain is
valuable on gfx906 and is retained with its narrow guard. It is not the default
release configuration because one-image C1 regressed around 1.3 percent on
average. Use it only for a deployment that explicitly prioritizes sustained
eight-request decode throughput; otherwise retain the validated row-8 default.

Raw results remain outside Git in the configured local build root. The first
run is `20260823T-phase66-hybrid-c8-ab`; the independent repeat is
`20260823T-phase66-hybrid-c8-ab-repeat`.
