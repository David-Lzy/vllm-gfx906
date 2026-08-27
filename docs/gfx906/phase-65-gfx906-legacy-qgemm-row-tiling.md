# Phase 65: gfx906 legacy QGEMM row tiling

## Scope

This phase evaluates one narrow launch-layout alternative inside the retained
gfx906 legacy W4A16 QGEMM composition. The production service, Router, model
cache, and port 8002 were not modified. The comparison used one development
MI50 with Qwen3.5 9B AWQ, 100K context, eight sequences, 32K batched tokens,
float16 KV cache, and the ordinary text plus one/two 256-square image gates.

The established legacy kernel groups eight activation rows per quantized GEMM
block. Commit `4af0965a057e5682e84014f52bb040b9f0558fe9` adds a gfx906-only
build option to use four rows. The smaller block reduces live accumulator and
shared activation storage, but requires additional row chunks. The CMake guard
only accepts `4` or `8`, and the validated row-8 behavior remains the default.

## Microbenchmark screen

Actual Qwen3.5 AWQ packed weights were exercised at M=1, M=8, and M=27. M=8
showed substantial wins for some QGEMM projections, including output and MLP
down projections, but losses in QKV and fused gate-up projections. This mixed
result required a full-service test rather than a kernel-only decision.

## Repeated service result

Two independent row-8 versus row-4 service A/B runs used separate scratch
compile/Triton caches. Each run passed health, model discovery, text, one-image,
two-image, C8, JSON-object 3/3, and post-test zero running/waiting checks. No
OOM, HTTP 5xx, traceback, xgrammar/FSM, RCCL, or NCCL fatal signature appeared.

| Scenario | Run A row-4 change | Run B row-4 change | Interpretation |
| --- | ---: | ---: | --- |
| Text C1 | -2.83% | -0.06% | No repeatable single-request benefit |
| Text C8 | +2.59% | +2.45% | Repeated positive; mean +2.52% |
| One 256px image C1 | +0.40% | +0.96% | Small positive, below a separate promotion claim |
| Two 256px images C1 | -1.25% | -0.54% | Small regression outside the C8 target |

The control/candidate C8 medians were `227.33 -> 233.21 tok/s` in Run A and
`227.17 -> 232.74 tok/s` in Run B. The numerical output checks remained within
the existing GPTQ tolerance.

## Disposition

**retained-targeted, default off.** The row-4 build is a reproducible C8
throughput optimization worth retaining even though it is not a universal
single-request improvement. It should only be considered when the deployment
prioritizes sustained eight-request decode throughput and can tolerate the
small non-target trade-offs above. It is not a production promotion by itself.

The experiment image is intentionally not a release image. Its Dockerfile and
the CMake guard make the alternative reproducible from the linked source
commit. Raw artifacts remain outside Git in the configured local build root.
