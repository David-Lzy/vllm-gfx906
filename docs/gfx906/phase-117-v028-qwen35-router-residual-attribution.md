# v0.28 Qwen3.5 9B Router Residual Attribution

## Purpose

The v0.28 real Router canary was functionally clean but measured a small
throughput deficit relative to the retained v0.27 production image. This
follow-up separated a real source regression from Router allocation, device
state, and normal host timing variation before opening another kernel project.

## Method

The control read the two idle v0.27 production workers through their private
Compose-network endpoints without changing the Router or production settings.
The candidate ran the identical Qwen3.5 9B AWQ checkpoint and fixed
128-token fixture on development GPU2. It used an A-B-A order: v0.27 on GPU2,
then three independent warmed v0.28 C1/C4 repetitions on the same GPU.

Every candidate completed text, one/two 256-square image, and JSON 3/3 gates.
Metrics drained to zero and bounded logs contained no HTTP 5xx, OOM,
structured-output, ROCm, or RCCL fatal signature.

## Result

| Image and device | C1 samples | C1 median tok/s | C4 samples | C4 median tok/s |
| --- | ---: | ---: | ---: | ---: |
| v0.27 production image, GPU2 | 6 | 77.29 | 6 | 211.02 |
| v0.28 Phase 114 image, GPU2 | 18 | 77.20 | 18 | 209.03 |

On the matched GPU, v0.28 differs by `-0.11%` at C1 and `-0.94%` at C4.
The independent v0.27 production controls varied from `77.71` to `79.12`
tok/s at C1 and from `211.78` to `214.46` tok/s at C4. The original Router
delta is therefore not sufficient evidence of a source-level regression.

## Decision

Keep every retained gfx906 optimization in the v0.28 development line. Do not
add a speculative kernel change merely to chase a sub-one-percent unpaired
result, and do not promote v0.28 to Qwen3.5 production until a future source
change demonstrates a repeatable net gain. The v0.27 production image remains
unchanged.

Raw logs, metrics, and benchmark JSON are intentionally kept outside Git with
the local phase evidence.
