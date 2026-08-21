# Phase 26: v0.27 AOT Runtime Parity

## Scope

This isolated GPU2 experiment tested the v0.27 Qwen3.5 9B AWQ worker with
`VLLM_USE_AOT_COMPILE=0`. It retained the Phase 24 serving configuration:
gfx906 ExLlama W4A16 selection, forced Triton decoder attention, 100K context,
eight sequences, FP16 KV cache, and the routine text plus one/two 256-square
image gate. Production was not modified.

## Result

The non-AOT path was stable: health, model discovery, all routine text and
image requests, JSON-constrained output `3/3`, and idle request metrics passed.
Against the same v0.27 Phase 24 baseline, median completion throughput changed
only from `62.015` to `62.102 tok/s` for text, `58.415` to `58.561 tok/s` for
one image, and `55.818` to `55.859 tok/s` for two images. This is measurement
noise, not a performance recovery.

It also did not improve startup. The graph compilation line shortened, but the
combined compilation and initial profiling warmup took `496.70s`; engine
initialization was `523.46s` versus `519.58s` for the AOT-on control.

## Decision

Keep AOT compilation enabled. The v0.27 runtime remains a functional gfx906
candidate, but it remains below the legacy production worker's small-request
performance floor. This configuration change is rejected and does not alter a
production default.
