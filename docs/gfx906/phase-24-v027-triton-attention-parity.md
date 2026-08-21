# Phase 24: v0.27 Triton Decoder-Attention Parity

## Scope

Phase 23 exposed a concrete decoder-attention difference: the retained gfx906
worker selected `TRITON_ATTN`, while the otherwise comparable v0.27 worker
selected `ROCM_ATTN`. This isolated v0.27 experiment forced only
`TRITON_ATTN`; model, GPU, checkpoint cache, and production-equivalent serving
parameters remained unchanged. Production was not modified.

## Correctness

The server logged `Using TRITON_ATTN backend (selected via
--attention-backend)`. It passed health, model discovery, text, one/two
256-square image requests, JSON constrained output `3/3`, fatal-log scan, and
idle metrics.

## Result

| Scenario | Phase 23 v0.27 auto | Forced TRITON_ATTN | Retained v0.23 | Forced / retained |
| --- | ---: | ---: | ---: | ---: |
| Text, C1 | 60.32 tok/s | 62.01 tok/s | 75.25 tok/s | 82.4% |
| One 256-square image, C1 | 55.73 tok/s | 58.41 tok/s | 66.76 tok/s | 87.5% |
| Two 256-square images, C1 | 51.46 tok/s | 55.82 tok/s | 63.29 tok/s | 88.2% |

Forcing Triton recovers part of the regression, especially for two images, but
does not satisfy the 95% per-scenario release floor. It is not a production
candidate and must not become a blanket gfx906 override.

The benchmark harness now warms the C8 concurrent shape before its measured
batch. The retained Phase 23 C8 row was recorded before that correction, so it
is diagnostic only and not used for release comparison.

## Decision

Attention-backend priority is a contributing cause, not the complete v0.27
gfx906 performance regression. Keep automatic backend selection unchanged.
Any further recovery work requires a new evidence-backed decoder or compiler
hypothesis; Phase 10 production canary remains closed.
