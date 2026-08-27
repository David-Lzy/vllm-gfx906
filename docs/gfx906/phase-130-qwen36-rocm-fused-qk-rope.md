# Phase 130: Qwen3.6 ROCm Fused QK/RoPE Dispatch

## Scope

This experiment evaluated the upstream Qwen3.6 model-level fused QK RMSNorm,
partial RoPE, and output-gate path on gfx906. It used the prior Triton 3.6
fat-pointer intersection fix and changed only the eligibility test from
CUDA-only to CUDA-alike. Qwen3.6's Qwen3.5 model implementation reuses the
Qwen3Next attention class that owns this dispatch.

## Result

The TP2 AWQ service passed text, one-image, two-image, and constrained JSON
smoke checks. No compiler assertion, allocation failure, or distributed runtime
failure appeared during startup or serving.

Matched 128-token measurements were:

| Variant | C1 completion tok/s | C8 aggregate completion tok/s |
| --- | ---: | ---: |
| CUDA-only control | 40.78 | 156.42 |
| ROCm-enabled warmed repeat | 41.42 | 156.12 |

An initial candidate measurement was slower, then a warmed repeat produced a
small C1 gain with effectively unchanged C8 throughput. The patch is retained
as an optional source-level candidate under the project's positive-gain policy.
It is not a release default and does not solve the separate Qwen3.8 hybrid
decode bottleneck.

## Boundary

The generic compiler setting named `enable_qk_norm_rope_fusion` is independent
from this model-level callable and remains disabled. Future promotion requires
an independent larger-topology evaluation and a canary decision.
