# Qwen GDN Output-Norm Portability

## Result

Retain a narrow, default-off gfx906 overlay for the Qwen GDN output
normalization path. It is not a production promotion and does not change any
default runtime behavior.

The overlay preserves the `[tokens, heads, head_dim]` layout through
`RMSNormGated.forward_native`, then flattens only for the row-parallel output
projection. The generic flattened path remains the default. The opt-in is
available only when both ROCm and
`VLLM_ROCM_ENABLE_GFX906_QWEN_GDN_OUTPUT_NORM=1` are present.

## Scope

- Source branch: `perf/gfx906-gdn-output-norm-v028`
- Base: the v0.28 gfx906 development image with the retained legacy QGEMM
  path
- Model-level service gate: Qwen3.5 9B AWQ TP1 on an isolated MI50
- Routine correctness gate: text, one/two 256-square images, and JSON `3/3`
- Throughput gate: fixed 128-token text C1 and synchronized C8

No large-grid or high-image-count capacity benchmark was run because this
change is in the decoder GDN output path, not multimodal capacity handling.

## Direct Kernel Evidence

The direct comparison used identical FP16 inputs and `RMSNormGated` weights.
All four shapes had exact FP16 output agreement (`max_abs_error=0`).

| Geometry | Control median | Opt-in median | Improvement |
| --- | ---: | ---: | ---: |
| Qwen3.5 TP1 C1, `[1, 32, 128]` | 0.20648 ms | 0.19024 ms | +7.86% |
| Qwen3.5 TP1 C8, `[8, 32, 128]` | 0.20880 ms | 0.19056 ms | +8.74% |
| Qwen 27B TP4 C1, `[1, 12, 128]` | 0.20720 ms | 0.18936 ms | +8.61% |
| Qwen 27B TP4 C8, `[8, 12, 128]` | 0.21376 ms | 0.19536 ms | +8.61% |

## Qwen3.5 Service Evidence

Both variants used the same warmed compile cache, model revision, serving
arguments, `ignore_eos=true`, and fixed 128-token completion budget.

| Metric | Control | Opt-in | Change |
| --- | ---: | ---: | ---: |
| C1 aggregate completion throughput | 75.431 tok/s | 76.584 tok/s | +1.53% |
| C8 aggregate completion throughput | 251.484 tok/s | 256.203 tok/s | +1.88% |
| C1 p50 latency | 1.696 s | 1.671 s | -1.49% |
| C8 p50 latency | 4.062 s | 3.988 s | -1.81% |

Text, one-image, two-image, and all three JSON-constrained smoke requests
returned valid non-empty output. The final metrics snapshot had zero running
and waiting requests. Fatal-log scans found no OOM, traceback, xgrammar/FSM,
or RCCL/NCCL-fatal signature.

## Startup Caveat

The cache-warmed control and opt-in engines each became healthy in 201 seconds.
The initial cold opt-in image run took longer while building its distinct
compile cache; it is not a steady-state comparison. Initial v0.28 multimodal
warmup remains a separate startup concern and is outside this output-path
overlay.

## Decision

Keep the source overlay because the complete service gate is stable and both
fixed-length throughput measurements are positive. It remains disabled by
default because the gain is modest and only Qwen3.5 TP1 has received a
model-service A/B. The exact Qwen 27B rank-local shapes are numerically
equivalent and directly faster, but Qwen3.6/Qwen3.8 service validation must be
run in their own TP4 development phase before enabling it for those models.

The production Qwen3.5 Router and its worker image are unchanged.
