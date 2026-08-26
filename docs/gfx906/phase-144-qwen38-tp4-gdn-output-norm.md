# Qwen3.8 TP4 GDN Output-Norm Validation

## Result

Do not enable the Phase 142 GDN output-normalization overlay for the standard
Qwen3.8 27B AWQ TP4 profile. It is correct and stable, but its measured
service-level effect is within run-to-run noise and slightly regresses the
concurrent and long-context medians.

The source overlay remains available and default-off. Its prior positive
Qwen3.5 9B evidence is unchanged; this result only prevents an unsupported
generalization to Qwen3.8 27B.

## Scope

- Checkpoint: `cyankiwi/Qwen3.8-27B-AWQ-INT4` at revision
  `63768c10df38c0395e12ef49edac1bd539eaeeea`
- Runtime: v0.28 gfx906 development image with the retained legacy AWQ GEMM
  and the compatible SplitKV `16/16` profile
- Hardware: four AMD MI50 GPUs with tensor parallelism four
- Fixed service contract: FP16 KV cache, 100K model length, eight sequences,
  8,192 batched tokens, no MTP, text plus one/two 256-square image smoke, and
  JSON-constrained output
- Control: default flattened output-projection input
- Candidate: identical service with
  `VLLM_ROCM_ENABLE_GFX906_QWEN_GDN_OUTPUT_NORM=1`

Each performance request used `ignore_eos=true` and an exact 128-token
completion budget. Control and candidate used isolated compile caches so the
steady-state comparison did not inherit one another's generated artifacts.

## Results

| Fixed-budget metric | Control median | Output-norm median | Change |
| --- | ---: | ---: | ---: |
| C1 completion throughput | 53.1619 tok/s | 53.2810 tok/s | +0.22% |
| C8 aggregate completion throughput | 207.1648 tok/s | 206.3451 tok/s | -0.40% |
| 32K prefix-cache-hit decode | 17.8382 tok/s | 17.8164 tok/s | -0.12% |

Cold readiness was 643.8 seconds for control and 633.9 seconds for the
candidate. The independent cache setup makes this startup difference
non-actionable; startup time is not used for the decision.

Both services returned non-empty text, one-image and two-image responses, and
valid JSON for all three constrained requests. The final queue state was
drained. The retained result logs contain no OOM, traceback, xgrammar/FSM,
RCCL/NCCL fatal, or RAS illegal-access signature.

## Why The Direct Microbenchmark Did Not Transfer

The Phase 142 rank-local screen removed a reshape around the normalized GDN
output and was exact FP16 with an approximately 8.6% improvement for the
isolated Qwen27 shape. In a Qwen3.8 TP4 service, that local operation is not
the limiting cost: AWQ linear work, the Triton GDN decode path, tensor-parallel
communication, and scheduler overhead dominate the fixed-budget end-to-end
measurement. The small local win is therefore not visible as a service gain.

## Decision

Keep the overlay source as an opt-in, default-off experiment for the models
where a dedicated service A/B is positive. Do not add it to the Qwen3.8 TP4
profile or production defaults. Future Qwen3.8 performance work should target
the GDN decode, AWQ GEMM, and TP communication path rather than another
output-layout reshape.
