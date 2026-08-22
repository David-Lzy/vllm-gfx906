# Phase 41: v0.27 Normal C8 Runtime Attribution

## Goal

Recover the remaining normal-HTTP C8 regression in the Qwen3.5 9B AWQ v0.27
gfx906 candidate. Phase 40 measured 158.77 tok/s with the retained B4/gfx906
GPTQ path versus 224.26 tok/s in the v0.23 worker. This phase treats that
29.2% deficit as the target. It does not reopen the distinct Qwen3.6/Qwen3.8
long-context split-KV work.

## Required Evidence

1. Repeat the normal C8 request five times after identical warmups for the
   v0.23 control, v0.27 B4/gfx906 GPTQ, and v0.27 automatic ExLlama. Report
   median, p95, and per-round values. The benchmark harness now supports this
   with `C8_ITERATIONS`; a one-shot C8 result cannot approve a source change.
2. Keep an attribution lane separate from the release lane. It may use a
   profiler or a narrowly scoped HIP-event counter around scheduling,
   `execute_model`, and model forward, but it must never substitute profiler
   wall time for normal HTTP throughput.
3. Correlate the normal C8 median with a concrete residual category before
   modifying source: unquantized projections outside B4 coverage, W4A16,
   graph/Inductor partitioning, GDN/attention, or host scheduling.

## Candidate Rules

- The ExLlama selector is closed unless the repeated lane overturns the Phase
  40 result by more than five percent.
- Compile mode zero and the legacy V1/V2 format switch are closed by prior
  normal-serving experiments; do not repeat them as generic fallbacks.
- RDNA3/RDNA4 HybridW4A16 and AITER kernels are evidence only. A gfx906
  candidate needs a native wave64 design, numerical tests, and a measured
  hotspot. Do not widen an architecture gate.
- A new HIP C++ operator is allowed only when its category is at least 15% of
  C8 model-forward time and a shape-level prototype is at least 15% faster.
  It still needs a five-percent end-to-end C8 gain and routine text, one/two
  256px image, and JSON `3/3` parity.
- If graph/Inductor time is dominant, test one narrowly scoped graph partition
  or custom-op boundary. Preserve compilation mode 3 and CUDA graphs in the
  control; the rejected global compile-mode-zero result is not a baseline.

## Exit

Retain only a candidate that raises normal C8 by at least five percent with no
C1/multimodal regression, no HTTP 5xx, OOM, xgrammar/FSM, RCCL/NCCL fatal, or
non-idle final metrics. Production promotion remains gated on the 95% v0.23
floor in every routine scenario.
