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

## Result: Stable Regression, Not Backend Selection

The repeated ordinary-HTTP lane completed on GPU2 with the production model,
FP16 KV cache, 100K maximum context, eight sequences, and the same C8 payload.
Every run passed text, one/two 256px image, and JSON `3/3` gates. The final
metrics were idle and the fatal-log scans were empty.

| Candidate | C8 per-round completion tok/s | C8 median | Relative to v0.23 |
| --- | --- | ---: | ---: |
| v0.23 automatic selection | 66.31, 218.15, 216.79, 214.51, 216.70 | 216.70 | 100.0% |
| v0.27 explicit `gfx906_gptq` | 158.44, 158.13, 157.90, 157.33, 157.70 | 157.90 | 72.86% |
| v0.27 automatic ExLlama | 157.70, 157.55, 157.36, 157.57, 157.46 | 157.55 | 72.70% |

The v0.23 first round was a one-off post-warmup residual. Its following four
rounds were 214.51--218.15 tok/s and its five-round median remains 216.70
tok/s. Both v0.27 selections were tightly clustered. The ordinary serving
regression is therefore stable at about 27.1--27.3%, rather than a sampling or
router artifact.

Automatic ExLlama is 0.22% below the explicit gfx906 GPTQ selection, which is
well within noise and does not meet the five-percent retention gate. This
closes the backend-selector branch on MI50. The corresponding C1 and
multimodal results are also below v0.23: explicit GPTQ retained 79.6% text C1,
83.4% one-image C1, and 82.1% two-image C1. Production remains on v0.23.

The next bounded task is Phase 42: collect comparable old/new C8 kernel traces
with shape metadata, then modify only the largest measured execution category.
The Phase 32/35 evidence already indicates that native GPTQ W4A16 is material,
but it does not explain the old/new runtime delta by itself.

## Evidence

Raw results are excluded from Git and retained under:

- `phase-41/results/20260822T045716Z-small-mm-parity` for v0.23 versus explicit
  gfx906 GPTQ;
- `phase-41/results/20260822T052523Z-small-mm-parity` for v0.23 versus
  automatic ExLlama.
