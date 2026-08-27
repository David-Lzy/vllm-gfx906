# Phase 151: Qwen3.6 Packed-Plus-Fused TP2x2 Router

## Status

Completed - rejected for the default and the Qwen3.6 TP2x2 development
profile. The candidate was correct and stable but slower in every measured
steady-state target, while adding a substantial cold-start cost. The selected
Qwen3.5 9B TP1x4 Router service was restored after the experiment.

## Question

Phase 150 showed that copy-on-write INT8 packing of the Qwen3.6 embedding and
LM head improves TP2x2 Router C8 throughput by 5.5 percent. Phase 130's
guarded ROCm fused QK-RMSNorm, partial MRoPE, and output-gate route was a small
positive in isolation. This phase asks whether those two already-correct
changes compose in the high-concurrency Qwen3.6 service topology.

## Controlled Comparison

Both arms use the same packed Qwen3.6 27B AWQ model, two TP2 engines, the
same Router, 100K context, FP16 KV cache, eight sequences and 8,192 batched
tokens per engine, SplitKV-16, `image=64`, `video=0`, and cache-busted routine
text/image prompts. The candidate alone layers the retained gfx906 Triton
conditional-pointer wheel and Qwen3.6 fused source dispatch onto the Phase
142 image.

Routine acceptance consists of text, one/two 256-square image, JSON `3/3`,
C1, C8, and mixed C16. The phase rejects any HTTP 5xx, malformed output,
OOM, xgrammar/FSM, RCCL/NCCL fatal, RAS, illegal instruction, or undrained
queue. A stable one-percent target improvement retains the profile for Qwen3.6
development; five percent is required for a material serving recommendation.

## Reproducibility

- Candidate Docker overlay:
  `docker/Dockerfile.gfx906-v028-phase151-qwen36-fused`
- Reversible all-GPU runner:
  `tools/gfx906/run_qwen36_packed_fused_tp2x2_router_ab.sh`
- Compact benchmark client:
  `tools/gfx906/benchmark_qwen36_packed_tp2x2.py`

Weights, caches, generated images, and raw result files remain outside Git.

## Result

The 2026-08-27 all-four-MI50 A/B passed text, one/two-image, and JSON `3/3`
smoke in both arms. Both runs left an empty fatal-signature scan and drained
all request queues. The control used the Phase 150 packed-INT8 checkpoint on
the selected Phase 142 image; the candidate used the same checkpoint with the
Phase 129 Triton wheel and Phase 130 Qwen3.6 fused-source overlay.

| Metric | Packed control | Packed plus fused | Delta |
| --- | ---: | ---: | ---: |
| C1 median completion throughput | 45.72 tok/s | 44.81 tok/s | -2.0% |
| C8 aggregate median completion throughput | 247.06 tok/s | 235.56 tok/s | -4.7% |
| Mixed C16 aggregate median completion throughput | 292.65 tok/s | 288.88 tok/s | -1.3% |

The candidate's two TP2 engines also required about 394 and 406 seconds to
reach initialized-engine state, versus the cached control startup. This is a
development-time penalty rather than a request-time metric, but it is too
large to ignore for a maintenance-window deployment.

The packed-INT8 TP2x2 profile from Phase 150 therefore remains the retained
Qwen3.6 development baseline. Do not compose it with this fused overlay by
default. Raw evidence is retained under the configured external build root in
the `phase-151-qwen36-packed-fused-tp2x2-router` result directory.
