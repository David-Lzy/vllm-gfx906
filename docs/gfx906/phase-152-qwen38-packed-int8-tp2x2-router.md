# Phase 152: Qwen3.8 Packed-INT8 TP2x2 Router

## Status

Completed - retained as a Qwen3.8 TP2x2 Router development profile. It is not
a production promotion: the selected Qwen3.5 9B TP1x4 Router remains the
aggregate-throughput winner and was restored after the benchmark.

## Question

Phase 123 repaired the Qwen3.8 packed-INT8 numerical path and Phase 124 found
a small TP4 improvement. Phase 145 showed that TP2x2 Router is the stronger
concurrent topology for Qwen3.8. This phase tests that previously unmeasured
combination under one current runtime contract.

## Controlled Comparison

- Control: the retained Qwen3.8 27B AWQ checkpoint.
- Candidate: the copy-on-write checkpoint with packed INT8 embedding and
  lm-head shards; the original checkpoint is mounted read-only at `/source`
  because the overlay retains symlinks for its unchanged shards.
- Runtime: the same Phase 142 v0.28 image, `gfx906_gptq`, two TP2 workers on
  GPU pairs `0,1` and `2,3`, and a round-robin Router for both arms.
- Contract: 100K context, FP16 KV cache, eight sequences and 8,192 batched
  tokens per worker, `image=64`, `video=0`, a 16 Mi-pixel limit, prefix
  caching, chunked prefill, no MTP, and SplitKV 16.

The only intended semantic difference is the packed checkpoint. The test does
not build a new kernel, download a model, or alter production deployment files.

## Routine Gates

- Non-empty text, one 256-square image, and two 256-square image responses.
- JSON constrained response `3/3` with exact `{"status":"ok"}` content.
- Fixed 128-token C1, C8 text, and mixed C16 Router benchmarks.
- Material distribution to both workers; no pending requests after each arm.
- No HTTP 5xx, OOM, traceback, xgrammar/FSM, RCCL/NCCL fatal, RAS, or illegal
  instruction signature.

## Decision Policy

Reject the candidate for any correctness or stability failure. Retain it for
the Qwen3.8 TP2x2 development profile only if it improves repeated C8 or mixed
C16 performance without regressing any core metric by more than five percent.
A five-percent gain is required for a material serving recommendation. This
phase cannot change the selected Qwen3.5 production profile.

## Reproducibility

- Runner: `tools/gfx906/run_qwen38_packed_int8_tp2x2_router_ab.sh`
- Workload: `tools/gfx906/benchmark_qwen38_packed_tp2x2.py`
- Raw logs and JSON are stored outside Git under the phase build root.

## Result

Run `20260827T015442Z` passed all routine gates for both arms. Text, one/two
256-square images, and constrained JSON `3/3` returned non-empty correct
responses. The round-robin Router delivered `41/40` HTTP 200 requests to its
two workers in each arm. Both fatal-signature scans were empty and the queue
drained after the workload.

| Metric | Standard AWQ | Packed INT8 | Delta |
| --- | ---: | ---: | ---: |
| C1 median completion | 44.65 tok/s | 45.80 tok/s | +2.6% |
| C8 aggregate median completion | 237.07 tok/s | 248.34 tok/s | +4.8% |
| Mixed C16 aggregate median completion | 271.95 tok/s | 292.16 tok/s | +7.4% |

Packed engine initialization completed in about `404s` and `434s` for the two
TP2 workers, versus `421s` and `421s` for standard AWQ under the same cold
image/runtime contract. It therefore adds no observed startup penalty. Phase-
local compiler caches were deleted after evidence capture.

The packed checkpoint clears the retained-profile gate, including the material
mixed-C16 threshold. Keep it as the preferred concurrent Qwen3.8 27B TP2x2
development profile. Do not replace the selected Qwen3.5 production service:
the 9B TP1x4 Router remains materially faster for aggregate concurrent work.
