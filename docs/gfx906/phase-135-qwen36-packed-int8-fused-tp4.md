# Phase 135: Qwen3.6 packed INT8 plus fused QK/RoPE TP4 composition

## Decision

Retain the ROCm-enabled Qwen3.6 fused QK-RMSNorm, partial MRoPE, and output
gate path as a `provisional-positive` development overlay. It passed the
normal multimodal and structured-output gates and improved the matched Qwen3.6
packed-INT8 TP4 median by less than two percent. This is useful evidence, but
not enough to change a production default or to claim a material speedup.

## Scope

- Hardware: four AMD MI50 GPUs (`gfx906`), one TP4 engine.
- Model: Qwen3.6 27B AWQ with the Phase 125 copy-on-write INT8 embedding and
  lm-head tables.
- Control: `v0.28.0-phase123-qwen38-int8-semantic`.
- Candidate: the control plus the upstream Qwen3.6 fused-QK/MRoPE source,
  guarded with CUDA-alike eligibility for ROCm, and the Phase 129 Triton 3.6
  conditional-pointer fix.
- Shared runtime: FP16, 100K maximum model length, eight sequences, 8,192
  batched tokens, legacy gfx906 QGEMM, SplitKV, no MTP, and warmed fixed-128
  text completion tests.

The control and candidate were launched serially in the same maintenance
window. Each used an independent compile cache, then received the same text,
one-image, two-image, exact JSON `3/3`, C1, and C8 workload sequence.

## Results

| Metric | Packed control median | Packed plus fused median | Change |
| --- | ---: | ---: | ---: |
| C1 completion throughput | 56.303 tok/s | 56.780 tok/s | +0.85% |
| C1 elapsed time | 2.2734 s | 2.2543 s | -0.84% |
| C8 aggregate completion throughput | 231.989 tok/s | 234.534 tok/s | +1.10% |
| C8 elapsed time | 4.4140 s | 4.3661 s | -1.08% |

All six measured repetitions per variant completed. The first C8 repetition
in both arms incurred a remaining JIT warmup and was slower; the table uses
the middle repetition, not an average distorted by that common outlier.

Text, one/two synthetic 256-square images, and JSON `3/3` passed for both
arms. Both final metrics snapshots drained to zero running and waiting
requests. The server logs contained no OOM, traceback, xgrammar/FSM, RCCL, or
NCCL fatal signature.

## Implementation and evidence

- Source branch: `perf/qwen36-packed-int8-fused-tp4`.
- Source commits: `5c4369253c` (harness and overlay) and `96050efb73`
  (copy-on-write source mount and startup-failure evidence retention).
- The runner deliberately mounts the compact model as `/model` and its
  standard AWQ source as `/source`; the copy-on-write model stores most files
  as `/source` symlinks.
- Raw machine-local artifacts are intentionally excluded from Git: startup
  logs, cache directories, smoke logs, metrics, and per-repetition JSON.

## Limits and next step

The gain is below the five-percent material-performance bar and has not yet
been independently re-run. Keep the fusion behind its Qwen3.6/ROCm guard and
re-run it when the enclosing attention, Qwen3.6 model code, or Triton wheel
changes. A repeatable positive result can then be promoted to
`retained-targeted`; it remains development-only until a separate serving
canary is approved.
