# Phase 153: Qwen3.8 Packed-INT8 TP4 with SplitKV-29

## Status

Completed and retained as a Qwen3.8 TP4 development-profile composition.

## Question

The retained Qwen3.8 27B development profiles answer adjacent questions but
not this one directly: Phase 124 measured packed-INT8 with the older TP4
SplitKV-16 setup; Phase 136 measured standard AWQ with SplitKV-29; and Phase
152 measured packed-INT8 with SplitKV-16 in TP2x2 Router. This phase isolates
whether packed-INT8 also improves the best current Qwen3.8 TP4 attention
geometry.

## Controlled Comparison

- Hardware: four MI50 GPUs in one TP4 engine.
- Runtime: retained Phase 142 v0.28 gfx906 image and explicit
  `gfx906_gptq` backend.
- Control: standard `Qwen3.8-27B-AWQ-INT4` checkpoint.
- Candidate: copy-on-write packed-INT8 embedding/lm-head overlay, with the
  standard checkpoint mounted read-only at `/source` for unchanged shards.
- Shared settings: 100K maximum model length, FP16 KV cache, eight sequences,
  8,192 batched tokens, 64 images, video disabled, 16 Mi-pixel cap, prefix
  caching, chunked prefill, no MTP, and forced gfx906 SplitKV `29` with cap
  `32`.

No source kernel, production Compose file, model download, or quantization
format changes in this phase. The checkpoint overlay is the sole variable.

## Gates

- Text, one/two 256-square image, and JSON `3/3` smoke requests.
- Three fixed-128 C1 and C8 text rounds plus three 32K prefix-cache-hit C1
  decode rounds.
- No HTTP 5xx, OOM, traceback, xgrammar/FSM, RCCL/NCCL fatal, RAS, illegal
  instruction, or residual running/waiting queue.
- The runner stops the selected Qwen3.5 production Router only after an idle
  preflight and restores it through an exit trap.

## Retention Rule

Reject on any routine correctness or stability failure. Retain only if the
packed profile improves a target metric without regressing C1 or C8 by more
than five percent. A result of at least five percent on C8 or 32K decode is a
material Qwen3.8 TP4 development recommendation. This cannot replace the
separate Qwen3.5 9B TP1x4 production topology.

## Result

Both variants completed text, one/two 256-square image, and JSON `3/3` gates.
The fatal signature scans were empty and the engines drained before teardown.
Both use the same Phase 142 v0.28 image, explicit `gfx906_gptq`, TP4, and
SplitKV cap `32` / forced split count `29`.

| Variant | Cold health | Fixed-128 C1 | Fixed-128 C8 aggregate | 32K cache-hit C1 |
| --- | ---: | ---: | ---: | ---: |
| Standard AWQ | 643.897 s | 53.459 tok/s | 206.851 tok/s | 20.539 tok/s |
| Packed INT8 overlay | 613.597 s | 54.961 tok/s | 220.873 tok/s | 20.639 tok/s |
| Packed delta | -4.70% | +2.81% | +6.78% | +0.48% |

The packed overlay clears the material C8 retention gate without a C1 or
long-context regression. Retain it for Qwen3.8 TP4 SplitKV-29 development
work. Its 32K advantage is within routine variance, so SplitKV-29 remains the
long-context mechanism rather than a reason to expose the packed overlay as a
new capacity profile. The selected Qwen3.5 9B TP1x4 Router remains the
production aggregate-throughput winner.

The runner now treats the Server2 Ollama maintenance fallback as a required
precondition before it stops production, then releases the fallback listener
before its production recovery trap restarts Compose.

## Artifacts

- Branch: `perf/qwen38-packed-int8-splitkv29-tp4`.
- Runner: `tools/gfx906/run_qwen38_packed_int8_splitkv29_tp4_ab.sh`.
- Raw evidence: external phase-153 build root on disk2.
