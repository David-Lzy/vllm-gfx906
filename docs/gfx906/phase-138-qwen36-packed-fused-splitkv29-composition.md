# Phase 138: Qwen3.6 packed/fused SplitKV-29 TP4 composition

## Decision

Retain the composed 29-partition SplitKV profile only as a Qwen3.6 packed-INT8
TP4 **long-context development option**. It passed every routine correctness
and stability gate and improved the intended 32K prefix-cache-hit decode by
14.8 percent. It regressed fixed-128 C8 aggregate throughput by 6.8 percent,
which exceeds the five-percent regression limit. It is therefore not retained
as a general packed/fused default and does not change Qwen3.5 production.

## Scope

- Hardware: four AMD MI50 GPUs (`gfx906`) in one TP4 engine.
- Model: Qwen3.6 27B AWQ with the copy-on-write packed-INT8 embedding and
  lm-head profile, backed by the standard AWQ source checkpoint.
- Shared runtime: FP16 KV cache, 100K maximum model length, eight sequences,
  8,192 batched tokens, explicit `gfx906_gptq`, no MTP, prefix caching, and
  chunked prefill.
- Control: the retained packed/fused overlay with a 16-partition cap and
  forced 16-partition selection.
- Candidate: the identical overlay with a 32-partition cap and forced
  29-partition selection.
- Gates: text, one/two 256-square images, JSON `3/3`, drained metrics, and a
  bounded fatal-log scan. Scores are three warmed fixed-128 C1 samples, three
  C8 aggregates, and three paired 32K prefix-cache-hit fixed-128 samples.

## Results

| Metric | 16-split control | 29-split candidate | Change |
| --- | ---: | ---: | ---: |
| C1 completion throughput, median | 54.181 tok/s | 54.151 tok/s | -0.06% |
| C8 aggregate completion throughput, median | 235.620 tok/s | 219.663 tok/s | -6.77% |
| 32K cache-hit completion throughput, median | 17.979 tok/s | 20.644 tok/s | +14.83% |
| Engine startup | 623.708 s | 613.634 s | not a score gate |

The candidate emitted `splits=29 cap=32 forced=29`, confirming that the
bounded selection executed. Both arms passed text, one/two 256-square image,
and JSON `3/3` gates. Final metrics were drained, and the fatal scans found no
HTTP 5xx, OOM, traceback, xgrammar/FSM, RCCL/NCCL, or RAS signature.

The first C8 batch in each arm remained slower after warmup. The matched
median remains the conservative comparison: the long-context improvement is
real for this run, but it does not compensate for the general C8 regression.

## Implementation and evidence

- Source branch: `perf/qwen36-packed-fused-splitkv29`.
- Source commit: `be999204e0` (Qwen3.6 packed/fused SplitKV composition
  image and source-mount capable runner).
- The bounded selector and generic harness unit suite passed before the GPU
  window: `11 passed`.
- The production Qwen3.5 Router was stopped only for the serial all-GPU run.
  It was restored with two healthy workers, a healthy Router, successful text
  and image smoke requests, and no queued requests.
- Raw logs, request bodies, model files, compile caches, and machine paths
  are deliberately excluded from Git.

## Limits and next step

Do not combine this profile into the default packed/fused Qwen3.6 image, the
generic SplitKV selector, or the Qwen3.5 production Router. Preserve the
source and harness for a long-context-specific re-run when the packed layout,
Qwen3.6 attention path, Triton wheel, or hybrid-KV geometry changes. Any
future promotion must show the same long-context benefit without a material
short or saturated-decode regression.
