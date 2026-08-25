# Phase 136: v0.28 Qwen3.8 TP4 SplitKV 29-partition rebase

## Decision

Retain the gfx906 Qwen3.8 TP4 SplitKV 29-partition profile as a targeted,
default-off v0.28 development option. In a paired all-GPU measurement it
improved the intended 32K prefix-cache-hit decode path by 13.51 percent while
leaving short C1 effectively unchanged and improving C8 modestly. This is a
material long-context result, but it is not a reason to change the unrelated
Qwen3.5 9B production service.

## Scope

- Hardware: four AMD MI50 GPUs (`gfx906`) in one TP4 engine.
- Model: Qwen3.8 27B standard AWQ, FP16 KV cache, 100K maximum model length,
  no MTP.
- Control: existing gfx906 SplitKV path with a 16-partition cap and a forced
  16-partition selection.
- Candidate: the same image and request settings with a 32-partition cap and
  a forced 29-partition selection.
- Workload: warmed fixed-128 text C1/C8 requests and a 32K
  prefix-cache-hit fixed-128 decode, alongside text, one/two 256-square image,
  and JSON `3/3` routine gates.

The source exposes two bounded environment controls:
`VLLM_ROCM_GFX906_SPLITKV_MAX_SPLITS` and
`VLLM_ROCM_GFX906_SPLITKV_FORCE_SPLITS`. They are inert unless the existing
gfx906 SplitKV route is selected, reject values outside `1..32`, and do not
change the generic default.

## Results

| Metric | 16-split control | 29-split candidate | Change |
| --- | ---: | ---: | ---: |
| C1 completion throughput, median | 56.552 tok/s | 56.604 tok/s | +0.09% |
| C8 aggregate completion throughput, median | 222.116 tok/s | 226.064 tok/s | +1.78% |
| 32K cache-hit completion throughput, median | 18.036 tok/s | 20.472 tok/s | +13.51% |
| Engine startup | 462.0 s | 452.0 s | measurement noise / not a gate |

Both arms passed text, one/two synthetic 256-square images, and JSON `3/3`.
Their final metrics snapshots had zero running and waiting requests. The
bounded fatal scan found no HTTP 5xx, OOM, traceback, xgrammar/FSM, RCCL/NCCL,
or RAS signature. Unit coverage for default, valid, and invalid environment
overrides passed (`11 passed`).

## Implementation and evidence

- Source branch: `perf/v028-qwen38-tp4-splitkv29-rebase`.
- Source commits: `abbe5b9170` (bounded selector and test/harness) and
  `b27df4bb87` (multimodal CLI argument correction).
- The candidate logs recorded `splits=29 cap=32 forced=29` during warmup and
  serving, confirming the selected path rather than merely accepting unused
  environment variables.
- The production Qwen3.5 Router was stopped only for the all-GPU maintenance
  window and was restored healthy afterward. The unrelated indexer remained
  paused; the pre-existing XMR wrapper was paused for the measurement and
  resumed afterward.
- Raw logs, request bodies, model files, compile caches, and machine paths are
  deliberately excluded from Git.

## Limits and next step

Keep this option confined to compatible Qwen3.8 TP4 development profiles until
an explicit model-serving canary is justified. Do not enable it globally, on
the Qwen3.5 production workers, or together with a new MTP/quantization change
without another paired comparison. Re-run the long-context check whenever the
paged-decode implementation, hybrid KV geometry, Triton version, or Qwen3.8
attention layout changes.
