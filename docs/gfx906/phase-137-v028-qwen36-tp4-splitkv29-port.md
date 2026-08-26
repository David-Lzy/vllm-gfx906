# Phase 137: v0.28 Qwen3.6 TP4 SplitKV 29-partition port

## Decision

Retain the gfx906 Qwen3.6 TP4 SplitKV 29-partition configuration as a
targeted, default-off v0.28 development profile. In a paired all-GPU run it
improved both fixed-128 C8 completion throughput and the intended 32K
prefix-cache-hit decode path, while leaving C1 unchanged within measurement
noise. This does not justify a change to the unrelated Qwen3.5 9B production
Router.

## Scope

- Hardware: four AMD MI50 GPUs (`gfx906`) in one TP4 engine.
- Model: standard Qwen3.6 27B AWQ, FP16 KV cache, 100K maximum model length,
  no MTP, and explicit `gfx906_gptq`.
- Control: existing gfx906 SplitKV route with a 16-partition cap and forced
  16-partition selection.
- Candidate: same image and request settings with a 32-partition cap and a
  forced 29-partition selection.
- Workload: warmed fixed-128 text C1/C8 and a paired 32K
  prefix-cache-hit fixed-128 decode, alongside text, one/two 256-square image,
  and JSON `3/3` routine gates.

The source exposes two bounded environment controls:
`VLLM_ROCM_GFX906_SPLITKV_MAX_SPLITS` and
`VLLM_ROCM_GFX906_SPLITKV_FORCE_SPLITS`. They are inert unless the existing
gfx906 SplitKV route is selected, reject values outside `1..32`, and leave the
generic default unchanged.

## Results

| Metric | 16-split control | 29-split candidate | Change |
| --- | ---: | ---: | ---: |
| C1 completion throughput, median | 52.904 tok/s | 52.868 tok/s | -0.07% |
| C8 aggregate completion throughput, median | 206.923 tok/s | 217.289 tok/s | +5.01% |
| 32K cache-hit completion throughput, median | 17.784 tok/s | 20.157 tok/s | +13.34% |
| Engine startup | 653.829 s | 653.798 s | measurement noise / not a gate |

Both arms passed text, one/two synthetic 256-square images, and JSON `3/3`.
Their final metrics snapshots had zero running and waiting requests. The fatal
scan found no HTTP 5xx, OOM, traceback, xgrammar/FSM, RCCL/NCCL, or RAS
signature. Unit coverage for the bounded selector and generic harness passed
before the run (`11 passed`).

## Implementation and evidence

- Source branch: `perf/v028-qwen36-tp4-splitkv29-port`.
- Source commit: `bf26adc855` (generic portability harness and Qwen3.6 overlay
  image).
- The candidate logged `splits=29 cap=32 forced=29` during warmup and serving,
  confirming that the selected path executed.
- Both engine startups were about 654 seconds because graph compilation and
  all-GPU warmup dominate the cold-start path; startup is not a score gate.
- The production Qwen3.5 Router was stopped only for this all-GPU maintenance
  window and was restored healthy afterward with zero queued requests.
- Raw logs, request bodies, model files, compile caches, and machine paths are
  deliberately excluded from Git.

## Limits and next step

Keep this option confined to compatible Qwen3.6 TP4 development profiles. Do
not enable it globally, on Qwen3.5 production workers, or together with MTP,
packed INT8, or a different attention implementation without another paired
comparison. Re-run the long-context check whenever the paged-decode
implementation, hybrid KV geometry, Triton version, or Qwen3.6 attention
layout changes.
