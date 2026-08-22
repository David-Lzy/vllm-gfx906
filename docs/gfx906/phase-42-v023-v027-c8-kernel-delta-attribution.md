# Phase 42: v0.23-v0.27 Normal C8 Kernel Delta Attribution

## Goal

Identify the concrete, reproducible category behind the stable 27% normal-HTTP
C8 regression measured in Phase 41. This phase is deliberately not another
backend-selector or global compiler-mode sweep. It compares the retained v0.23
automatic path with v0.27 explicit `gfx906_gptq` under the same single-GPU,
fixed-output C8 request geometry.

## Why This Is Needed

Phase 41 closes the lowest-risk community hypothesis from
[issue #49699](https://github.com/vllm-project/vllm/issues/49699): forcing
ExLlama does not recover MI50 throughput. Phase 35 shows native GPTQ W4A16 is
material in a v0.27 trace, but an earlier K=256 HIP C++ trial improved the
microbenchmark without a sufficient server gain. A new implementation now
requires a direct old/new category delta, rather than inference from one
candidate trace.

The hybrid-model `0.238 tok/s` long-context result is out of scope. That
separate 256-wide-head, nonstandard-page attention problem is already addressed
by the opt-in gfx906 SplitKV path documented in Phase 31. The community report
in [issue #50264](https://github.com/vllm-project/vllm/issues/50264) supports
that separation: it identifies SplitKV as the appropriate remedy and warns
that simply widening native ROCm attention eligibility is invalid.

## Method

1. On development GPU2 only, start one isolated v0.23 automatic worker and
   one isolated v0.27 explicit `gfx906_gptq` worker, sequentially.
2. Keep checkpoint, FP16 KV, 100K maximum context, eight sequences, 32,768
   batched tokens, chunked prefill, prefix caching, one renderer, and the
   fixed 64-token C8 text payload unchanged.
3. Use vLLM's opt-in torch profiler only for attribution. Run two C8 warmups,
   profile one eight-request C8 generation, stop the profiler, and retain trace
   events with shape metadata. Release throughput remains the Phase 41 ordinary
   HTTP result, never profiler wall time.
4. Aggregate GPU duration and call counts by native GPTQ W4A16, B4
   unquantized GEMM, rocBLAS fallback, GDN/Mamba, paged attention, graph launch,
   and host synchronization. Match C8 token counts before comparing traces.
5. Map any category that explains at least 15% of the old/new model-forward
   delta to its exact operation and shape. Only then create one guarded source
   candidate.

## Candidate Rules

- A new gfx906 HIP C++ operator is eligible only when an exact shape category
  clears both a 15% old/new delta contribution and a 15% shape microbenchmark
  improvement. It must then raise ordinary C8 by at least 5%.
- A compiler partition candidate is eligible only when compiled graph, graph
  launch, or synchronization accounts for the material delta. It must preserve
  compilation mode 3 and CUDA graphs; compile mode zero remains rejected.
- Gfx11/gfx12 wave32, AITER, and HybridW4A16 implementations are design
  references only. No architecture guard may be widened without a native
  gfx906 wave64 implementation and numerical coverage.
- If no category is material, record a no-go result and keep v0.23 production.

## Gates

The profiler workers must pass text, one/two 256px image, and JSON `3/3` after
each trace. Reject on HTTP 5xx, empty output, OOM, xgrammar/FSM, RCCL/NCCL
fatal, traceback, or non-idle final metrics. Production GPU0/GPU1, port 8002,
Router, Compose, and production cache remain read-only.

## Exit

Phase 42 produces one attributable source target or a documented no-go. It
does not authorize production promotion; a retained implementation must still
clear the normal C1/C8/multimodal release gates against v0.23.

## Result: W4A16 Is The Remaining Material Category

The matching profiler captures completed on GPU2 with identical C8 request
counts and 64 completion tokens per request. Both workers then passed the
routine text, one/two 256px image, and JSON `3/3` gates. Neither worker left
running or waiting requests, and the log scans found no OOM, HTTP 5xx,
xgrammar/FSM, traceback, or RCCL/NCCL fatal event.

| C8 trace category | v0.23 retained worker | v0.27 initial runtime | Delta |
| --- | ---: | ---: | ---: |
| `gpu_model_runner: forward`, mean | 14.86 ms | 48.98 ms | +34.12 ms |
| Native GPTQ W4A16 GPU time | 1.085 s / 8,192 calls | 2.084 s / 8,192 calls | +92.1% |
| Host-visible `hipGraphLaunch`, mean | 0.389 ms | 39.16 ms | +38.77 ms |

The graph-launch delta justified the Phase 43 runtime control, but it is not
by itself a source-change target: the control restores normal graph-launch
timing without recovering ordinary C8 throughput. The native W4A16 category
remains materially slower after that control and is the only category that
clears this phase's source-candidate threshold.

The v0.23 implementation is not merely a 256-wide K-block variant. It is a
gfx906-tuned QGEMM/QDQ bundle with a distinct dequantization sequence,
accumulator, launch bound, and output initialization behavior. Previous local
experiments rejected changing only the block width or only `fdot2`; Phase 44
therefore evaluates the complete bundle behind an opt-in, single-architecture
build guard rather than reviving either partial candidate.

## Decision

Phase 42 is complete. Keep the v0.23 worker in production. Continue with the
isolated Phase 44 legacy-QGEMM composition test on GPU2 only.
