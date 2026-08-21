# Phase 31: Qwen3.8 GPTQ Plus Split-KV Composition

## Scope

Phase 28 established that the unusable long-context decode rate came first
from generic Triton W4A16 GEMM, not from RCCL or attention. It added the
explicit `gfx906_gptq` linear backend. This phase checks whether that backend
also composes with the existing opt-in gfx906 split-KV decode candidate before
considering a new HIP C++ attention operator.

All work used a temporary Qwen3.8 27B AWQ TP2 server on MI50 GPU2/GPU3. The
Qwen3.5 9B production Router, GPU0/GPU1 workers, port 8002, Compose files,
and production model cache were not changed.

## Community Basis

The hybrid Qwen geometry has 256-wide full-attention heads and an Mamba-aligned
784-token physical KV page. [vLLM issue #50264](https://github.com/vllm-project/vllm/issues/50264)
attributes an analogous long-context hybrid-model slowdown to paged attention
on newer AMD hardware. The upstream [split-KV PR #45916](https://github.com/vllm-project/vllm/pull/45916)
adds a compatible FP16/BF16 Triton split-and-reduce design, but limits its
automatic selection to newer GPUs. This fork retains the upstream design and
gates gfx906 use behind `VLLM_ROCM_ENABLE_GFX906_SPLITKV=1`; the default remains
off.

That scope matters: it reuses a numerically validated attention implementation
instead of widening the unsupported native ROCm paged-attention selector. The
native ROCm kernel still lacks the required 256-wide-head and 784-page
instantiation, so merely changing an eligibility predicate would be incorrect.

## Implementation

- `Gfx906GPTQWNA16LinearKernel` remains explicitly selected with
  `--linear-backend gfx906_gptq`.
- The split-KV launch stays limited to FP16/BF16, 256-wide heads, non-FP8 KV,
  no ALiBi/sliding-window/sinks, and `VLLM_ROCM_ENABLE_GFX906_SPLITKV=1`.
- `VLLM_ROCM_GFX906_SPLITKV_DEBUG=1` emits the selected launch geometry only
  for isolated measurements. Both flags are registered vLLM environment
  variables, so the server no longer reports them as unknown.
- The test image overlays only the attention launcher and inherits the Phase
  28 native GPTQ W4A16 implementation.

## Method

Both candidates used `cyankiwi/Qwen3.8-27B-AWQ-INT4`, TP2, FP16, a 100K
maximum context, no MTP, and the same 32,768-word prefix-cache-hit probe with
an eight-token completion. The control used the explicit GPTQ backend only;
the candidate additionally enabled split-KV. Fixed decode used five 128-token
samples. The candidate also ran text, one/two 256-square images, and JSON
structured output three times.

Raw local evidence is intentionally outside Git:

- control: `phase-28/results/20260821T224528Z-qwen38-27b-awq-no-mtp-fixed128`
  and `20260821T224551Z-qwen38-27b-awq-no-mtp-longctx`;
- combined candidate: `phase-28/results/20260821T230921Z-qwen38-27b-awq-no-mtp-gates`,
  `20260821T230939Z-qwen38-27b-awq-no-mtp-fixed128`, and
  `20260821T231001Z-qwen38-27b-awq-no-mtp-longctx`.

The candidate log confirms the launch, rather than inferring it from an
environment variable:

```text
gfx906 split-KV decode: batch=1 kv_heads=2 seq_len=32780
physical_block=784 compute_block=16 splits=14
```

The physical page size is not divisible by 32, so the current safe logical
tile is 16 and the heuristic selects 14 splits. That differs from an earlier
microbenchmark's idealized 32-token tile; this phase does not claim that a
non-divisible 32-token mapping is correct without a separate numerical test.

## Results

| Candidate | Fixed 128 decode | 32K cache-hit decode | Result |
| --- | ---: | ---: | --- |
| explicit gfx906 GPTQ control | 34.987 tok/s | 0.906 tok/s | Same-model control |
| GPTQ plus gfx906 split-KV | 35.612 tok/s | 1.337 tok/s | +1.8% short, +47.6% long |
| Phase 28 original generic W4A16 geometry control | 0.441 tok/s | 0.238 tok/s | Historical cross-model geometry reference |
| Qwen3.6 GPTQ reference | 35.911 tok/s | 1.226 tok/s | Matching architecture-family reference |

The combined Qwen3.8 result is 5.61x the original `0.238 tok/s` geometry
control and 9.0% above the same-runtime Qwen3.6 GPTQ reference. The first
ratio is historical and not a same-model A/B; the decisive same-model result is
`0.906 -> 1.337 tok/s`.

Text, one-image, two-image, and JSON `3/3` gates returned non-empty/correct
responses. The run had no HTTP 5xx, OOM, RCCL fatal, xgrammar/FSM, or stuck
request. The temporary server was stopped after recording results; production
health remained good throughout.

## Decision

The combined recovery meets the Phase 31 retention gate and the original
Qwen3.8 objective: it improves the primary long-context result by more than
10%, preserves fixed decode, and reaches the Qwen3.6 comparison level. No new
HIP C++ attention operator is justified now.

The option remains explicit and non-production: Phase 30 showed the v0.27
Qwen3.5 9B serving path remains below the retained production throughput floor.
Any later attempt to exceed this result should first add a numerical test for
a non-divisible 32-token logical tile and then remeasure the full Qwen3.8
server path. It must not turn on split-KV automatically for all gfx906 models.
