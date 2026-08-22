# Phase 39: v0.27 Throughput Measurement Normalization

## Why This Phase Exists

The historical `0.238 tok/s` value was being interpreted as a general v0.27
gfx906 serving rate. It was not: it was the Qwen3.6 27B AWQ long-context
control used as a cross-model geometry reference while investigating Qwen3.8.
It must not be compared with ordinary short HTTP decode or with the wall time
of a request that has torch profiling enabled.

Phase 31 already addressed the actual Qwen3.8 long-context failure. With the
same Qwen3.8 checkpoint and a 32K cache-hit prefix, explicit gfx906 GPTQ plus
the opt-in split-KV path improved decode from `0.905636` to `1.336635 tok/s`
(+47.6%). The historical `0.238 tok/s` reference remains useful only as a
record of the pre-recovery geometry.

## Community Findings

Two upstream ROCm reports match the two distinct failure modes observed here:

- [Issue #50264](https://github.com/vllm-project/vllm/issues/50264) attributes
  long-context collapse in hybrid Qwen models to a paged-attention Triton
  fallback. It identifies split-KV decode as the safe class of remedy and
  explains why merely widening the native ROCm selector is invalid for
  256-wide heads and hybrid physical page sizes. This is the basis for the
  retained, explicit gfx906 split-KV option used by Phase 31.
- [Issue #49699](https://github.com/vllm-project/vllm/issues/49699) reports
  load-dependent ROCm W4A16 behavior with compile mode 3 and demonstrates that
  an ExLlama selection can help on its MI100 environment. It is a hypothesis,
  not a portable backend default: the reporter also notes compiled-kernel
  interference, and backend behavior depends on GPU and workload.

The latter warning matters because the existing Phase 37/38 harness starts
and stops the torch profiler around each fixed-length request. Its timings are
valuable for operator attribution, but profiler overhead makes their absolute
wall-clock throughput unsuitable for a release comparison.

## Ordinary HTTP Recheck

On GPU2, the existing v0.27.1 gfx906 image served the current
`cyankiwi/Qwen3.5-9B-AWQ-4bit` checkpoint with one GPU, FP16 KV, 100K context,
and 64 fixed completion tokens. The requests did not enable the torch
profiler. Two warmups preceded five measured requests.

| Linear selection | Mean measured completion throughput | Range | Outcome |
| --- | ---: | ---: | --- |
| automatic policy (`VLLM_ROCM_GFX906_PREFER_EXLLAMA=1`) | 60.052 tok/s | 59.952-60.259 | pass |
| explicit `gfx906_gptq` | 60.026 tok/s | 59.919-60.128 | pass |

The explicit run logged `Gfx906GPTQWNA16LinearKernel`; the automatic run
logged `ExllamaLinearKernel`. Both ran the routine text, one/two 256px image,
and JSON `3/3` gates successfully. No HTTP 5xx, OOM, RCCL/NCCL fatal,
xgrammar/FSM, traceback, or residual running/waiting request occurred.

The 0.04% difference is measurement noise. It does **not** establish a new
global winner between these two linear selections. It does establish that
ordinary v0.27 short decode is about 60 tok/s here, not 0.238 tok/s.

## Decisions

1. Keep the Qwen3.8 long-context preset model-specific: explicit
   `gfx906_gptq` plus `VLLM_ROCM_ENABLE_GFX906_SPLITKV=1`. Do not globally
   enable split-KV or weaken the native ROCm attention eligibility checks.
2. Keep Phase 37 profiler data for kernel attribution only. New performance
   gates must report an ordinary HTTP lane separately from a profiler lane.
3. Do not revive the rejected legacy V1/V2 format switch. Phase 04 already
   showed it was not an end-to-end improvement.
4. A future W4A16 recovery experiment is justified only if a fixed ordinary
   workload regresses. Its first candidate is a targeted compilation/Inductor
   interaction study motivated by issue #49699, but Phase 14 already showed
   that globally switching this stack to compile mode 0 is substantially
   slower. It must preserve compile mode 3 for the control and change only a
   measured component.

## Evidence

Raw benchmark artifacts are deliberately excluded from Git:

- automatic selection:
  `phase-39/phase21-auto/results/20260822T035253Z-normal-http`;
- explicit GPTQ:
  `phase-39/gfx906-gptq/results/20260822T035857Z-normal-http`.

Both temporary GPU2 containers were removed after the run. Production GPU0/
GPU1, port 8002, Router, Compose files, and production cache were not changed.
