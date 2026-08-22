# Phase 44: v0.27 gfx906 Legacy QGEMM Composition

## Goal

Test whether the complete, historically faster gfx906 QGEMM/QDQ implementation
can recover normal-concurrency Qwen3.5 9B AWQ throughput on v0.27 without
changing the production worker, generic ROCm behavior, checkpoint format, or
OpenAI multimodal API.

The historical Qwen3.8 long-context `0.238 tok/s` result is not the target of
this phase. That hybrid-attention geometry has an independently validated
SplitKV recovery. This phase targets the v0.27 Qwen3.5 W4A16 C8 regression:
`157.900 tok/s` versus the v0.23 reference `216.703 tok/s`.

## Why A Complete Composition

The retained v0.23 extension contains a gfx906-specific W4A16 composition:

- 256-wide K blocks and a matching launch bound;
- legacy GFX906 QDQ bit-field sequence;
- the legacy reduction/accumulator behavior; and
- matching output initialization and reconstruction choices.

Previous experiments prove that neither the K=256 block change nor the
`fdot2` change alone is a safe performance recovery. A full gfx906-only
composition improved v0.26 operator and routine-server results in an earlier
phase, while still missing the older image's end-to-end floor. The v0.27
runtime has a different graph/runtime balance, so it must be measured anew.

## Candidate Design

The source candidate is a CMake option, default `OFF`, which is accepted only
for a single-architecture `PYTORCH_ROCM_ARCH=gfx906` build. It adds
`VLLM_GFX906_LEGACY_QGEMM` only to `_C_stable_libtorch`. CUDA, non-gfx906 ROCm,
and mixed-architecture builds retain upstream code. A mixed target must fail
configuration rather than silently selecting the legacy implementation.

The candidate is built from the validated Phase 43 PyTorch 2.11 diagnostic
base. It uses an isolated image, cache, port, and GPU2. It never mounts or
modifies production Compose, port 8002, GPU0/GPU1, or Router. The existing
model snapshot is mounted read-only, so it cannot change production weights or
the Hugging Face cache.

## Measurement Order

1. Confirm the build defines the option only for gfx906 and imports the native
   extension.
2. Run the actual-packed-weight GPTQ shape benchmark at M=1, 8, 27, and 190.
   Compare output error with the current stable operator and record warm median
   plus p95 latency.
3. Run the routine server gates: text, one 256px image, two 256px images, and
   JSON `3/3`.
4. Measure ordinary fixed-64 C1 and C8 after warmup. Do not use profiler wall
   time as serving throughput.
5. Record final `running=0` and `waiting=0`, then scan logs for fatal
   signatures before removing the temporary worker.

## Result

The option built successfully in a ROCm 7.2, PyTorch 2.11, Triton 3.6 image
and selected the explicit `Gfx906GPTQWNA16LinearKernel` on GPU2. The temporary
worker preserved the production-serving contract: Qwen3.5 9B AWQ, 100K context,
64-image limit, float16 KV cache, chunked prefill, one/two 256px image gates,
and JSON `3/3`. It never used GPU0/GPU1, port 8002, or Router, and it mounted
the shared model snapshot read-only.

The architecture restriction was also exercised directly: configuring the
same source with `VLLM_GPU_ARCHES=gfx90a` and
`VLLM_GFX906_LEGACY_QGEMM=ON` failed at CMake configuration with the intended
single-gfx906 diagnostic. The option therefore fails closed outside MI50's
target ISA.

| Measurement | Phase 43 control | Legacy composition | Change |
| --- | ---: | ---: | ---: |
| Actual-weight M=8 `mlp_gate_up` | 0.445 ms | 0.200 ms | +122.4% |
| Actual-weight M=8 `mlp_down` | 0.276 ms | 0.135 ms | +104.0% |
| Actual-weight M=8 `mlp_gate` | 0.246 ms | 0.131 ms | +87.3% |
| Fixed-64 C1 | 58.13 tok/s | 71.12 tok/s | +22.3% |
| Fixed-64 C8 profile run | 161.20 tok/s | 231.78 tok/s | +43.8% |
| Fixed-64 C8, five HTTP rounds | n/a | 237.28 tok/s median | +9.5% vs v0.23 |

The five C8 rounds measured `201.70`, `239.51`, `234.69`, `238.87`, and
`237.28 tok/s`; the lower first round is a warm remainder, while the following
four stayed within 2.1%. The v0.23 production reference is `216.703 tok/s`,
so the warmed candidate clears the 95% release-consideration floor and exceeds
the old C8 reference by 9.5%. The GPTQ repeat deltas stayed within the expected
non-bitwise-deterministic K-split range, and all API outputs were non-empty and
semantically valid. Final request metrics were `running=0`, `waiting=0`; no
OOM, HTTP 5xx, xgrammar/FSM, RCCL/NCCL fatal, or traceback signature occurred.

Raw build, microbenchmark, server, and HTTP-repeat evidence is retained outside
Git under the Phase 44 build root. The temporary service was removed after the
gates; production remained healthy and unchanged.

## Retention Gates

- The complete QGEMM candidate must preserve numerical tolerance on every
  measured GPTQ shape and never select on a non-gfx906 build.
- It must improve the GPTQ M=8 dominant shape by at least 15% versus the
  current Phase 43 native operator.
- It must improve ordinary v0.27 C8 by at least 5% with no C1, image, or JSON
  regression. Release consideration additionally requires at least 95% of the
  v0.23 C8 reference.
- Reject on build/import failure, output mismatch, HTTP 5xx, OOM,
  xgrammar/FSM, RCCL/NCCL fatal, traceback, or residual running/waiting work.

## Decision

Retain the default-off, gfx906-only option as the first v0.27 candidate that
recovers the ordinary Qwen3.5 AWQ C8 regression. It is eligible for the later
TP1x2 Router canary comparison, but it is not a production promotion by itself:
that comparison must still use the production topology, workload, rollback
checklist, and soak gate.

## Community Boundaries

Upstream ROCm native W4A16 and AITER work target newer RDNA3/MI300 hardware;
their wave32/MFMA assumptions do not apply to MI50's gfx906 wave64 ISA. The
public compile-mode W4A16 regression report is useful evidence that a generic
graph flag is not a substitute for kernel attribution, but it does not supply
a portable kernel. This phase therefore adapts only the locally proven,
architecture-restricted source bundle and retains all API/quality gates.
