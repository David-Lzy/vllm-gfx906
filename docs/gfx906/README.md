# AMD gfx906 support

This directory tracks the MI50/MI60 (`gfx906`) maintenance work for this fork.
It contains the public evidence for the active experimental v0.27-era line as
well as the earlier v0.26 integration record. Historical version targets are
kept so a result can be compared with its actual source base.

## Status

- Validated reference lineage: `gfx906/v0.23.1rc0.x`
- Active experimental line: v0.28 gfx906 work with retained, narrow patches
- Historical integration target: upstream `v0.26.0`
- Primary parity model: Qwen3.5 9B AWQ multimodal
- Target hardware: four AMD MI50 GPUs, tested as independent TP1 workers and,
  where required, TP4
- Release policy: `main` receives only versions validated on real gfx906
  hardware and reviewed with a reproducible canary/rollback record

This work remains experimental until the matching release and production-canary
gates are complete. Model weights, caches, build outputs, credentials, and
machine-specific deployment configuration do not belong in this repository.

## Documents

- [v0.26 roadmap](roadmap-v0.26.md)
- [v0.27 exploration roadmap](roadmap-v027.md)
- [v0.27 selected-backport report](phase-16-v027-selected-backports.md)
- [Cost-aware Router evaluation](phase-18-cost-aware-router.md)
- [Qwen3.8 Hybrid Mamba state-copy tiling screen](phase-57-qwen38-mamba-state-copy-tiling.md)
- [Qwen3.5 multimodal prefix-cache transport screen](phase-58-qwen35-mm-prefix-cache-transport.md)
- [Qwen Mamba prefill-checkpoint screen](phase-59-qwen-mamba-prefill-checkpoint-screen.md)
- [Positive-gain retention policy](positive-gain-retention.md)
- [Evidence lifecycle and branch ledger](evidence-lifecycle.md)
- [Phase 64 positive-gain revalidation](phase-64-positive-gain-revalidation.md)
- [Phase 65 legacy QGEMM row-tiling result](phase-65-gfx906-legacy-qgemm-row-tiling.md)
- [Phase 66 exact-M8 legacy QGEMM dispatch result](phase-66-gfx906-legacy-qgemm-c8-row4-dispatch.md)
- [Phase 67 exact-M8 legacy QGEMM row-2 screen](phase-67-gfx906-legacy-qgemm-c8-row2-sweep.md)
- [Phase 68 exact-M8 legacy QGEMM call attribution](phase-68-gfx906-qgemm-m8-call-attribution.md)
- [Phase 69 exact-M8 legacy QGEMM remaining-row sweep](phase-69-gfx906-qgemm-m8-row-sweep.md)
- [v0.28 Qwen3.6 SplitKV parity](phase-115-v028-qwen36-splitkv-parity.md)
- [v0.28 Qwen3.5 Router canary](phase-116-v028-qwen35-router-canary.md)
- [v0.28 Qwen3.5 same-GPU residual attribution](phase-117-v028-qwen35-router-residual-attribution.md)
- [v0.28 Qwen 27B TP4 standard-AWQ baseline](phase-118-v028-qwen27-tp4-baseline.md)
- [Qwen3.8 packed-INT8 embedding port](phase-119-v028-qwen38-int8-embedding-port.md)
- [Qwen3.8 packed-INT8 rule-precedence screen](phase-120-v028-qwen38-int8-rule-precedence.md)
- [Qwen3.8 packed-INT8 GDN-mapper control](phase-121-v028-qwen38-int8-mapper-ab.md)
- [Qwen3.8 packed-INT8 loader parity](phase-122-v028-qwen38-int8-loader-parity.md)
- [Qwen3.8 packed-INT8 numeric parity and precision fix](phase-123-v028-qwen38-int8-numeric-parity.md)
- [Qwen3.8 packed-INT8 TP4 revalidation](phase-124-v028-qwen38-packed-int8-tp4.md)
- [Qwen3.6 packed-INT8 TP4 portability](phase-125-v028-qwen36-packed-int8-tp4.md)
- [Qwen3.5 packed-INT8 TP1 portability](phase-126-v028-qwen35-packed-int8-tp1.md)
- [Qwen3.5 packed-INT8 C1 attribution](phase-127-v028-qwen35-packed-int8-c1-attribution.md)
- [Qwen3.6 MRoPE fusion backport screen](phase-128-qwen36-mrope-fusion-v028.md)
- [Triton 3.6 conditional-pointer intersection](phase-129-triton36-scf-pointer-intersection.md)
- [Qwen3.6 packed-INT8 fused QK/RoPE TP4 composition](phase-135-qwen36-packed-int8-fused-tp4.md)
- [Qwen3.8 TP4 SplitKV 29-partition rebase](phase-136-v028-qwen38-tp4-splitkv29-rebase.md)
- [Qwen3.6 TP4 SplitKV 29-partition port](phase-137-v028-qwen36-tp4-splitkv29-port.md)
- [Qwen3.5 multimodal truncation and profiling](phase-131-qwen35-mm-truncation-profiling.md)
- [Compatibility matrix](compatibility-matrix.md)
- [Patch ledger](patch-ledger.md)
- [Benchmark protocol](benchmark-protocol.md)
- [Release process](release-process.md)
- [Current production partial baseline](baselines/production-v0231-partial-20260819.md)
- [Phase 130: Qwen3.6 ROCm fused QK/RoPE dispatch](phase-130-qwen36-rocm-fused-qk-rope.md)

## Priorities

1. Preserve Qwen3.5 AWQ text and multimodal parity on the active gfx906 line.
2. Prevent performance regressions against the validated reference lineage.
3. Keep Qwen3.6/Qwen3.8 27B compatibility and throughput work isolated from
   the small-model production path.
4. Evaluate optional DFlash, MoE, TurboQuant KV, and FP8 KV features only with
   their own correctness and MI50 performance evidence.
5. Publish a reproducible image only after hardware validation and canary soak.

The v0.27 work is a separate exploration track. It first evaluates narrowly
selected upstream backports on the validated v0.26 gfx906 base, then addresses
the Triton 3.7 gfx906 compiler gap. It does not authorize an automatic runtime
upgrade or production replacement.
