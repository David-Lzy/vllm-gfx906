# AMD gfx906 support

This directory tracks the MI50/MI60 (`gfx906`) maintenance work for this fork.
The project targets a validated vLLM v0.26 release while keeping the existing
production-compatible branch available for rollback.

## Status

- Validated production lineage: `gfx906/v0.23.1rc0.x`
- Integration target: upstream `v0.26.0`
- Primary parity model: Qwen3.5 9B AWQ multimodal
- Target hardware: four AMD MI50 GPUs, tested as independent TP1 workers and,
  where required, TP4
- Release policy: `main` receives only versions validated on real gfx906
  hardware

The v0.26 work is experimental until the release and production-canary gates
are complete. Model weights, caches, build outputs, credentials, and
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
- [Compatibility matrix](compatibility-matrix.md)
- [Patch ledger](patch-ledger.md)
- [Benchmark protocol](benchmark-protocol.md)
- [Release process](release-process.md)
- [Current production partial baseline](baselines/production-v0231-partial-20260819.md)

## Priorities

1. Build stock v0.26 for gfx906 and establish a text-only floor.
2. Restore Qwen3.5 AWQ multimodal parity.
3. Prevent performance regressions against the validated production lineage.
4. Port optional DFlash, MoE, TurboQuant KV, and FP8 KV features separately.
5. Publish a reproducible image only after hardware validation and canary soak.

The v0.27 work is a separate exploration track. It first evaluates narrowly
selected upstream backports on the validated v0.26 gfx906 base, then addresses
the Triton 3.7 gfx906 compiler gap. It does not authorize an automatic runtime
upgrade or production replacement.
