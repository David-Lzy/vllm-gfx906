# v0.27 gfx906 exploration roadmap

## Purpose

Evaluate useful upstream v0.27 behavior without weakening the validated
gfx906 production lineage. This is an exploration program, not a commitment to
ship v0.27 on MI50.

## Sequence

| Stage | Scope | Promotion gate |
| --- | --- | --- |
| Selected backports | Completed: self-contained multimodal and hybrid/Mamba improvements on v0.26 | No candidate cleared the independent performance gate; see the Phase 16 report |
| CPU scheduling | Active: core affinity and thread-budget comparison on isolated workers | A reproducible latency or throughput gain without tail regression |
| Cost-aware routing | Sidecar validation against the existing Router policy | Better mixed-workload tail latency without API or streaming regressions |
| Triton 3.7 port | Minimal gfx906 compiler enablement for the full v0.27 stack | Compiler microtests and routine Qwen3.5 multimodal parity pass |

Only one stage is active at a time. The ordinary release requirements still
apply: deterministic routine quality checks, no stranded work, no fatal ROCm
communication errors, and a performance result that justifies a canary.

## Boundaries

- The existing production endpoint and production Router remain the rollback
  path throughout this program.
- Qwen3.5 9B AWQ is the primary multimodal compatibility model. Qwen3.8 27B
  AWQ is a separate hybrid/Mamba compatibility target, not a production default.
- Routine gates use text, one 256-square image, two 256-square images, and
  JSON-constrained output. Large image-count and grid workloads remain special
  capacity tests.
- A full upstream v0.27 runtime is deferred until a standalone Triton 3.7
  gfx906 port reaches compiler and server parity.

## Deferred work

MoE expert parallelism remains an independent maintenance-window experiment.
It must compare ordinary TP4 with expert parallelism on the same four GPUs and
cannot use an isolated two-GPU result as a conclusion.
