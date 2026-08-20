# v0.26 roadmap

## Objective

Produce a maintainable gfx906 release based on upstream vLLM v0.26.0. The
first release must replace the current Qwen3.5 9B AWQ multimodal service
without reducing its supported API surface, context limits, image limits, or
stability.

## Phase sequence

| Phase | Scope | Exit gate |
| --- | --- | --- |
| 00 | Repository bootstrap | Remotes, baseline tag, integration branch, docs, and local control plane exist |
| 01 | Production baseline | Reproducible text, one/two 256-square image, JSON, utilization, and quality baseline captured |
| 02 | Stock v0.26 gfx906 build | Text model starts and completes smoke tests without optional patches |
| 03 | Qwen3.5 AWQ multimodal parity | OpenAI text, one/two 256-square image, and JSON requests pass on the production route |
| 04 | Performance regression | Core throughput is at least 95% of the validated baseline with no material quality loss |
| 05 | DFlash | Retained only with measured end-to-end benefit and stable output |
| 06 | MoE tensor parallelism | 35B-A3B BF16 TP4 text and multimodal paths are stable |
| 07 | TurboQuant KV | Long-context capacity modes pass quality and stability gates |
| 08 | FP8 KV | Optional E4M3 KV mode is retained only when end-to-end results justify it |
| 09 | MoE expert parallelism | EP is retained only when it is stable and competitive with TP4 |
| 10 | Release and canary | Image is published, soaked, canaried, and has an immediate rollback path |

## Dependency rules

- Phases 02 through 04 form the minimum release path.
- Optional features must not block the base v0.26 release.
- DFlash, MoE, TurboQuant KV, FP8 KV, and expert parallelism use independent
  branches and independent benchmark evidence.
- Quantized MoE work starts only after the BF16 MoE path is stable.
- Expert parallelism starts only after ordinary TP4 MoE is stable.

## Release gates

- Text, one/two 256-square image, and JSON-constrained tests pass.
- The supported context and multimodal limits remain available.
- 32/64-image and 4096-grid capacity tests run only when a release changes the
  relevant capacity, scheduler, KV-cache, or media-processing path.
- There are no OOMs, HTTP 500 responses, empty outputs, runaway repetition,
  xgrammar/FSM failures, fatal RCCL errors, or stranded running/waiting work.
- Core image and text throughput reaches at least 95% of the validated
  production baseline.
- A 30-60 minute canary soak completes before production replacement.

## Non-goals for the first release

- FP8 weight acceleration on MI50
- A broad model-support claim beyond the tested compatibility matrix
- Automatic upstream merges into `main`
- Public CI jobs that execute untrusted code on the private MI50 host
