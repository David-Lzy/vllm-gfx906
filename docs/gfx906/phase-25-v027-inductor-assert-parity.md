# Phase 25: v0.27 Inductor Assertion Parity

## Scope

Phase 25 tested whether v0.27's omission of explicit Inductor runtime-assertion
settings explained the remaining small-request Qwen3.5 regression. The retained
v0.23 worker records `size_asserts`, `alignment_asserts`, and `scalar_asserts`
as `false`; v0.27 with PyTorch 2.13 otherwise relies on its assert-once
behavior.

The test ran only on development GPU2. It retained the Phase 24 Qwen3.5 9B
AWQ configuration, including ExLlama W4A16, `TRITON_ATTN`, Flash Attention for
the vision encoder, Triton/FLA GDN prefill, 100K context, FP16 KV cache, and
the same CUDA-graph capture set. Production GPU0/GPU1, Router, and port 8002
were not changed.

## Result

The candidate accepted and logged:

```json
{"inductor_compile_config":{"size_asserts":false,"alignment_asserts":false,"scalar_asserts":false}}
```

Routine health, model discovery, text, one/two 256-square image requests, and
JSON constrained output `3/3` all passed. No OOM, traceback, xgrammar/FSM, or
RCCL/NCCL fatal signature was found, and the final engine metrics reported zero
running and waiting requests.

| Scenario | Retained v0.23 tok/s | Phase 24 tok/s | Phase 25 tok/s | Phase 25 vs Phase 24 |
| --- | ---: | ---: | ---: | ---: |
| Text C1 | 75.25 | 62.01 | 62.16 | +0.2% |
| One image C1 | 66.76 | 58.41 | 58.73 | +0.5% |
| Two images C1 | 63.29 | 55.82 | 55.68 | -0.3% |
| Text C8 warm | 76.24* | 125.40 | 125.02 | -0.3% |

`*` The retained C8 record was a cold-shape diagnostic and is not a direct
release comparison.

## Decision

Rejected as a performance recovery. Explicitly disabling the three assertions
is accepted and stable, but the change is within measurement noise and does not
close the 95% retained-throughput floor. Do not add it to the production
configuration solely for performance.

The isolated runner now exposes an optional `COMPILATION_CONFIG` input so later
compiler experiments can preserve a full record of their configuration without
editing the production compose.

## Evidence

Local, non-versioned evidence:

- `/mnt/disk2/vllm-gfx906-build/phase-25/results/20260821T130000Z-v027-inductor-assert-parity`
- Retained v0.23 control:
  `/mnt/disk2/vllm-gfx906-build/phase-23/results/20260821T114748Z-small-mm-parity`
- Phase 24 control:
  `/mnt/disk2/vllm-gfx906-build/phase-24/results/20260821T122500Z-v027-triton-attn`
