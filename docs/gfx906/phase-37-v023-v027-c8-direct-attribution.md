# Phase 37: v0.23-v0.27 Direct C8 Attribution

## Goal

Replace the historical cross-topology comparison with a direct GPU2 A/B between
the retained mobydick worker and the current v0.27 worker. This test holds the
checkpoint, request payload, fixed output length, context/KV settings, and
single-GPU hardware constant.

## Method

Both servers used `cyankiwi/Qwen3.5-9B-AWQ-4bit`, FP16 weights and KV cache,
100K maximum context, eight sequences, 32,768 batched tokens, prefix caching,
and one GPU. Each request produced exactly 64 completion tokens from a
27-token text prompt. The routine text, one/two 256px image, and JSON `3/3`
gates ran after the C1/C8 profiler capture.

The v0.23 control used its automatic linear backend, which selected
`ExllamaLinearKernel`. v0.27 used the explicitly retained
`gfx906_gptq` backend. Production GPUs, Router, Compose, model cache, and port
8002 remained untouched.

## Result

| Load | v0.23 automatic ExLlama | v0.27 gfx906 GPTQ | Change in v0.27 |
| --- | ---: | ---: | ---: |
| C1, 64 output tokens | 0.947107 s / 67.57 tok/s | 2.989087 s / 21.41 tok/s | -68.3% throughput |
| C8, 512 aggregate output tokens | 4.992770 s / 102.55 tok/s | 3.897610 s / 131.36 tok/s | +28.1% throughput |

All three gate classes passed in both controls. Output usage was identical:
27 prompt, 64 completion, and 91 total tokens for every measured request. No
OOM, HTTP 5xx, RCCL/NCCL fatal, xgrammar/FSM, traceback, or residual request
occurred.

The modes are deliberately reported separately: v0.27 has a real C1 decode
regression but a higher aggregate C8 throughput under this harness. A single
score would conceal that operational tradeoff.

## Attribution

The C1 trace locates the obvious difference in the W4A16 decode path:

| C1 trace item | v0.23 | v0.27 |
| --- | ---: | ---: |
| GPTQ 4-bit GEMM kernel time | 7.694 ms | 709.205 ms |
| `gpu_model_runner: forward` | 198.929 ms | 4,833.704 ms |
| `hipGraphLaunch` host time | 48.727 ms | 2,186.794 ms |

Profiler wall-clock aggregation includes asynchronous graph activity, so these
numbers are operator attribution rather than a second latency measurement.
They nevertheless identify a practical next question: whether the v0.27
compiled graph/linear interaction can recover C1 without losing its C8
advantage. The following isolated ExLlama selection experiment answers the
lowest-risk version of that question.

## Decision

Do not add another generic HIP GPTQ kernel based only on a historical
throughput gap. The controlled result shows that v0.27 behavior is
load-dependent; any recovery must preserve C8 and demonstrate at least a 5%
end-to-end gain on its intended target.
