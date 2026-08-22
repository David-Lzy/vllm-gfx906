# Phase 38: v0.27 ExLlama W4A16 Recovery

## Community Basis

The upstream ROCm performance report [vLLM issue #49699](https://github.com/vllm-project/vllm/issues/49699)
describes a W4A16 decode regression with compile mode 3 and reports that
selecting ExLlama improved the reporter's Qwen GPTQ throughput. That is a
credible, low-maintenance candidate for gfx906 because it changes an existing
backend selection rather than adding a new operator.

## Method

Use the same v0.27 image and GPU2 harness as Phase 37, changing only:

```text
--linear-backend exllama
```

The runner supports an explicitly empty backend for historical auto-selection
controls and a named backend for current images. It otherwise preserves the
model, FP16/KV settings, 100K context, C1/C8 payload, and routine multimodal
and structured-output gates.

## Result

The v0.27 server selected `ExllamaLinearKernel` successfully. Text, one/two
256px image, and JSON `3/3` gates passed with no HTTP 5xx, OOM, RCCL/NCCL
fatal, xgrammar/FSM, traceback, or stuck requests. Every measured response had
the same 27 prompt and 64 completion tokens as the Phase 37 controls.

| Load | v0.27 gfx906 GPTQ | v0.27 ExLlama | ExLlama change |
| --- | ---: | ---: | ---: |
| C1, 64 output tokens | 2.989087 s / 21.41 tok/s | 3.194494 s / 20.03 tok/s | -6.4% throughput |
| C8, 512 aggregate output tokens | 3.897610 s / 131.36 tok/s | 4.129603 s / 123.98 tok/s | -5.6% throughput |

The ExLlama selection is therefore functional but slower on this exact
gfx906/v0.27 image. Its C1 trace still contains the same
`gemm_half_q_half_gptq_4bit_kernel<true,1>` family, at 749.298 ms versus
709.205 ms for the explicit backend. This explains why selection alone does
not recreate the retained v0.23 path.

## Decision

Reject ExLlama as a v0.27 recovery setting. It does not meet the 5% retention
gate on either target and should not become a production default. The community
report remains useful evidence that W4A16/compiled-graph interactions are
hardware- and version-dependent, but this result prevents treating it as a
portable fix.

This does not reopen the historical Qwen3.8 `0.238 tok/s` long-context issue.
That distinct hybrid-attention failure is handled by the retained gfx906
split-KV composition documented in
[Phase 31](phase-31-qwen38-gptq-splitkv-composition.md): its same-model 32K
cache-hit rate improved from `0.906` to `1.337 tok/s` (+47.6%), with routine
multimodal and JSON gates passing. The original `0.238 tok/s` geometry result
was a historical cross-model reference, not the Phase 31 A/B baseline.
