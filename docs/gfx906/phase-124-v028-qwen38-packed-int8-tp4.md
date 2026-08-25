# Phase 124: Qwen3.8 Packed-INT8 TP4 Revalidation

## Purpose

Revalidate the Qwen3.8 27B AWQ copy-on-write profile after the v0.28 gfx906
INT8 GPTQ numeric-parity repair. The profile packs only the embedding and
lm-head tables to INT8; the remaining checkpoint stays on the standard AWQ
path. It is a development profile, not a replacement for the Qwen3.5 9B
production service.

## Environment

- vLLM image: `local/vllm-gfx906:v0.28.0-phase123-qwen38-int8-semantic`
- Source fix: `e7593a1689` (`gfx906` INT8 GPTQ FP32 accumulation restoration)
- Hardware: four AMD MI50 GPUs, TP4
- Runtime: `gfx906_gptq`, FP16 KV cache, no MTP, 100K context, eight
  sequences, 8,192 batched tokens, prefix caching and chunked prefill
- Media contract: 64 images maximum, video disabled, 16 MiB maximum pixels

The routine development gate intentionally used text, one 256-square image,
two 256-square images, JSON `3/3`, and fixed-128 C1/C8 decode. It does not
claim capacity coverage for large images or 32/64-image batches.

## Result

Startup completed in `633.501s` and provisioned `1,267,164` GPU KV-cache
tokens, or `12.67x` the 100K context contract. Text, image, and JSON gates
passed with coherent non-empty output. Error scans found no HTTP 5xx, OOM,
traceback, xgrammar/FSM, illegal instruction, RCCL/NCCL fatal, or residual
running/waiting request.

| Metric | Standard v0.28 Qwen3.8 AWQ TP4 | Packed-INT8 v0.28 TP4 | Delta |
| --- | ---: | ---: | ---: |
| C1 fixed-128 tok/s | 52.13 | 52.94 | +1.55% |
| C8 fixed-128 aggregate tok/s | 217.75 | 222.33 | +2.10% |

The packed profile result is the mean of two independent median measurements:
`52.9490/222.7898` and `52.9236/221.8669` tok/s for C1/C8 respectively.

## Interpretation

This is a retained positive result: it improves the standard v0.28 Qwen3.8
TP4 baseline, especially at C8, while preserving the development gate. It is
still below the historical v0.27 retained profile (`53.14` C1 and `229.43` C8),
so it is not evidence of full historical-profile parity.

Current v0.28 SplitKV bounds the split count at `16`. Consequently, this is a
current-code TP4 revalidation and not an exact reproduction of the older
29-split profile.

## Decision

Keep this checkpoint/profile for Qwen3.8 development benchmarks. It does not
change the Qwen3.5 production model, Router topology, or production Compose
configuration. The production service was restored after the maintenance
window and passed health, model discovery, text, one-image, and JSON smoke;
its final queue state was zero running and zero waiting requests.
