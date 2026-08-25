# Phase 126: Qwen3.5 9B packed-INT8 TP1 portability

## Result

**Retained as a development-only concurrency profile.** On MI50/gfx906, a
copy-on-write profile that packs only Qwen3.5's embedding and output-head
tables to INT8 passed text, one/two-image, and deterministic JSON gates. It
improves saturated TP1 throughput, but regresses C1. It is therefore useful
evidence for a future throughput-oriented composition, not a production
replacement for the current Qwen3.5 service.

## Scope

- Runtime: `v0.28.0+gfx906.phase110.legacyqgemm`, with the retained
  `gfx906_gptq` W4A16 transformer route.
- Model: `cyankiwi/Qwen3.5-9B-AWQ-4bit`.
- Profile: symmetric packed INT8, group size 128, for only
  `model.language_model.embed_tokens.weight` and `lm_head.weight`. All AWQ
  transformer weights remain the source checkpoint.
- Contract: FP16 KV cache, 100K model length, eight sequences, 32,768 batched
  tokens, images 64/video 0, and no speculation.
- Test: fixed-128 text generation and cache-busted one/two 256-square-image
  requests. The two-worker result uses two independent TP1 workers and a
  temporary round-robin vLLM Router.

The converted tensors had relative reconstruction errors of `0.00578`
(embedding) and `0.00657` (lm head), both below the phase `0.01` limit. The
copy-on-write checkpoint keeps all other source files linked read-only.

## Gates

| Gate | Result |
| --- | --- |
| Text smoke | Passed; exact `healthy` response |
| One/two 256-square images | Passed; correct image-text recognition and identical-image comparison |
| JSON constrained output | Passed `3/3`; exact `{"ok":true}` |
| Worker metrics after tests | Both workers: `running=0`, `waiting=0` |
| Fatal-log scan | No OOM, HTTP 5xx, xgrammar/FSM, or RCCL/NCCL fatal signature |

## Matched TP1 result

Each side used a separate development MI50, the same image and launch
contract, and the same fixed-128 requests. Two repeats were used for text C1
and C8. Positive values mean the packed profile is faster.

| Workload | Standard AWQ | Packed INT8 | Delta |
| --- | ---: | ---: | ---: |
| Text C1 completion throughput | 74.78 tok/s | 71.65 tok/s | -4.2% |
| Text C8 completion throughput | 259.49 tok/s | 280.25 tok/s | +8.0% |
| One image C1 completion throughput | 66.38 tok/s | 63.19 tok/s | -4.8% |
| One image C8 completion throughput | 202.22 tok/s | 221.17 tok/s | +9.4% |
| Two images C1 completion throughput | 62.42 tok/s | 59.59 tok/s | -4.5% |
| Two images C8 completion throughput | 174.06 tok/s | 180.85 tok/s | +3.9% |

The direction is consistent with a bandwidth-saving optimization that needs
batching to amortize its pack/dequant and dispatch cost. It must not become the
default for latency-sensitive single-request traffic.

## Two-worker router confirmation

The temporary Router sent C16 to two packed TP1 workers. These values are
reported as a topology confirmation, not a replacement for the individual
worker A/B because an equivalent two-standard-worker Router control was not
started in the same run.

| Workload | Packed TP1x2 Router result |
| --- | ---: |
| Text C16 completion throughput, repeat 1 | 524.46 tok/s |
| Text C16 completion throughput, repeat 2 | 558.27 tok/s |
| Text C16 mean | 541.37 tok/s |
| One image C16 completion throughput | 431.79 tok/s |
| Two images C16 completion throughput | 354.71 tok/s |

For context only, twice the matched single-worker standard C8 value is
`518.98 tok/s` for text, `404.44 tok/s` for one image, and `348.12 tok/s` for
two images. The Router confirmation has the same positive saturated direction,
but a promotion comparison still needs an interleaved two-worker control.

## Decision

Keep the converter and the small copy-on-write profile on the development
volume. Remove temporary workers, Router, and compile caches after the result
is recorded. Do not alter production Compose, the v0.27 service, or the public
model alias.

Any production-canary proposal must first show at least five-percent
end-to-end improvement on the actual mixed workload without degrading the
latency-sensitive C1 path. A policy that selects this profile only for known
high-concurrency work is a separate routing/composition task.
