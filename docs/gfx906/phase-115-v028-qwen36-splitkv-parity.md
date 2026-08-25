# v0.28 Qwen3.6 SplitKV Parity

## Scope

This experiment verifies that the retained gfx906 SplitKV recovery on the
v0.28 line applies to both supported Qwen 27B AWQ checkpoints. It uses Qwen3.6
27B AWQ with tensor parallelism two, a 100K context limit, FP16 KV cache, and
the explicit gfx906 W4A16 backend. The production Qwen3.5 service is not part
of the test.

## Result

The v0.28 image passed text, one-image, two-image, and JSON-constrained
requests. Queues drained after testing and the bounded log scan found no OOM,
structured-output, or ROCm/RCCL fatal condition.

| Measurement | Retained v0.27 | v0.28 gfx906 | Change |
| --- | ---: | ---: | ---: |
| Fixed-128 C1 median | 43.97 tok/s | 44.74 tok/s | +1.75% |
| 32K cache-hit fixed-128 mean | 11.84 tok/s | 12.27 tok/s | +3.67% |

The 32K result is effectively identical to the corresponding Qwen3.8 v0.28
result of 12.28 tok/s. This confirms that the 784-token hybrid cache page,
16-token logical block, and eight-row SplitKV route are applicable to both
models rather than a checkpoint-specific workaround.

## Decision

Keep the SplitKV path as a default-off development option on the v0.28 gfx906
branch. It improves the Qwen3.6 reference without a routine multimodal or
structured-output regression. It does not change the current Qwen3.5 9B
production deployment; a separate Router canary remains necessary for that
model before any runtime promotion.
