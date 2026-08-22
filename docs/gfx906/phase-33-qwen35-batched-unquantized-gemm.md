# Phase 33: gfx906 Batched Unquantized Decode GEMM

## Scope

Phase 32 identified generic rocBLAS/Tensile unquantized-linear launches as the
dominant incremental cost in the Qwen3.5 9B AWQ C8 decode path. This phase adds
a narrow MI50-only native kernel for batch sizes two through eight. It does not
change the existing single-token `LLMM1` implementation, production image, or
production service.

The implementation is `LLMMB4`: one workgroup reads each four-row weight
fragment once and reuses it across up to four activation rows. Batches five
through eight use a second y-grid slice. The kernel has no atomic reduction and
does not invoke `wvSplitK`, whose gfx906 build branch is intentionally
unimplemented.

## Shape Guard

The dispatch is deliberately selective. MI50 microbenchmarks show that the
fused kernel wins for Qwen's 4096-wide projections and its large vocabulary
head, but loses to rocBLAS for the 12288-wide gate/up MLP projection at C8.
Therefore the native path accepts:

- batch two through four, when the existing FP16/BF16, divisibility, and
  no-bias requirements hold;
- batch five through eight only for output widths at most 8192 or at least
  65536.

The 12288-wide C8 MLP shape stays on the normal fallback. This is a measured
compatibility/performance guard, not a model-specific hardcode.

## Microbenchmarks

All figures are median single-GPU MI50 timings for FP16 `A @ B.T` with
`K=4096`:

| Output width | Batch | rocBLAS / torch | LLMMB4 | Result |
| --- | ---: | ---: | ---: | --- |
| 4096 | 8 | 0.314 ms | 0.130 ms | 2.41x faster |
| 12288 | 8 | 0.357 ms | 0.417 ms | 0.86x; guarded out |
| 248320 vocab head | 8 | 8.185 ms | 7.842 ms | 1.04x faster |

For the vocabulary-head case, `LLMMB4` is bitwise identical to applying the
existing `LLMM1` independently to each row. Both differ from `torch.linear`
by the same FP16 reduction-order error, so the batched kernel adds no new
numerical behavior beyond the established gfx906 single-token path.

## Server Result

The isolated Qwen3.5 9B AWQ server used the Phase 30 serving configuration:
FP16, 100K context, `max-num-seqs=8`, 32,768 batched tokens, explicit
`gfx906_gptq`, current multimodal settings, and a 256-square image fixture.

| Scenario | Phase 30 v0.27 | Phase 33 | Change |
| --- | ---: | ---: | ---: |
| Text C1 | 59.97 tok/s | 59.99 tok/s | +0.0% |
| Text C8 | 125.27 tok/s | 157.57 tok/s | +25.8% |
| One image C1 | 55.24 tok/s | 55.51 tok/s | +0.5% |
| Two images C1 | 51.28 tok/s | 51.65 tok/s | +0.7% |
| Retained v0.23 Text C8 reference | 223.37 tok/s | 157.57 tok/s | 70.5% of reference |

Text, one-image, two-image, and JSON constrained output `3/3` passed. The
server ended with zero running and waiting requests; logs had no OOM, HTTP
5xx, xgrammar/FSM failure, or RCCL/NCCL fatal. Targeted ROCm tests passed
`10/10`.

The candidate recovers 32.30 tok/s of the 98.10 tok/s C8 gap between Phase 30
v0.27 and the retained v0.23 worker: 32.9% of that gap. It does not meet the
release parity floor, so it remains a development improvement rather than a
production promotion.

## Community Context And Next Work

[vLLM issue #52631](https://github.com/vllm-project/vllm/issues/52631) shows a
separate ROCm dispatch hole for `m=1` MoE shared-expert gates. On a newer AMD
GPU it accounts for 13.7% of decode and a targeted replacement improves decode
15.5%. Qwen3.5 9B is dense, so that is not the explanation for this C8 result.
It is, however, a high-value independent candidate for Qwen3.8 MoE/hybrid
evaluation, where the same `m=1` shared-expert structure may exist.

The next experiment should profile the Qwen3.8 candidate at fixed decode and
long-context decode, record every `m=1` fallback, and only then test an FP32
accumulating dot-product or a padded narrow GEMM. No upstream RDNA3/MI300
skinny-GEMM path should be enabled on gfx906 without a native implementation
and numerical test.

## Evidence

Raw artifacts are intentionally not versioned. Relevant run roots are:

- Phase 32 profiler: `20260821T233617Z-qwen35-c8`;
- Phase 33 profiler/gates: `20260822T002307Z-qwen35-c8`;
- Phase 33 throughput: `20260822T004353Z-b4`.
