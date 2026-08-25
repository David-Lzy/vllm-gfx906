# Phase 127: Qwen3.5 packed-INT8 C1 attribution

## Result

**No component-level source candidate.** The Phase 126 copy-on-write INT8
embedding/output-head profile remains a useful saturated-throughput experiment,
but its C1 regression does not justify a custom gfx906 embedding or lm-head
kernel. The measured embedding path is negligible and the packed output head
is faster than the baseline FP16 logits path in the captured decode geometry.

## Scope

- Runtime: `v0.28.0+gfx906.phase110.legacyqgemm` with explicit
  `gfx906_gptq`.
- Model: `cyankiwi/Qwen3.5-9B-AWQ-4bit` and the Phase 126 copy-on-write
  profile that packs only `embed_tokens` and `lm_head` to symmetric INT8,
  group size 128.
- Hardware: one isolated MI50/gfx906 worker. Production workers, Router, port,
  and Compose configuration were not changed.
- Runtime contract: 100K context, FP16 KV cache, eight sequences, 32,768
  batched tokens, images 64/video 0, and no speculation.

The worker used a bounded Torch eager profiler after multimodal warmup. Eager
mode deliberately disables compilation and graph replay, so this evidence is
for operator attribution only, not for reporting serving throughput.

## Attribution

| Component | Packed trace result | Standard trace result | Interpretation |
| --- | ---: | ---: | --- |
| Selected-token INT8 embedding gather | 13.4us over four active steps | N/A | Far below a meaningful C1 budget; reject a bespoke gather kernel. |
| Vocabulary output head | 6.00ms over four active steps | 9.37ms FP16 `LLMM1` path | INT8 output-head execution is not slower in the captured `M=1, K=4096, N=248320` geometry. |
| Aggregate 4-bit GPTQ work | 85.30ms | 65.25ms | Not a comparable A/B figure: captured prefills had `M=29` and `M=22` respectively. |

The packed output head was verified in the trace as
`gptq_8bit_kernel` with the full vocabulary geometry. The standard FP16
`LLMM1` time includes the corresponding logits route. The faster packed output
head and microsecond-scale embedding gather directly reject the two original
implementation hypotheses.

## Control limitation

The two profiler captures did not have identical prompt-prefill geometries,
and a second bounded start/stop cycle did not produce a fresh vLLM Torch trace.
Neither aggregate trace time nor eager-mode request throughput is therefore
used to revise Phase 126's normal-mode C1/C8 measurements. A future
investigation of the roughly four-to-five-percent C1 difference must use a
fresh, normal-mode interleaved A-B-A worker sequence.

## Decision

Keep the Phase 126 checkpoint builder and the development-only C8 evidence;
do not change production selection. Do not add a custom HIP/Triton kernel for
embedding gather or the INT8 lm head. Any later C1 work should first prove a
repeatable normal-mode regression and then attribute it to an unchanged
transformer/graph/scheduler path rather than this component pair.
