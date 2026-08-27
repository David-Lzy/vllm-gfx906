# Phase 141: Qwen 27B unified-attention decode screen

## Decision

Rejected before a model-service build. vLLM's pure-Triton unified-attention
operator is numerically compatible with the tested Qwen 27B decode geometry,
but is substantially slower than the retained gfx906 SplitKV paged-decode
control. No runtime selector, serving patch, or production configuration is
added.

## Scope

The screen isolates one TP4 rank of the Qwen3.6/Qwen3.8 27B full-attention
path. It uses FP16 with 12 query heads, two KV heads (GQA 6:1), `head_dim=256`,
a 784-token hybrid physical page, 32,780 cached tokens, and the retained
29-way SplitKV control. Both arms consume the same logical K/V values and
block table. No checkpoint is loaded and no server is started.

## Result

| Arm | Median per-layer decode | Mean | p95 |
| --- | ---: | ---: | ---: |
| gfx906 SplitKV paged decode | 0.865 ms | 0.909 ms | 1.185 ms |
| Pure-Triton unified attention | 13.961 ms | 14.001 ms | 14.172 ms |

The outputs passed the FP16 tolerance check (`max_abs_error=1.53e-05`). The
candidate's median was `-1513.4%` relative to the control, rather than the
required `+20%` direct-kernel gain. That is approximately 16.1 times slower.

## Interpretation

The generic candidate processes this batch-one, two-KV-head decode geometry
without the retained 29-way SplitKV decomposition. At 32K context, that
parallel decomposition is material for occupying MI50 compute resources; this
screen establishes that a simple substitution to the generic operator is not a
viable workaround for the head-256 hybrid path.

The result is a geometry-specific rejection, not a claim about generic Triton
attention on other models or prefill workloads. Phase 140 already documented a
separate long-prefill experiment with a different Qwen3.5 route.

## Follow-up

Future Qwen 27B work should retain the existing compatible SplitKV path and
screen only designs that preserve or improve its long-sequence decomposition.
The source harness is retained on branch
`perf/gfx906-triton-unified-qwen27` for reproducibility.
