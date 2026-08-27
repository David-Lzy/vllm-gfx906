# v0.28 Qwen3.8 Packed-INT8 Mapper Screen

## Scope

Phase 121 tested whether the inner Qwen GDN stacked-weight mapper could cause
the failed optional Qwen3.8 packed-INT8 embedding/output-head profile. The
screen restored the Phase 119 packed-embedding construction and removed only
that mapper for an isolated v0.28 TP2 candidate. This mapper fuses the
checkpoint's separate `in_proj_a` and `in_proj_b` values into `in_proj_ba`.

The rest of the contract was unchanged: two MI50 GPUs, 100K context, FP16 KV,
eight sequences, 8,192 batched tokens, no MTP, and the retained gfx906
SplitKV selector. No production file or model shard was changed.

## Result

The candidate failed during worker initialization, before health or any
request could run. The loader reported that
`layers.0.linear_attn.in_proj_a` has no destination in `Qwen3_5Model`.

This is an architectural compatibility failure, not a performance result. The
same GDN mapping exists in the Phase 82 v0.27 source and in v0.28, and is
needed for ordinary Qwen3.8 weights. Removing it therefore makes the current
model structure unable to load the source projection names.

## Decision

This confirms that the GDN mapper must remain enabled; it does not establish a
v0.27-to-v0.28 source delta. The temporary source patch, image, and phase
cache are removed. Rule precedence remains excluded as a cause. Standard
Qwen3.8 AWQ remains the supported v0.28 development profile; later
packed-INT8 work must audit tensor assignment and forward numerics.
