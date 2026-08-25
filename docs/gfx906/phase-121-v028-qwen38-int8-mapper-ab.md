# v0.28 Qwen3.8 Packed-INT8 Mapper Screen

## Scope

Phase 121 tested the remaining simple source-level explanation for the failed
optional Qwen3.8 packed-INT8 embedding/output-head checkpoint. The known-good
v0.27 experiment predated v0.28's Qwen3.5 weight mapper, which maps the
checkpoint's `model.language_model.*` namespace into the current model
namespace. This screen restored the Phase 119 packed-embedding construction
and removed that mapper only for an isolated v0.28 TP2 candidate.

The rest of the contract was unchanged: two MI50 GPUs, 100K context, FP16 KV,
eight sequences, 8,192 batched tokens, no MTP, and the retained gfx906
SplitKV selector. No production file or model shard was changed.

## Result

The candidate failed during worker initialization, before health or any
request could run. The loader reported that
`layers.0.linear_attn.in_proj_a` has no destination in `Qwen3_5Model`.

This is an architectural compatibility failure, not a performance result.
The v0.28 Qwen GDN implementation fuses the checkpoint's separate
`in_proj_a` and `in_proj_b` values into `in_proj_ba`; its mapper participates
in the compatibility path needed to load those source names. Removing the
mapper therefore makes ordinary Qwen3.8 weights unloadable under the current
model structure.

## Decision

The mapper-removal hypothesis is rejected. The temporary source patch, image,
and phase cache are removed. Together with Phase 120, this excludes both rule
precedence and mapper removal as explanations for the packed-INT8 semantic
failure. Standard Qwen3.8 AWQ remains the supported v0.28 development
profile; any later packed-INT8 work must audit tensor assignment and forward
numerics while retaining the required GDN mapper.
