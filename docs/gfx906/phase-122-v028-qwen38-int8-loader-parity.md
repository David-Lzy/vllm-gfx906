# v0.28 Qwen3.8 Packed-INT8 Loader Parity

## Scope

Phase 122 audited the immutable Phase 82 packed-INT8 embedding/output-head
checkpoint after v0.28 loaded it with the expected memory footprint but
produced malformed text, image, and JSON output. The goal was to determine
whether v0.28 assigned different tensors from the known-good v0.27 profile.
No production service, standard-AWQ model, or checkpoint shard changed.

## Method

The same instrumented `AutoWeightsLoader` probe ran against the Phase 82
v0.27 image and a temporary v0.28 compatibility image. It recorded the
source/destination metadata and sampled values before and after assignment for
the six affected tensors on both TP2 ranks:

- `embed_tokens.weight_{packed,scale,shape}`
- `lm_head.weight_{packed,scale,shape}`

The probe stopped after assignment evidence was captured, before the
long-running compile and shared-memory-broadcast warm-up. It was a loader
audit, not a service performance trial.

## Result

Both images emitted 24 events: six parameters, two ranks, and before/after
records. Normalizing event order made the v0.27 and v0.28 assigned-state JSON
byte-for-byte identical.

The packed vocabulary tensors were partitioned consistently in both versions:

| Parameter | Global checkpoint | Per TP2 rank |
| --- | --- | --- |
| packed weights | `248320 x 1280` `int32` | `124160 x 1280` `int32` |
| group scales | `248320 x 40` `float16` | `124160 x 40` `float16` |
| logical shape metadata | `[248320, 5120]` | `[248320, 5120]` |

Rank-local samples also matched exactly. This eliminates checkpoint-key
mapping, namespace translation, dtype, vocabulary-shard boundary, and raw
parameter assignment as explanations for the v0.28 semantic regression.

## Decision

The packed-INT8 v0.28 profile remains unsafe because the earlier semantic
failure still exists. Further mapper/rule experiments are stopped. The next
step is an isolated numerical comparison of packed embedding lookup and the
packed WNA16 LM-head projection; a code change is justified only if that
comparison exposes an actual compute-path mismatch.
