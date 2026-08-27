# v0.28 Qwen3.8 Packed-INT8 Embedding Port

## Scope

Phase 119 evaluated the copy-on-write Qwen3.8 27B checkpoint from the earlier
gfx906 line. The checkpoint leaves the AWQ transformer weights unchanged while
storing its embedding and output-head tables as packed symmetric INT8. The
previous profile reduced per-rank model memory and showed a positive same-model
throughput direction, so retaining compatibility would be useful but optional.

The v0.28 model constructor initially omitted the compressed-tensors quantizer
and the embedding prefix when it created `embed_tokens`. A narrow two-line
candidate restored that wiring. The test used two development MI50 GPUs, TP2,
100K context, FP16 KV cache, eight sequences, 8,192 batched tokens, no MTP,
and the retained gfx906 SplitKV selector. Production workers and their Router
were not modified.

## Result

The loader issue was resolved: the candidate became healthy in 406.52 seconds,
selected `compressed-tensors`, loaded the packed checkpoint, and reduced the
per-rank model footprint to 8.87 GiB. There were no OOM, RCCL fatal, xgrammar,
or queue-drain failures.

The quality gate failed immediately. Text repeated a short phrase, image
responses repeated blank quote tokens, and all three constrained JSON attempts
returned malformed repeated tokens. Every response was HTTP 200, so transport
health alone would have hidden the error.

| Gate | Result |
| --- | --- |
| Health and model list | Pass |
| Packed embedding construction | Pass |
| Text quality | Fail |
| One/two 256-square image quality | Fail |
| JSON 3/3 | Fail |
| Throughput benchmark | Not run by stop rule |

The same checkpoint has preserved v0.27 Phase 82 records with coherent text,
image, and JSON outputs. This rules out a damaged checkpoint as the primary
explanation. The compressed embedding implementation is byte-identical across
the two images; the remaining compatibility boundary is the v0.28 handling of
a compressed-tensors configuration that combines a broad AWQ `Linear` rule
with more-specific packed-INT8 embedding and output-head rules.

## Decision

The two-line source change and disposable image are not retained. The standard
v0.28 Qwen3.8 AWQ checkpoint remains the supported development profile.

The configuration-only Phase 120 follow-up then placed the two packed-INT8
regex rules ahead of the broad AWQ `Linear` rule. It reached health in 633.224
seconds and retained the same 8.87 GiB per-rank model footprint, but text,
image, and JSON outputs were identically malformed. Rule precedence is not the
explanation, and the optional packed-INT8 profile remains rejected pending a
separate, source-level compatibility diagnosis.
