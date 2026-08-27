# v0.28 Qwen3.8 Packed-INT8 Rule-Precedence Screen

## Scope

Phase 120 tested a single, data-only explanation for the failed optional
Qwen3.8 packed-INT8 embedding and output-head profile. The checkpoint's
compressed-tensors configuration contains a broad AWQ `Linear` rule plus two
more-specific packed-INT8 regex rules. A temporary metadata overlay moved the
two packed-INT8 groups ahead of the broad rule. No model shard, production
configuration, or vLLM source line was changed.

The candidate reused the Phase 119 image on two development MI50 GPUs in TP2:
100K context, FP16 KV cache, eight sequences, 8,192 batched tokens, no MTP,
and the retained gfx906 SplitKV selector.

## Result

The service became healthy in `633.224 s`, selected `compressed-tensors`, and
loaded its model weights at `8.87 GiB` per rank. It ended with no running or
waiting requests and no OOM, RCCL/NCCL fatal, xgrammar/FSM, or traceback
signature.

The semantic gate failed exactly as it had in Phase 119:

| Gate | Result |
| --- | --- |
| Text | Fail: repeated `一种是` |
| One 256-square image | Fail: repeated quote tokens |
| Two 256-square images | Fail: repeated quote tokens |
| JSON 3/3 | Fail: repeated `Safety` with incomplete JSON |
| Throughput benchmark | Not run by stop rule |

## Decision

Configuration rule order is rejected as the cause of the v0.28 incompatibility.
The standard Qwen3.8 AWQ checkpoint remains the supported development profile.
Any further work on the optional packed-INT8 checkpoint must identify and test
a concrete compressed-tensors semantic change at the source level; it must not
be presented as a throughput candidate until routine quality has passed.
