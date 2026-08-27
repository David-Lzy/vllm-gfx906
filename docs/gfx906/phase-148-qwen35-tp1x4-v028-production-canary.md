# Phase 148: Qwen3.5 TP1x4 v0.28 production canary

## Decision

Promoted on 2026-08-27. The selected Qwen3.5 production service now runs four
independent TP1 workers behind vLLM Router, using the retained gfx906 v0.28
image and the explicit `gfx906_gptq` backend. The public API, model ID,
100K-context contract, image/video limits, and FP16 KV cache remain unchanged.

This promotes aggregate service capacity, not single-request latency. Qwen3.5
TP4 remains the faster topology for an isolated interactive text request, while
TP1x4 is the validated throughput configuration for concurrent text/image work.

## Release composition

- Checkpoint: `cyankiwi/Qwen3.5-9B-AWQ-4bit`.
- Runtime: `local/vllm-gfx906:v0.28.0-phase142-qwen-gdn-output-norm`.
- Topology: four independent TP1 engines and one round-robin vLLM Router.
- Per worker: 100K maximum model length, FP16 KV cache, 0.90 GPU-memory target,
  eight sequences, 32,768 batched tokens, image limit 64, video disabled,
  16,777,216 maximum image pixels, 16 GiB multimodal cache, one renderer
  worker, and twelve CPU-library threads.
- API: the existing OpenAI-compatible endpoint and model identifier are
  preserved.

The previous two-worker v0.27 Compose/environment snapshot remains the exact
rollback source. The selected service uses `unless-stopped` restart policies
for the Router and all four workers.

## Canary evidence

The maintenance window first ran the isolated release composition for at least
twenty minutes before editing the selected deployment. Every round included
text, one-image, two-image, JSON-constrained `3/3`, C1, C8, C16 mixed, C32
text, and C32 mixed requests. Images were cache-busted 256-square fixtures.

The canary completed sixteen rounds with `6,192` successful chat requests.
Each backend completed exactly `1,548` requests, so round-robin distribution
was balanced. No HTTP 5xx, OOM, traceback, xgrammar/FSM, RCCL/NCCL fatal, RAS,
or illegal-instruction signature was observed; every worker queue drained.

| Measurement | Phase 147 isolated reference | Phase 148 canary median | Change |
| --- | ---: | ---: | ---: |
| C16 mixed aggregate completion | 721.42 tok/s | 690.53 tok/s | -4.3% |
| C32 mixed aggregate completion | 868.76 tok/s | 828.57 tok/s | -4.6% |
| C32 text aggregate completion | 1,028.17 tok/s | 963.01 tok/s | -6.3% |

The release gate was 90% of the Phase 147 C16/C32 reference. Both primary
measurements cleared it with room to spare. The modest reduction is expected
between an isolated benchmark network and the persistent selected composition;
it does not change the topology decision.

For comparison, the earlier matched two-worker v0.27 Router canary recorded
527.99 tok/s at C16. The new production composition's 690.53 tok/s C16 median
is approximately 30.8% higher under the same fixed-128-token mixed workload.

## Post-promotion verification

The selected service was restarted from the promoted Compose, then passed
health, model discovery, text, one-image, two-image, JSON `3/3`, and drained
queue checks. A three-round post-promotion rerun measured:

| Scenario | Aggregate completion throughput |
| --- | ---: |
| C1 text | 77.68 tok/s |
| C8 text | 499.61 tok/s |
| C16 mixed | 691.58 tok/s |
| C32 text | 1,018.05 tok/s |
| C32 mixed | 823.42 tok/s |

All four selected workers were healthy with the `unless-stopped` policy, and
each handled 107--109 verification requests. The Router and every worker ended
with `running=0` and `waiting=0`.

## Operational notes

- Four first-start profiles are still a maintenance cost. With the retained
  AOT cache, compile loading took seconds, while worker profile/KV warmup took
  several minutes. This is recorded as startup behavior, not steady-state
  throughput.
- The release keeps the same 0.90 memory target. Initial worker logs allocated
  roughly 14.5 GiB to KV cache; no capacity pressure appeared during the canary.
- Raw per-request JSONL, worker metrics, logs, image assets, and local Compose
  snapshots remain outside Git by policy.
- Rollback is intentionally simple: restore the Phase 148 selected
  Compose/environment snapshot and restart that composition. Do not mix old
  worker caches with the v0.28 cache root.
