# Phase 147: Qwen serving topology shootout

## Decision

For aggregate completion throughput, `Qwen3.5 9B AWQ` as four independent
TP1 engines behind vLLM Router is the clear winner on four MI50 GPUs. It is
the only tested arm to exceed 700 tok/s at mixed C16 and 850 tok/s at mixed
C32.

For one interactive text request, the same Qwen3.5 checkpoint in one TP4
engine is faster. For the Qwen3.8 27B capability tier, TP2x2 Router is the
batch/concurrent winner while direct TP4 remains the long-context winner.

This is a topology result, not a model-quality equivalence claim. The 9B and
27B checkpoints remain different capability tiers. No production Compose file
was changed or promoted by this phase.

## Contract

All candidates ran serially on the same four MI50 GPUs with a fixed
128-token completion budget, FP16 KV cache, 100K maximum model length, eight
sequences per engine, image limit 64, video disabled, 16,777,216 maximum image
pixels, one renderer worker, and 12 CPU-library threads. Qwen3.8 used 8,192
batched tokens; Qwen3.5 used 32,768.

The workload used C1 text, C8 text, C16 mixed requests, C32 text, and C32
mixed requests. Mixed traffic was deterministic text plus one/two 256-square
image data URLs. Each measured image had a pixel-level cache buster, so a
multimodal processor-cache hit could not select the winner. Qwen3.8 also ran a
32K prefix-cache-hit decode control. Router arms used isolated round-robin
Router instances rather than production routing state.

Every measured arm passed text, one-image, two-image, and JSON-constrained
`3/3` smoke gates. The final queues drained, with no OOM, HTTP 5xx,
xgrammar/FSM, RCCL/NCCL fatal, or RAS error in the completed service tests.

## Results

Aggregate completion throughput in tok/s. C1 is one request; C8, C16, and C32
are end-to-end aggregate throughput for the stated client concurrency.

| Candidate | C1 text | C8 text | C16 mixed | C32 text | C32 mixed | 32K decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8 27B AWQ, v0.28, TP4, SplitKV-29 | 53.39 | 219.12 | 195.16 | 211.99 | 192.41 | **18.66** |
| Qwen3.8 27B AWQ, v0.28, TP2x2 Router, SplitKV-29 | 44.01 | 224.15 | **268.35** | **301.75** | **267.20** | 12.90 |
| Qwen3.5 9B AWQ, v0.28, TP1x4 Router | 81.48 | **518.73** | **721.42** | **1028.17** | **868.76** | n/a |
| Qwen3.5 9B AWQ, v0.28, TP4 | **113.11** | 431.95 | 406.21 | 460.04 | 417.81 | n/a |
| Qwen3.5 9B AWQ, v0.23.1, TP4 | 111.18 | 367.31 | 356.70 | 387.94 | 355.78 | n/a |

The Qwen3.5 TP1x4 result represents four schedulers with a total effective
in-flight capacity of 32 requests (eight per engine). TP4 has one scheduler
with a capacity of eight. This is intentional: the table answers the
production-capacity topology question, not a per-engine kernel microbenchmark.

## Comparisons

### Qwen3.5 v0.28 TP4 versus v0.23.1 TP4

The new runtime wins every saturated TP4 scenario while retaining the same
model, contract, and TP degree: C8 `+17.6%`, C16 mixed `+13.9%`, C32 text
`+18.6%`, and C32 mixed `+17.4%`. C1 is effectively unchanged at `+1.7%`.

### Qwen3.5 TP1x4 Router versus v0.28 TP4

TP4 wins C1 by `38.8%` because the request gets four-way tensor parallelism
without a router hop. The independent-replica topology wins aggregate work:
C8 `+20.1%`, C16 mixed `+77.6%`, C32 text `+123.5%`, and C32 mixed `+108.0%`.

### Qwen3.8 27B topology

TP2x2 Router improves C16 mixed by `37.5%`, C32 text by `42.3%`, and C32
mixed by `38.9%` relative to TP4. TP4 remains `21.4%` faster for the 32K
cache-hit decode control and `21.3%` faster at C1, so it remains appropriate
for interactive or long-context Qwen3.8 requests.

## Legacy-runtime note

The retained v0.23.1 image does not implement the gfx906-specific
`--linear-backend` selector or SplitKV variables; the legacy command omitted
that new selector and the image logged the SplitKV variables as unknown. It
used its automatic `ExllamaLinearKernel` path and completed all functional and
throughput gates. Its long shared-memory/AOT warmup is a maintenance concern,
not part of the steady-state score.

The v0.23.1 `/metrics` endpoint returned HTTP 401 only during post-benchmark
artifact collection. The benchmark itself, including all smoke cases and five
throughput scenarios, had already completed successfully. It is recorded as a
legacy observability incompatibility rather than a model-serving failure.

## Operational recommendation

- Use Qwen3.5 TP1x4 Router when aggregate node throughput is the priority and
  four GPUs may be allocated to the service.
- Use Qwen3.5 TP4 for the best C1 response speed when model quality is
  sufficient and only one service engine is desired.
- Use Qwen3.8 TP2x2 Router for concurrent 27B work and TP4 SplitKV-29 for
  interactive or long-context 27B work.
- A separate canary/promotion phase must validate the selected 9B TP1x4
  production composition, startup/recovery behavior, host CPU budget, and
  live Router load distribution before it replaces the current service.

The reproducible runner and client are maintained on the corresponding Phase
147 benchmark branch. Raw model weights, caches, image digests, logs, and
machine-specific paths remain outside Git by policy.
