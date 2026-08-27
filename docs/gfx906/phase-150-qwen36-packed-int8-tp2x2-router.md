# Phase 150: Qwen3.6 Packed-INT8 TP2x2 Router

## Status

Completed on 2026-08-27. The packed-INT8 copy-on-write overlay passed the
compact Qwen3.6 multimodal and structured-output gates, and improved the
measured TP2x2 Router workload enough to retain it as a development profile.
It is not a production promotion: the current production workload uses
Qwen3.5 9B TP1x4 Router, which remains the aggregate-throughput selection.

## Purpose

Phase 125 showed a small positive Qwen3.6 packed-INT8 result on TP4. This
phase checks whether the same overlay helps the high-concurrency TP2x2 Router
topology where Qwen 27B models are otherwise more attractive.

## Method

Two independent TP2 vLLM workers were attached to a round-robin Router. Both
arms used the same v0.28 gfx906 image, `gfx906_gptq` linear backend, FP16 KV
cache, 100K context, `max-num-seqs=8`, 8192 batched-token budget, image limit
64, 16 Mi pixel limit, prefix caching, and chunked prefill. The only model
difference was the standard Qwen3.6 AWQ checkpoint versus the copy-on-write
packed-INT8 overlay for the embedding and LM-head weights.

The compact gate included text, one 256-square image, two 256-square images,
and JSON constrained output three times. The performance cases generated 128
tokens with C1, C8 text, and a mixed C16 workload of text, one-image, and
two-image requests. Each performance case used three rounds.

## Results

| Metric | Standard AWQ | Packed-INT8 | Delta |
| --- | ---: | ---: | ---: |
| C1 completion throughput | 44.18 tok/s | 45.31 tok/s | +2.6% |
| C8 text completion throughput | 235.52 tok/s | 248.41 tok/s | +5.5% |
| Mixed C16 completion throughput | 278.49 tok/s | 293.10 tok/s | +5.2% |
| Router worker requests | 40 / 41 | 40 / 41 | balanced |
| Model memory during loading | 10.00 GiB | 8.84 GiB | -1.16 GiB |
| TP2 KV cache | about 463K-471K tokens | about 478K tokens | increased |

Both arms returned non-empty text and image outputs, passed JSON 3/3, drained
their request queues, and had no OOM, traceback, xgrammar/FSM, RCCL/NCCL
fatal, RAS, or illegal-instruction signatures in the saved server logs.

## Decision

Retain the packed-INT8 overlay as an opt-in Qwen3.6 TP2x2 development profile.
The gain exceeds the project's small positive-gain retention threshold and is
consistent across C1, C8, and mixed C16. Do not replace the production Qwen3.5
9B TP1x4 service with this 27B profile: this phase compares only Qwen3.6
variants, not the selected production model family.

## Evidence And Follow-up

Raw logs, Router metrics, worker metrics, and summary JSON are stored outside
Git under the phase build-result directory. This run exposed an artifact
mount omission in the benchmark client: the aggregate summaries were persisted
but the detailed request JSON was not. The runner now mounts the result
directory explicitly, so follow-up runs preserve per-sample latency and smoke
payload details. The omission does not invalidate the reported aggregate
throughputs, worker distribution, or server-side health evidence.
