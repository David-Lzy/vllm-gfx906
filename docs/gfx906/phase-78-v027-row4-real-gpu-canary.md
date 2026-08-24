# Phase 78: v0.27 Row-4 Real-GPU Canary

## Goal

Validate the retained exact-M8 row-4 legacy-QGEMM composition on the two MI50
devices that serve the production Qwen3.5 9B AWQ workload. The experiment is
fully reversible: it uses a private endpoint, preserves the production
checkpoint and Compose inputs, and restores the retained v0.23.1 image before
returning service.

## Contract

Both the candidate and the direct old-worker control used Qwen3.5 9B AWQ with:

- 100K context, FP16 KV cache, 0.90 GPU-memory utilization, eight sequences,
  and 32,768 batched tokens per worker;
- a 64-image / zero-video prompt limit, 16 GiB shared-memory multimodal cache,
  prefix caching, and chunked prefill; and
- text, one/two 256-square image, and JSON constrained-output gates before
  the fixed-256-token throughput tests.

## First Router Result

The candidate passed health, model discovery, all routine text/image requests,
JSON `3/3`, C8, and C16. The subsequent `power_of_two` Router C16/256 soak
returned one HTTP 500 at batch 78. The Router reported a connection failure
while sending the typed request to one worker; neither vLLM worker crashed or
reported OOM, traceback, structured-output, or RCCL/NCCL failure.

The capture also found that this Router repeatedly queried `/get_load` on the
native v0.27 workers and received HTTP 404. Its `power_of_two` decisions then
recorded a zero load for both backends. This indicates a Router/worker
observability mismatch. It does not by itself prove the cause of the forwarded
connection reset, but it prevents a production promotion.

## Router-Bypassed Repetition

The candidate was repeated on the same two MI50 devices. After the routine
Router gates had passed, each worker received a direct C8 stream in parallel,
forming aggregate C16 while completely bypassing the Router forwarding path.

| Fixed 256-token direct test | Phase 66 v0.27 | Retained v0.23.1 | Change |
| --- | ---: | ---: | ---: |
| Aggregate C16 median throughput | 541.07 tok/s | 309.70 tok/s | +74.7% |
| Batch count | 90 | 20 | - |
| HTTP responses | 1,440 / 1,440 | 320 / 320 | all 200 |
| Candidate aggregate range | 537.88-545.90 tok/s | 190.36-519.55 tok/s | - |

The candidate completed all 90 paired batches without residual running or
waiting requests and without an error signature in the captured worker logs.
The old service was then restored and its health, model discovery, text/image,
JSON, and idle-metrics smoke all passed. The recovery harness now explicitly
removes a candidate image variable from production Compose evaluation and
asserts both worker image tags before treating the restore as successful.

## Decision

**Keep the row-4 worker composition, but do not change production yet.** The
real-GPU evidence now supports the worker/runtime performance and stability
branch. The service-level Router branch remains blocked until it completes a
long C16 soak without forwarded connection failures and with a policy that is
compatible with native v0.27 worker observability.

No model weights, cache contents, deployment paths, or private raw artifacts
are stored in this document.
