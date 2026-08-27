# Phase 18 cost-aware Router evaluation

## Scope

This phase evaluated an isolated, Rust-based request-cost-aware proxy against
the vLLM Router `power_of_two` policy. Each endpoint addressed two temporary
single-GPU Qwen3.5 9B AWQ workers. The production endpoint, its workers, and
its deployment files were not changed.

The proxy used only request metadata already present in the OpenAI-compatible
body: text byte count, data-URL byte count, image count, and `max_tokens`.
It forwarded the original request bytes and streaming response without
downloading or decoding media.

## Result

The isolated workers and both routers completed health, text, image, and JSON
smoke checks. The mixed benchmark completed 576 requests with non-empty HTTP
200 responses and no OOM, traceback, structured-output, or ROCm fatal errors.

| Mixed load | Cost-aware | Power of two | Delta |
| --- | ---: | ---: | ---: |
| C16 mean request/s | 4.2922 | 4.2942 | -0.05% |
| C16 p95 batch seconds | 10.1640 | 8.5564 | 18.8% worse |
| C32 mean request/s | 4.8523 | 4.7771 | +1.57% |
| C32 p95 batch seconds | 16.4246 | 17.0953 | 3.92% better |

The C32 result is directionally positive but substantially below the required
10% mixed-tail-latency improvement. The C16 regression makes promotion unsafe.
The current vLLM Router remains the production path; the sidecar does not
receive production traffic.

## Reproducibility notes

The harness and proxy are retained under `tools/gfx906-cost-router/`. The
temporary workers remain on an internal network. The two routers additionally
use a router-only edge network because Docker cannot publish a host port from
an internal-only network on this host. This does not give workers a route to
production. The control Router image validates `retry-max-retries >= 1`; the
harness uses one permitted attempt and does not retry failed chat requests.

## Decision

Status: **rejected for production promotion**. Revisit only with a stronger
predictor of actual engine cost, such as backend queue telemetry combined with
server-side request tokenization. A raw request-body estimate is not enough to
justify replacing the current Router.
