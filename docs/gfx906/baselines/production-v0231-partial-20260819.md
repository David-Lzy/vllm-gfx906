# Production v0.23.1 partial baseline

Date: 2026-08-19

Status: **partial**. Text and 8/32/64-image workloads are valid. The cold
4096-grid throughput gate remains open because production traffic entered the
long-running grid window.

## Configuration

- Worker image: `aiinfos/vllm-gfx906-mobydick:v0.23.1rc0.x-rocm7.2.1-pytorch2.11.0`
- Router image: `vllm/vllm-router:nightly-20260710-b93cbcb`
- Model: `cyankiwi/Qwen3.5-9B-AWQ-4bit`
- Topology: four independent TP1 workers behind the vLLM Router
- Context: 100,000 tokens
- Max sequences: 8 per worker
- Max batched tokens: 32,768 per worker
- GPU memory utilization: 0.90
- Multimodal limit: 64 images, video disabled, 16,777,216 max pixels
- KV cache: FP16

## Clean core results

Each scenario used three measured repeats. Router counters prove all 21 repeats
below contained exactly the benchmark warmup and measured requests, with no
additional production requests.

| Scenario | Success | Req/s median | Prompt tok/s | Completion tok/s | p95 s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Text | 48/48 | 0.7376 | 16.23 | 377.66 | 11.51 |
| 8 images, unique | 48/48 | 1.6466 | 396.82 | 210.76 | 5.78 |
| 8 images, reuse | 48/48 | 2.0954 | 505.00 | 268.21 | 4.43 |
| 32 images, unique | 24/24 | 0.8329 | 742.12 | 93.88 | 5.91 |
| 32 images, reuse | 24/24 | 1.1924 | 1062.44 | 103.14 | 3.57 |
| 64 images, unique | 12/12 | 0.5623 | 985.76 | 71.98 | 6.40 |
| 64 images, reuse | 12/12 | 0.9646 | 1690.99 | 123.47 | 4.11 |

The Router assigned the 237 clean requests, including warmups, as 66, 50, 57,
and 64 requests across workers 0 through 3. This is a 1.32 max/min ratio over
the complete mixed workload, with no worker failure.

## Functional and stability gates

- Text and 1/8/32/64-image smoke passed.
- JSON-schema output passed 3/3 after increasing the test output cap from 128
  to 256 tokens. The initial 1/3 result was caused by output truncation, not an
  xgrammar or FSM error.
- The post-test API and all five containers remained healthy.
- No OOM, HTTP 500, traceback, xgrammar/FSM, or fatal RCCL error appeared in
  the phase log scan.
- All four GPUs reached 100% busy during the suite.
- Peak observed VRAM was 31.634, 31.567, 31.975, and 30.912 GiB on the four
  cards. The largest grid leaves almost no physical VRAM margin on one card.
- Peak worker CPU samples ranged from approximately 403% to 443%; Router CPU
  remained below 5%.

## Grid findings and invalid data

The first 4096-grid cold attempt completed 4/4 requests at 0.0040 req/s, but 14
additional production requests entered the same scenario window. It is not a
valid throughput baseline.

The original grid generator also reduced its seed to an eight-color rotation.
Later repeats therefore reused identical decoded pixels and measured processor
cache hits rather than unique inputs. Their 0.26-0.69 req/s results are retained
only as cache diagnostics.

The generator was corrected so the seed changes decoded pixels, and the JSON
output cap was corrected to 256. A focused cold-grid rerun was stopped when at
least 37 unrelated production requests entered the window. Production service
configuration was never changed or restarted.

## Resume gate

Reserve a traffic-free window of at least 45 minutes and run the corrected
4096-grid tests once at concurrency 1 and once at concurrency 4. Accept a run
only when the Router counter delta exactly equals benchmark requests plus
warmups. Phase 01 remains open until those two cold results and post-run idle
metrics pass.
