# Phase 23: v0.27 Small Multimodal Release Parity

## Scope

This isolated comparison evaluates the retained gfx906 worker against the
v0.27.1 gfx906 image with the Phase 21 LLMM1 decoder recovery. It uses the
same Qwen3.5 9B AWQ checkpoint, GPU, 100K context, serving parameters, and
small routine multimodal gate. It does not modify production.

## Configuration

Both candidates ran one at a time on development MI50 GPU2 with the production
configuration: FP16 KV cache, `gpu_memory_utilization=0.90`, eight sequences,
32,768 batched tokens, prefix caching, shared-memory multimodal processor
cache, image limit 64, video limit zero, and the current 16,777,216 pixel
limit. The v0.27 candidate explicitly enabled the gfx906 ExLlama W4A16 policy;
the generic Triton W4A16 fallback was excluded before measurement.

Each candidate passed health, model discovery, text, one/two 256-square image
requests, JSON constrained output `3/3`, and idle request checks. There were
no HTTP 5xx, OOM, xgrammar/FSM, or RCCL/NCCL fatal errors.

## Result

| Scenario | Retained v0.23 tok/s | v0.27 tok/s | v0.27 / retained | Result |
| --- | ---: | ---: | ---: | --- |
| Text, C1 | 75.25 | 60.32 | 80.2% | Below floor |
| One 256-square image, C1 | 66.76 | 55.73 | 83.5% | Below floor |
| Two 256-square images, C1 | 63.29 | 51.46 | 81.3% | Below floor |
| Text, C8 one-shot | 76.24 | 127.39 | 167.1% | Diagnostic only |

The C8 row includes first-use concurrent-shape JIT in both images, so it is
not a warmed-throughput release comparison. The consistent C1 regression is
sufficient to fail the release rule: every routine scenario must retain at
least 95% of the current worker's throughput.

## Capacity Note

At the unchanged `gpu_memory_utilization=0.90`, v0.27 allocates 463,265 KV
tokens, equivalent to 4.63 concurrent 100K contexts. CUDA graph profiling
reserves about 0.75 GiB; the runtime reports that 0.9208 would match the older
effective capacity. This test intentionally did not change the production
setting, because the purpose was like-for-like release parity rather than
capacity tuning.

## Decision

v0.27 is functionally compatible with the current small Qwen3.5 multimodal
workload, but it is not a production replacement. Phase 10 remains closed.
Future performance work needs a new, specifically evidenced gfx906 decoder or
compiler lead; generic parameter sweeps must not reinterpret this result as a
release win.
