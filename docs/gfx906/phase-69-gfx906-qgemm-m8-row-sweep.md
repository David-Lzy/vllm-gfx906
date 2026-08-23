# Phase 69: gfx906 legacy QGEMM exact-M8 remaining row sweep

## Scope

Phase 66 retained a gfx906-only exact-M8 QGEMM row-4 profile for Qwen3.5 9B
AWQ C8 decode. Phase 67 showed that row-2 was materially slower. This phase
screened every remaining valid row geometry: `3`, `5`, `6`, and `7`.

The work ran on development GPU2 only. Production GPU0/GPU1, the production
Router, port 8002, compose files, and production model cache were not
modified. The production health endpoint returned `All servers healthy` after
the experiment.

The implementation branch is `perf/gfx906-qgemm-m8-row-sweep` at
`45d8b497b1`. It extends the guarded CMake selector from the previously
exposed values to `0..8`; `0` remains the generic default and no unguarded
model behavior changes.

## Method

Each candidate was an overlay on the retained Phase 44 legacy-QGEMM image.
Only exact `M=8` GPTQ QGEMM calls changed. All other calls continued to use
the row-8 geometry. The service configuration was kept constant:

- Qwen3.5 9B AWQ, float16 weights/KV cache, 100K context.
- One development MI50, `max_num_seqs=8`, 32K batched tokens.
- Prefix cache and chunked prefill enabled; image limit 64, video disabled.
- Two C8 warmup rounds followed by three concurrent C8 measured rounds.
- Eight text requests per round, each fixed at 64 completion tokens.

Each custom-op image necessarily generated a distinct torch.compile cache key.
The common Triton/vLLM cache was copied into each disposable candidate cache
before launch. Startup compilation and multimodal warmup were not included in
the request-throughput measurements.

## Results

| Exact-M8 rows per block | C8 samples, tok/s | Median tok/s | Delta vs row-4 | Disposition |
| ---: | --- | ---: | ---: | --- |
| 4 (control) | 236.71, 238.97, 239.47 | **238.97** | baseline | retained-targeted |
| 3 | 200.51, 200.87, 200.99 | 200.87 | -15.94% | rejected |
| 5 | 197.90, 199.38, 199.58 | 199.38 | -16.57% | rejected |
| 6 | 197.64, 197.74, 197.93 | 197.74 | -17.25% | rejected |
| 7 | 192.11, 192.45, 193.65 | 192.45 | -19.46% | rejected |

Raw per-round JSON and server logs remain outside Git under the configured
local build root in `phase-69/results`. The experimental images and caches
are disposable and are removed after this report is recorded.

## Disposition

**Retain exact-M8 row-4.** This is stronger than the earlier positive A/B:
the full row-geometry screen shows that the successful `4 + 4` decomposition
is a specific gfx906 property, not a generic benefit of smaller M8 blocks.
No new candidate reached the functional gate because none achieved a positive
C8 pre-screen result. Phase 66 routine text/image/JSON gates and Phase 68
request-time call attribution remain the functional evidence for the retained
row-4 profile.

This does not change the production deployment. The generic CMake default is
still disabled; the validated Qwen3.5 9B AWQ gfx906 C8 profile explicitly
selects `VLLM_GFX906_LEGACY_QGEMM_C8_ROWS_PER_BLOCK=4`.
