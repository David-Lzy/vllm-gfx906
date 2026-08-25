# Phase 131: Qwen3.5 multimodal truncation and profiling

## Result

Retained as a correctness patch. It makes full multimodal memory profiling work
for Qwen3.5/Qwen3.6 Qwen3-VL processors at the configured maximum image
feature size. It is not a default performance or production-promotion change.

## Change

Upstream commit `54318114d3` disables Hugging Face tokenizer truncation only
for processor calls that contain multimodal data. This prevents the expanded
vision placeholders in a profiling dummy request from being truncated before
the Qwen processor validates their feature-token count.

The v0.28 branch has a later `_call_hf_processor` signature than the upstream
commit. Local commit `32d469cfe7` carries the narrow adaptation: merge
tokenizer kwargs into the processor kwargs and let `truncation=False` override
only on multimodal calls. Pure-text calls preserve the existing tokenizer
behavior.

## Test configuration

- Qwen3.6 27B AWQ, TP2, FP16 activations, gfx906 GPTQ backend.
- `max_model_len=100000`, image limit 64, video limit 0, and maximum image
  pixels 16,777,216.
- Candidate differed from the control only by removing
  `--skip-mm-profiling` and applying the processor fix.

## Evidence

The candidate completed full dummy-image profiling and reached health after
the initial warmup. Text, one 256-square image, two 256-square images, and
three JSON-constrained responses passed. After the requests, both running and
waiting request metrics were zero. No OOM, HTTP 5xx, xgrammar/FSM, or RCCL/NCCL
fatal error occurred.

| Metric | Skip-profiling control | Full-profiling candidate |
| --- | ---: | ---: |
| Initial profiling/warmup | skipped | 405.74 s |
| Engine initialization | fast path | 840.80 s, including 129.32 s compilation |
| Available KV memory | 14.86 GiB | 13.70 GiB |
| GPU KV tokens | 462,686 | 426,865 |
| 100K-context theoretical concurrency | 4.63x | 4.27x |

Full profiling therefore reserves approximately 7.7 percent more capacity than
the skip-profiling control. That is the expected safety cost of sizing the
large-image request rather than assuming a text-only activation profile.

## Decision

Keep the source patch for Qwen3-VL-derived models. Use full profiling for an
image-capacity-safe deployment or while validating a new maximum image limit.
Keep `--skip-mm-profiling` as an explicit operational choice where the existing
fast-start behavior and KV budget have already been validated. This phase does
not change the deployed service.
