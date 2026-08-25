# Phase 125: Qwen3.6 Packed-INT8 TP4 Portability

## Purpose

Validate whether the retained Qwen3.8 packed-INT8 embedding and lm-head
profile also works for Qwen3.6 27B AWQ on four AMD MI50 GPUs. The profile is
copy-on-write: only the two vocabulary-sized tables are re-encoded; all other
weights remain on the standard AWQ path.

This is a Qwen 27B development result. It does not replace the Qwen3.5 9B
Router-backed production deployment.

## Environment

- vLLM image: `local/vllm-gfx906:v0.28.0-phase123-qwen38-int8-semantic`
- Required source repair: `e7593a1689` (gfx906 INT8 GPTQ FP32 accumulation)
- Hardware: four AMD MI50 GPUs, TP4
- Runtime: `gfx906_gptq`, FP16 KV cache, SplitKV enabled, no MTP, 100K
  context, eight sequences, 8,192 batched tokens, prefix caching and chunked
  prefill
- Media contract: up to 64 images, video disabled, 16 MiB maximum pixels

The conversion used symmetric packed INT8 with group size 128 for
`model.language_model.embed_tokens.weight` and `lm_head.weight`. Their
relative reconstruction errors were `0.005907019` and `0.006463991`, below
the `0.01` conversion limit.

## Validation

The copy-on-write checkpoint first passed a TP2 loader gate. The all-GPU TP4
service then started in `603.390s`, provided `1,265,671` KV-cache tokens, and
passed text, one 256-square image, two 256-square images, and JSON `3/3`.
Responses were coherent and non-empty. Error scans were empty for HTTP 5xx,
OOM, traceback, xgrammar/FSM, RCCL/NCCL fatal, and leaked running or waiting
requests.

Routine validation intentionally excludes large 4096-square grids and 32/64
image workloads; those are capacity-specific gates rather than a normal
development gate.

## Throughput

The baseline is the Phase 118 Qwen3.6 TP4 standard-AWQ result. Two independent
fixed-128 runs were used for the packed profile.

| Metric | Standard AWQ TP4 | Packed run 1 | Packed run 2 | Packed mean | Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 fixed-128 tok/s | 52.65 | 52.8666 | 52.6411 | 52.7538 | +0.20% |
| C8 fixed-128 aggregate tok/s | 216.53 | 217.0853 | 234.3598 | 225.7225 | +4.24% |

C8 was directionally positive but variable. The result is therefore a
retained, small development-profile improvement rather than a broad
production-performance claim. The per-run evidence remains the source of
truth for future paired revalidation.

## Decision

Keep the generic packed embedding/lm-head conversion path and the Qwen3.6
copy-on-write profile for subsequent v0.28 Qwen 27B work. Do not modify the
Qwen3.5 9B production model, Router topology, or production Compose files.
After the maintenance window, production returned to healthy state and passed
model discovery, text, image, JSON, and idle-metric checks.
