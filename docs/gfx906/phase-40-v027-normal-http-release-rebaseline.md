# Phase 40: v0.27 Normal HTTP Release Rebaseline

## Purpose

Phase 39 separated profiler-attribution timings from normal serving timings.
This phase repeats the real release comparison with the retained Qwen3.5 9B
AWQ worker: v0.23 automatic selection versus the v0.27 B4/gfx906 GPTQ
candidate. It also checks the only remaining low-risk selector alternative,
v0.27 automatic ExLlama.

This is a separate problem from the historical `0.238 tok/s` long-context
control. That control was Qwen3.6 27B AWQ at 32K context and was used to guide
the Qwen3.8 investigation. Phase 31 has already retained the appropriate
Qwen3.8 remedy: explicit gfx906 GPTQ W4A16 plus opt-in split-KV decode.

## Method

All candidates used one isolated MI50 (GPU2), the same
`cyankiwi/Qwen3.5-9B-AWQ-4bit` checkpoint, FP16 KV cache, 100K context,
eight sequences, 32,768 batched tokens, one renderer, prefix caching, and the
same 256px PNG fixture. The serving path was ordinary OpenAI-compatible HTTP;
no torch profiler was enabled. Two C8 warmups preceded the timed C8 request.

The routine gate covered text C1, one/two-image C1, text C8, JSON constrained
output `3/3`, post-run idle metrics, and fatal-log scans. All gates passed for
every candidate. Production GPU0/GPU1, the Router, port 8002, production
compose, and production model cache were not changed.

## Results

| Scenario | v0.23 automatic | v0.27 B4 + gfx906 GPTQ | v0.27 automatic ExLlama |
| --- | ---: | ---: | ---: |
| Text C1 | 75.17 tok/s | 60.41 tok/s (80.4%) | 60.18 tok/s (80.1%) |
| Text C8 | 224.26 tok/s | 158.77 tok/s (70.8%) | 156.69 tok/s (69.9%) |
| One image C1 | 66.78 tok/s | 55.38 tok/s (82.9%) | 55.61 tok/s (83.3%) |
| Two images C1 | 62.81 tok/s | 51.80 tok/s (82.5%) | 51.57 tok/s (82.1%) |

The B4 candidate reproduces the Phase 33 C8 result under the corrected
normal-HTTP measurement lane. It is still `29.2%` below the retained v0.23
C8 reference. Automatic ExLlama is within 1.3% of B4 on C8 and does not supply
a recovery. All response bodies were non-empty, JSON passed `3/3`, the final
metrics were idle, and the fatal signature scans were empty.

## Community Interpretation

- [vLLM issue #49699](https://github.com/vllm-project/vllm/issues/49699)
  reports a compile-mode-3 W4A16 interaction on MI100 and suggests forcing
  ExLlama. This phase tests that hypothesis on MI50 with normal HTTP traffic;
  it is neutral to slightly slower, so it is not a portable gfx906 default.
- [vLLM issue #50264](https://github.com/vllm-project/vllm/issues/50264)
  identifies the different long-context hybrid-Qwen failure: 256-wide-head
  paged attention falls back to Triton and needs split-KV work rather than a
  selector change. Phase 31 already implements and validates that narrowly on
  gfx906.
- Recent AMD ROCm W4A16 and HybridW4A16 implementations explicitly target
  RDNA3/RDNA4 (`gfx11x`/`gfx12x`). Their wave32 and LDS assumptions do not
  apply to MI50 `gfx906`; enabling them without a native port would be
  incorrect.

## Decision

Do not promote v0.27 to production. Do not change the production backend flag
or globally enable split-KV. Phase 41 owns a repeated normal-C8 diagnostic
lane and must attribute the remaining v0.27 deficit before another HIP C++ or
compiled-runtime change is proposed.

## Evidence

Raw benchmark artifacts are intentionally outside Git:

- B4/gfx906 GPTQ comparison:
  `/mnt/disk2/vllm-gfx906-build/phase-40/results/20260822T040713Z-small-mm-parity/`.
- Automatic ExLlama comparison:
  `/mnt/disk2/vllm-gfx906-build/phase-40/results/20260822T043639Z-small-mm-parity/`.

