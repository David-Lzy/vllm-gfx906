# Phase 27: v0.27 Mega-AOT Artifact Parity

## Scope

This isolated GPU2 experiment disabled `VLLM_USE_MEGA_AOT_ARTIFACT` while
retaining the Phase 24 Qwen3.5 9B AWQ configuration: gfx906 ExLlama W4A16
selection, forced Triton decoder attention, 100K context, eight sequences,
FP16 KV cache, and the routine text plus one/two 256-square image gate.
Production was not modified.

The retained PyTorch 2.11 worker defaults this flag to false, while the v0.27
PyTorch 2.13 worker defaults it to true. The test therefore isolates the
standalone Inductor artifact path rather than changing an unrelated serving
parameter.

## Result

Health, model discovery, text C1/C8, one/two-image C1, JSON constrained output
`3/3`, the fatal-log scan, and idle request metrics all passed. The override
was stable, but the median changes against the Phase 24 forced-Triton control
are below one percent.

| Scenario | Phase 24 tok/s | Mega-AOT off tok/s | Change |
| --- | ---: | ---: | ---: |
| Text, C1 | 62.015 | 62.313 | +0.48% |
| Text, C8 | 125.401 | 125.977 | +0.46% |
| One 256-square image, C1 | 58.415 | 58.868 | +0.78% |
| Two 256-square images, C1 | 55.818 | 56.012 | +0.35% |

## Decision

Reject the override as a performance recovery. It neither improves two primary
C1 paths by the required three percent nor closes the retained v0.23
small-request throughput gap. Keep the v0.27 default enabled and leave the
production deployment unchanged.

## Evidence

Raw results are deliberately local and non-versioned under the configured
gfx906 build root, run `20260821T143000Z-v027-triton-attn-mega-aot-off`.
