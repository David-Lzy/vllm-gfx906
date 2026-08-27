# v0.28 Qwen3.5 9B Router Canary

## Scope

This canary exercised the v0.28 gfx906 image in the real Qwen3.5 production
shape: two independent MI50 TP1 workers behind the existing Router. It held
the model, 100K context contract, image settings, cache settings, Router
policy, and fixed 128-token benchmark requests constant. The trial used a
bounded maintenance window and restored the prior v0.27 deployment afterwards.

## Functional result

The v0.28 workers and Router became healthy. Text, one-image, two-image, and
JSON-constrained requests all returned successfully; JSON completed 3/3.
There were no HTTP 5xx responses, OOMs, structured-output failures, or
ROCm/RCCL fatal log signatures. Request metrics drained to zero after testing.

## Throughput result

| Router measurement | v0.27 median | v0.28 median | Change |
| --- | ---: | ---: | ---: |
| C1, 128-token decode | 77.53 tok/s | 75.71 tok/s | -2.35% |
| C8 aggregate, 128-token decode | 420.26 tok/s | 414.89 tok/s | -1.28% |
| C16 aggregate, 128-token decode | 527.99 tok/s | 526.30 tok/s | -0.32% |

The v0.28 line clears the compatibility gate and stays above 95 percent of the
immediately preceding Router baseline at C8 and C16. It does not yet beat the
existing v0.27 production composition, especially for low-concurrency decode.

## Decision

Keep v0.28 as the active development line, including the retained gfx906
SplitKV development option for Qwen 27B. Do not promote it for Qwen3.5 9B
production until it closes the remaining short-decode delta and passes a new
canary. The production service was restored to its v0.27 image after this
experiment.
