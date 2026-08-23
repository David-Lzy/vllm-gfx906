# Positive-gain retention policy

## Purpose

gfx906 improvements are expensive to discover and reproduce. A correct,
workload-specific improvement is retained even when it is too small or too
narrow to become the default release configuration.

This policy separates retaining an optimization from enabling it by default.
The latter still requires a broader performance and production-readiness
decision.

## Classifications

| Classification | Meaning | Default behavior |
| --- | --- | --- |
| retained-default | Reproducibly improves the primary workload without regressions | Eligible for the release image after ordinary gates |
| retained-targeted | Improves a defined model, shape, or context regime | Guarded and disabled outside that regime |
| provisional-positive | A single clean measurement is positive but its confidence interval is not yet known | Source, patch, and harness stay available; do not enable automatically |
| rejected | Incorrect, unstable, or non-positive in its stated target | Keep the report, but do not carry the implementation forward |

A target-specific result may remain retained even if it regresses a different
target. Its guard must make that distinction explicit rather than silently
changing the common serving path.

## Evidence required

Every retained entry records the source revision, guard, numerical or output
checks, benchmark command, and raw results. Small gains are measured with
paired warm runs under the same image, model, topology, cache state, and
request mix. A gain whose confidence interval crosses zero is
`provisional-positive`, not discarded.

No result is promoted to a production default solely because it is positive.
It must also preserve routine text, image, and JSON behavior and avoid a
meaningful regression in the primary serving workload.

## Current retained examples

| Optimization | Scope | Status |
| --- | --- | --- |
| gfx906 legacy QGEMM bundle, including K=256 launch geometry | Qwen AWQ W4A16 linear layers on gfx906 | retained-targeted; built only with the gfx906 CMake guard |
| gfx906 SplitKV composition | Qwen3.8 long-context decode | retained-targeted; long-context profile only |
| GDN output-projection reshape elision | Qwen3.8 32K cache-hit decode | provisional-positive; retained as a default-off overlay because short decode regressed |

The legacy QGEMM bundle is intentionally treated as one coherent optimization:
K=256 geometry, QDQ behavior, accumulation, and output initialization are not
independent knobs. The retained implementation is documented by the fork's
legacy-QGEMM phase and uses a gfx906-only build guard.

## Revalidation and removal

Retained paths are re-run whenever their enclosing kernel, PyTorch, Triton,
or model interface changes. They are removed only after a replacement is
proven correct and at least as fast for the same target, or after a reproducible
correctness or stability failure. Raw benchmark evidence remains in the phase
record after removal.
