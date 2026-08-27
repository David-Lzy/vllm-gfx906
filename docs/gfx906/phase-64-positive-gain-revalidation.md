# Phase 64: Positive-gain revalidation

## Scope

This phase applies the gfx906 positive-gain retention policy to narrow
optimizations that were previously screened with a default-promotion threshold.
It does not alter the production Router, workers, port 8002, or model cache.

## Phase 56 Qwen3.5 vision encoder compilation

Upstream vLLM commit `653ebb52dffd8b4653b430302473c771117529f1` enables
`torch.compile` for the Qwen3 vision encoder. The candidate was tested on one
development MI50 with the retained v0.27.1 legacy-QGEMM image, Qwen3.5 9B AWQ,
100K context, eight sequences, and cache-busted 256-square image inputs.

Both the control and candidate passed health, text, one-image, two-image, and
JSON-object 3/3 gates. No HTTP 5xx, OOM, traceback, structured-output, RCCL,
or queued-request failure was found.

After each worker's first measured round, the three warm medians were:

| Scenario | Control | Encoder compile | Candidate change |
| --- | ---: | ---: | ---: |
| One image | 980.64 ms | 981.52 ms | -0.09% |
| Two images | 1061.31 ms | 1061.80 ms | -0.05% |
| Text | 845.98 ms | 842.77 ms | +0.38% |

The earlier apparent image benefit was caused by the ordinary first visual
path becoming warm on later requests; the uncompiled control converged to the
same image latency. The small text direction is below measurement confidence
and is not evidence that the vision compilation improves image serving.

**Disposition:** `provisional-positive`, default off. The focused branch,
overlay Dockerfile, benchmark harness, and raw data remain available for a
future PyTorch, Triton, or vLLM revalidation. The option is not included in a
release or production profile because it has no repeatable visual throughput
gain and adds a material startup-compilation cost.

Raw artifacts are intentionally outside Git under the configured local build
root, in the `phase-56/results` directory.

## Retention implication

The Phase 36 K=256 GPTQ component remains retained inside the coherent
gfx906-only legacy-QGEMM bundle. Its end-to-end C8 gain was +1.3%, and the
complete bundle later became the Phase 44 v0.27 canary-eligible candidate.
Small verified gains remain retained; the result above only prevents a
warmup-induced measurement from being mislabeled as an image-speed gain.
