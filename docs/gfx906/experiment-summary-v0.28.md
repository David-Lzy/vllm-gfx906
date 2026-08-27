# gfx906 experiment summary through v0.28

The retained production result is Qwen3.5 9B AWQ as four independent TP1
workers behind a round-robin Router. It reached a balanced `1,548` requests per
worker during the 6,192-request canary and about `691 tok/s` at the matched C16
mixed gate.

For Qwen3.8 27B, the largest recovery came from selecting the native gfx906
GPTQ path and restoring head-256 SplitKV. The original generic path measured
`0.238 tok/s` on the long-decode reproducer; the repaired v0.27 path reached
`35.67 tok/s` for fixed-128 decode and `1.353 tok/s` at 32K cache hit. The v0.28
release TP4 standard-AWQ profile reached `56.73/226.13/20.75 tok/s` for
C1/C8/32K cache hit. The packed-INT8 release profile reached
`57.61/245.11/20.84 tok/s`. SplitKV-29 remains the retained long-context
selector, while packed INT8 adds a smaller development-only gain.

Negative results are part of the release decision: AITER, DFlash, MTP, GDN
output norm, cost-aware routing, CPU affinity, higher admission caps, generic
Triton unified attention, and several custom-kernel attempts did not justify a
general default. Their focused evidence remains in this documentation and the
release archive bundle; they are not silently enabled in production.

The release image also passed bounded startup and image/JSON compatibility
smokes for TurboQuant KV, FP8 KV, MTP1, DFlash, Qwen3.6 fusion plus SplitKV-29,
and Qwen3.5 35B-A3B BF16 MoE. These results do not reverse the earlier
performance decisions: compatibility is not evidence of a general speedup.
