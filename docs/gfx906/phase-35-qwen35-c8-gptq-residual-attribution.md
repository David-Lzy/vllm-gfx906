# Phase 35: Qwen3.5 C8 GPTQ Residual Attribution

## Scope

Phase 33 recovered the unquantized batched decode tail on gfx906. This phase
profiles the remaining Qwen3.5 9B AWQ C8 gap on the v0.27 candidate. It is a
separate issue from the Qwen3.8 hybrid long-context attention regression
addressed in Phase 31.

All measurements use an isolated GPU2 server. Production GPU0/GPU1 workers,
the Router, port 8002, Compose files, and production model cache are not
changed.

## Result

The Phase 33 candidate passes text, one-image, two-image, and JSON 3/3
routine gates, but reaches only 157.57 tok/s at C8 versus the retained v0.23
reference of 223.37 tok/s. The remaining throughput is therefore 70.5% of the
release floor.

Two healthy profiler runs identify native GPTQ W4A16 decode as the next
candidate. In the fixed-64 C8 trace it accounts for 2.087 seconds of 6.086
seconds of model-forward GPU duration (34.3%). The retained B4 kernel accounts
for 0.489 seconds and GDN/Mamba plus paged attention are below the phase's 10%
implementation threshold.

The trace records these real compressed-tensors GPTQ shapes, rather than a
synthetic square-only workload:

| Decode rows | Output N | Input K | Typical role |
| ---: | ---: | ---: | --- |
| 27 / 190 | 24,576 | 4,096 | fused MLP gate/up |
| 27 / 190 | 4,096 | 12,288 | MLP down |
| 27 / 190 | 12,288 | 4,096 | MLP projection |
| 27 / 190 | 4,096 | 4,096 | attention/state projection |
| 27 / 190 | 10,240 | 4,096 | fused hybrid projection |

The existing generic GPTQ kernel splits K into 128-wide blocks and atomically
accumulates every partial output. That is a plausible cost on the 4,096 and
12,288 K shapes, but this phase does not infer a replacement from profile time
alone.

## Community Check

The related Qwen3.8 `0.238 tok/s` long-context result has a different cause:
hybrid Qwen uses 256-wide heads and a Mamba-aligned physical KV page, which
keeps the native ROCm paged-attention kernel ineligible. The upstream split-KV
work described in [issue #50264](https://github.com/vllm-project/vllm/issues/50264)
and [PR #45916](https://github.com/vllm-project/vllm/pull/45916) is the safe
model for the opt-in gfx906 recovery already validated in Phase 31. It must not
be replaced by widening the native ROCm selector, because the native kernel has
no matching head/page instantiation.

The upstream shared-expert scalar-gate report in
[issue #52631](https://github.com/vllm-project/vllm/issues/52631) is relevant
to real Qwen MoE checkpoints, but not to the dense Qwen3.8 checkpoint tested
here. The RDNA3 W4A16 code is also not directly portable: it relies on wave32
and gfx11-specific instructions. The local candidate must target gfx906's
wave64 and be checked for numerical equivalence against the existing GPTQ op.

## Decision

Proceed to a narrow shape-driven microbenchmark phase. It will compare the
current operator on actual packed Qwen weights at M=1/8/27/190, verify output
error, and only then decide whether a gfx906 HIP C++ kernel can safely reduce
K-split atomics. No production promotion follows from this attribution.

