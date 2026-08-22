# Phase 32: Qwen3.5 9B C8 Decoder Attribution

## Scope

Phase 30 established that the v0.27.1 gfx906 GPTQ candidate is functionally
correct, but its eight-request text throughput reaches only 56.1% of the
retained v0.23 worker. This phase profiles the same Qwen3.5 9B AWQ checkpoint
on one isolated MI50 before changing a device kernel or serving default.

The server used FP16, 100K context, `max-num-seqs=8`, 32,768 batched tokens,
the explicit `gfx906_gptq` linear backend, chunked prefill, prefix cache, and
the existing multimodal settings. The production Router, GPU0/GPU1 workers,
port 8002, Compose files, and production cache were not changed.

## Result

The trace separates the regression by execution class rather than attributing
it to image processing or generic ROCm activity:

| Trace class | C1 GPU duration | C8 GPU duration | Interpretation |
| --- | ---: | ---: | --- |
| Native GPTQ W4A16 | 863 ms | 2,027 ms | Large but expected quantized model work |
| rocBLAS/Tensile fallback | 11 ms | 1,580 ms | C8-specific unquantized-linear hotspot |
| Qwen GDN / recurrent decode | 21 ms | 97 ms | Secondary contributor |
| Attention | 49 ms | 54 ms | Not the C8 bottleneck |
| Triton fused work | 44 ms | 40 ms | Not the C8 bottleneck |

The C8 trace contains roughly two thousand generic rocBLAS/Tensile launches.
At batch one they are negligible; at batch eight they dominate the incremental
cost. The source-level reason is also direct: gfx906 dispatch only uses the
native `LLMM1` unquantized-linear kernel for `N == 1`; multi-token decode falls
through to `torch.nn.functional.linear`, which selects generic rocBLAS kernels.

The result is consistent with current ROCm community reports. Upstream
[issue #52631](https://github.com/vllm-project/vllm/issues/52631) documents a
different skinny-GEMM dispatch gap that sends a tiny MoE projection to rocBLAS
and consumes 13.7% of a decode step. Its shape is not the same as Qwen3.5 9B,
but it independently confirms that ROCm fallback selection can be a material
decode bottleneck. Upstream [PR #40687](https://github.com/vllm-project/vllm/pull/40687)
addresses a related `wvSplitK` batch-width gap on newer AMD hardware. That
kernel contains no gfx906 implementation, so it is evidence for the direction,
not a patch that can be enabled on MI50.

## Decision

Proceed with a narrow gfx906-native batched unquantized-linear experiment. Do
not enable `wvSplitK` on gfx906: its compiled branch is an explicit
unreachable/device-assert path and an isolated invocation confirms that it is
not a usable fallback on MI50. Do not change attention, CPU pre-processing, or
the production deployment from this trace.

The next phase must preserve the established single-token `LLMM1` route, add
only an independently measured batch-2 through batch-8 route, and retain a
shape guard for any matrix where rocBLAS remains faster. It must rerun text,
one/two 256-square image, and JSON `3/3` gates before any release discussion.

## Evidence

The raw profiler output remains outside the repository. Its local run ID is
`20260821T233617Z-qwen35-c8`; the two profiler wall times are 3.131220 seconds
for fixed-64 C1 and 4.071621 seconds for eight concurrent fixed-64 requests.
All routine gates passed with no OOM, HTTP 5xx, xgrammar/FSM error, RCCL/NCCL
fatal, or residual running/waiting request.
