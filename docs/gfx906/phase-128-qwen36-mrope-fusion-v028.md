# Phase 128: Qwen3.6 MRoPE Fusion Backport Screen

## Decision

Rejected on gfx906 before model serving. Upstream commit `07ef21bc69`
extends the fused QK-RMSNorm, partial RoPE, and output-gate Triton kernel for
Qwen3.6 multimodal MRoPE. It is useful on CUDA, but its activation is guarded
by `current_platform.is_cuda()`. MI50 reports ROCm, not CUDA, so the candidate
cannot select the optimized path in an unmodified ROCm build.

## Evidence

- Source branch: `backport/qwen36-mrope-fusion-v028`, local backport commit
  `56ed1e5ec1`.
- Candidate image built successfully from the retained v0.28 gfx906 base.
- A direct GPU2 MRoPE numerical smoke reached Triton compilation, then failed
  in `TritonAMDGPUCanonicalizePointers` while lowering the conditional pointer
  merge in the fused kernel.
- The v0.28 control's original text-only version of the same kernel fails at
  the identical compiler assertion. This identifies an existing gfx906 Triton
  compiler limitation, not an MRoPE-indexing regression.

The failing compiler condition is an unequal fat-pointer assertion in
`ConvertSCFIfOp`. Enabling the ROCm path by changing the CUDA guard would turn
a deliberately disabled optimization into a deterministic compiler failure.

## Scope And Safety

No Qwen server was launched, no production worker was stopped, and no
production configuration changed. The disposable image and temporary cache
were removed after recording this result. The source branch is retained as
reproducible rejected evidence, not as a merge candidate.

## Revisit Gate

Revisit only when the gfx906 Triton fork accepts the conditional-pointer
kernel, or after a separate kernel rewrite has passed direct MRoPE numerical
parity. A future candidate must pass that focused GPU check before consuming a
TP4 maintenance window.
