# Phase 129: Triton 3.6 Conditional-Pointer Intersection

## Decision

Retained as a gfx906 compiler prerequisite. The Triton 3.6 AMD pointer
canonicalization pass asserted when two `scf.if` branches yielded pointer
metadata with unequal narrowing attributes. That stopped both the existing
Qwen text fusion and the Qwen3.6 interleaved-MRoPE fusion before their kernel
could run.

The retained patch ports Triton 3.7's conservative intersection rule to the
gfx906 Triton 3.6 fork. A merged pointer may narrow only if both inputs may
narrow; only identical metadata attributes survive; and a small-tensor base is
kept only when both branches use the exact same base. It does not discard a
failed assertion without preserving a safe metadata contract.

## Evidence

- Target compiler base: `ai-infos/triton-gfx906` commit `82957a5`.
- Local compiler patch commit: `93f6542`.
- A CPython 3.12 Triton 3.6 wheel built successfully from the isolated worktree.
- `git apply --check` passes against the clean retained compiler base using the
  exported patch under `patches/triton-3.6-gfx906/`.
- GPU2 MI50 direct BF16 parity passed for:
    - text RoPE, 37 tokens, 24 query and 4 KV heads;
    - interleaved Qwen3.6 MRoPE, 37 tokens, 16 query and 2 KV heads.

Both cases compare the fused Q/K normalization, RoPE, and gate-copy output to
vLLM's native reference within the established BF16 tolerance. Neither now
reaches `TritonAMDGPUCanonicalizePointers::ConvertSCFIfOp`.

## Scope

This phase did not launch a model server, stop a production worker, change the
Router, alter a production image, or enable the model-side ROCm dispatch. The
compiler repair is intentionally separated from that feature decision. Phase
130 evaluates the guarded model dispatch and TP2 serving behavior next.

## Reproduction

Apply `0001-amd-scf-pointer-intersection.patch` to the documented Triton 3.6
gfx906 base before building the wheel. The companion README records the exact
base and application command. Build products, model weights, caches, and
machine-specific configuration remain outside Git.
