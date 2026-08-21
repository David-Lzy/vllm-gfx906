# Phase 28: Qwen3.8 Hybrid Decode Attribution

## Scope

Phase 22 proved that `cyankiwi/Qwen3.8-27B-AWQ-INT4` is functional on two
gfx906 MI50 GPUs with the retained RCCL compatibility bridge. Native MTP1
improved fixed 128-token decode, but the absolute result remains unsuitable for
production. This phase is the dedicated Qwen3.8 throughput-optimization track:
it first measures the decode path, then implements the evidence-backed Triton,
HIP C++, or communication fix that can bring the model to the matching
Qwen3.6 27B AWQ throughput.

The work is isolated to GPU2/GPU3. It does not modify the GPU0/GPU1 production
workers, Router, port 8002, model cache, or Compose deployment.

## Questions

The model combines 48 Gated DeltaNet/Mamba layers with 16 full-attention layers
whose head dimension is 256. The profile must distinguish these categories:

1. W4A16 compressed-tensors linear work;
2. full paged attention, including any ROCm-to-Triton fallback;
3. Gated DeltaNet/Mamba work and state movement; and
4. per-token TP2 collectives.

## Reference And Objective

Before profiling Qwen3.8, run a fresh control with
`cyankiwi/Qwen3.6-27B-AWQ-INT4` on the same v0.27 image, retained RCCL bridge,
two development MI50 GPUs, TP2, 100K context, and fixed 128-token decode.
Measure no-MTP and native MTP1 separately when the checkpoint exposes those
prediction layers. Delete the reference checkpoint cache before restoring the
Qwen3.8 cache.

The final objective is Qwen3.8 throughput at least equal to the matching
Qwen3.6 reference for the same mode. An interim patch may be retained only if
it improves its target mode by at least ten percent with no correctness or
stability regression. No score mixes an older runtime, different quantization,
or a different TP topology.

## Community Cross-check

An independent report tested the same model family on three 16 GiB gfx906 MI50
cards using llama.cpp, ROCm 6.2.4, Q8_0 weights, tensor split, and one slot. It
reported about 29 tok/s without MTP and 37.33 tok/s with MTP depth two at short
context, with 75% draft acceptance. This is not an apples-to-apples vLLM/AWQ/
two-GPU comparison, so it is not a release target. It does show that gfx906 is
not intrinsically limited to the Phase 22 decode rate and makes a kernel or
runtime-path investigation worthwhile.

The same report found that disabling graphs cost roughly three percent of
short-context throughput but avoided a process-local post-236K slowdown. Add a
native vLLM MTP depth sweep of one, two, and four after the reference control.
Record acceptance, KV capacity, and peak VRAM. Treat graph disabling only as a
long-context stability diagnostic, never as a short-decode default without a
separate result.

Source: [Unsloth discussion 47](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/discussions/47).

## Method

Use a small, opt-in HIP-event timing hook inside the vLLM engine process. It
records aggregate device time and call counts for the existing execution
boundaries only, avoiding a production API change or a misleading host-side
timer. Test warmed fixed-128 decode for no-MTP and MTP1, then retain the normal
text, one/two 256-square image, and JSON `3/3` correctness gate. Implement
only the winning category: Python is limited to instrumentation, Triton is
eligible for an isolated kernel experiment, and a new HIP C++ custom op is
permitted when the profile proves a required gfx906 path is missing.

Before implementation, search vLLM issues, open pull requests, ROCm kernels,
and llama.cpp's HIP backend for the specific measured category. Reuse an
upstream-compatible approach where possible and record why any independent
implementation differs.

The checkpoint stays in an isolated temporary cache and is removed after the
evidence is recorded. No 32/64-image or 4096-square workload is part of this
phase.

## Reproducible Controls

`tools/run-gfx906-phase28-qwen27b.sh` runs only on the reserved development
pair. It refuses to start unless the production health endpoint is healthy;
all candidate weights, compile cache, logs, and raw results live outside the
repository. The control order is Qwen3.6 first, then explicit cleanup, then
Qwen3.8. Each model is measured in no-MTP mode first, followed by MTP depths
one, two, and four only while the previous mode stays correct and non-regressive.

## Community-first Split-KV Candidate

Before considering a new HIP C++ attention operator, evaluate the open upstream
[PR #45916](https://github.com/vllm-project/vllm/pull/45916). It adds a Triton
split-KV decode and reduction path for FP16/BF16, 256-wide heads and nonstandard
physical blocks. That is an architectural match for this model family: the
local Qwen3.6 control config has 24 query heads, 4 KV heads, 256-wide heads,
and the runtime has already selected physical blocks of 784 tokens.

The PR is not claimed compatible with MI50: its upstream gate is `on_gfx1x()`,
whereas MI50 is `gfx906`. After the control result, a disposable gfx906-only
candidate must first pass the PR's kernel-equivalence test adapted to FP16,
24/4 heads, and block 784. Only then may it run the routine service gate and a
context-length slope comparison. Reject it on Triton compilation failure,
numerical mismatch, or a short-decode regression. This is an opt-in reuse
experiment, not a platform-wide gate widening.

The experimental branch carries the upstream implementation with provenance,
then narrows its runtime guard to `on_gfx906()` plus
`VLLM_ROCM_ENABLE_GFX906_SPLITKV=1`. Its default is off, and the normal Phase
28 reference image does not contain the overlay. This makes a failed MI50
compile or numerics result a contained candidate failure rather than a change
to any other ROCm target.

The runner also records an identical long text prefix followed by a prefix-cache
reuse decode. This separates one-time prefill from the decode context slope that
split-KV is intended to change; fixed 128-token results remain the short-context
control rather than evidence for or against the long-context candidate.

## Initial Qwen3.6 Control

The same-runtime Qwen3.6 27B AWQ no-MTP control completed its five fixed
128-token samples at a median `0.441037 tok/s`. At 32,780 prompt tokens, the
first prefill completed after its one-time large-shape JIT in 618.65 seconds.
The immediately repeated prefix-cache request reported a 49% hit rate, but its
eight-token decode still required 33.57 seconds (`0.2383 tok/s`). This 46%
short-to-long decode decline is direct local evidence for the attention-context
slope before testing the gfx906 split-KV candidate.

The first long-prefix attempt used the upstream default 300-second executor
timeout and died in the multiprocess `sample_tokens` RPC while the large-shape
Triton kernel compiled. The runner now explicitly uses an 1,800-second timeout
for this isolated long-context probe. This avoids treating a first-JIT timeout
as a model or kernel-equivalence result; it is not a production configuration
change.

## Decision Gate

- Consider a HIP C++ paged-attention implementation only when full paged
  attention is at least 30% of decode time or has a material context-length
  slope. It must implement the actual 256-wide head and Mamba-aligned page
  geometry; widening a selection gate alone is invalid.
- Consider a gfx906 W4A16 custom linear implementation only when linear work is
  at least 40% of decode and the existing ExLlama/Triton candidates cannot
  service the observed shapes efficiently.
- If TP2 collective time dominates, create a RCCL/topology follow-up rather
  than misclassifying a communication limit as a model operator defect.
- If Mamba/GDN dominates, first compare vLLM's existing Triton kernels through
  a focused microbenchmark. The prior scalar-fill experiment was neutral, so it
  is insufficient evidence for a new kernel.

## Exit

Publish the category breakdown, the one highest-value next implementation path,
and a no-go or success decision. No result from this phase is a production
promotion by itself.
