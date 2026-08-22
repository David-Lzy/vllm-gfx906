# Phase 36: gfx906 GPTQ Shape Microbenchmark

## Goal

Evaluate a targeted gfx906 GPTQ W4A16 decode implementation for the real
Qwen3.5 9B AWQ shapes identified in Phase 35. The goal is to recover enough of
the v0.27 C8 deficit to clear the retained v0.23 production floor without
changing model precision, API behavior, or multimodal processing.

## Preconditions

- Use development GPU2 only; production GPU0/GPU1, port 8002, Router, Compose,
  and production cache are read-only.
- Reuse the isolated Qwen3.5 9B AWQ cache. Do not download model weights into
  the repository.
- Check and temporarily suspend XMR only when its live state can affect the
  isolated measurement; restore its previous state after the run.

## Matrix

Use the model's already-packed GPTQ weights. Compare the existing
`_C::gptq_gemm` operator with any candidate at:

| M | N | K |
| ---: | ---: | ---: |
| 1, 8, 27, 190 | 24,576 | 4,096 |
| 1, 8, 27, 190 | 4,096 | 12,288 |
| 1, 8, 27, 190 | 12,288 | 4,096 |
| 1, 8, 27, 190 | 4,096 | 4,096 |
| 1, 8, 27, 190 | 10,240 | 4,096 |

For each row, measure warm median latency, p95, and output error against the
current op. The baseline records repeat variation separately because K-split
atomic accumulation is not bitwise deterministic. Record group size, GPTQ
format, packed layout, device clock, and the number of K-split atomic updates
per output.

## Candidate Order

1. Establish the exact current-op baseline with actual packed weights.
2. Inspect whether an existing gfx906-compatible path can represent the same
   format and output semantics. RDNA3 wave32/WMMA paths are evidence only and
   are not enabled on gfx906.
3. If the baseline confirms at least a 10% end-to-end-equivalent hotspot,
   prototype one HIP C++ operator limited to the verified fp16, symmetric,
   no-act-order configuration. Start with K=4096 decode rows and reduce
   K-split atomics only when occupancy remains adequate.
4. Add an operator correctness test before any server benchmark. Extend the
   existing GPTQ test suite rather than adding a benchmark under `tests/`.
5. Run text, one/two 256px images, and JSON 3/3 only after operator correctness
   and microbenchmark retention gates pass.

## Retention Gates

- Numerical error must stay within the existing GPTQ fp16 tolerance for every
  supported shape.
- No fallback or candidate may alter checkpoints, quantization metadata, or
  select automatically on non-gfx906 hardware.
- A candidate must improve the target operator by at least 15% on the dominant
  shape and improve end-to-end C8 decode by at least 5% before it is retained.
- Stop a branch on compile failure, any incorrect output, or a regression on
  M=1/8 decode. Do not optimize only M=27/190 if normal serving regresses.

## Result

The actual packed `cyankiwi/Qwen3.5-9B-AWQ-4bit` checkpoint is symmetric
4-bit GPTQ with group size 32. The baseline splits the K=4096 decode shapes
into 32 atomic partial sums per output. A narrowly guarded HIP C++ trial used
256-wide K blocks, reducing that to 16 for K=4096, only for fp16 symmetric
4-bit input, no activation order, `K=4096`, `M<=8`, and output widths divisible
by 1024.

The trial built on gfx906 and its output stayed close to the existing operator:
the largest measured relative L2 delta was `0.00156`. This is consistent with
the expected change in fp16 atomic accumulation order, not a model-level
format change. The dominant fused MLP gate/up microbenchmark at M=8 improved
from `0.4389 ms` to `0.3527 ms` (19.6%). M=1 did not improve, and unsupported
shapes fell back to the original operator.

The isolated GPU2 vLLM server then used the same 100K-context Qwen3.5 setup as
Phase 35. It passed text, one/two 256px image, and JSON `3/3` gates with no
HTTP 5xx, OOM, RCCL/NCCL fatal, xgrammar/FSM error, or stuck request. Its
fixed-64 text timing was nevertheless only:

| Load | Existing Phase 35 | K=256 trial | Change |
| --- | ---: | ---: | ---: |
| C1 | 3.091 s | 3.029 s | +2.0% throughput |
| C8 | 4.017 s | 3.967 s | +1.3% throughput |

The run had no capacity backlog after the gates. Both C1 and C8 result files
report 64 completion tokens per request, so the elapsed-time comparison is
direct. Raw traces and JSON responses remain outside Git under the local Phase
36 results directory.

## Community Cross-check And Decision

This branch cannot address the historical Qwen3.8 `0.238 tok/s` long-context
result. Recent community investigation confirms that hybrid Qwen models with
256-wide heads and a Mamba-aligned non-power-of-two KV page fall into the
Triton paged-attention fallback at small-batch decode. The upstream
[split-KV paged decode PR](https://github.com/vllm-project/vllm/pull/45916)
implements the appropriate split-and-reduce recovery; its author reports a
47.2% C1 gain at 32K context, while the related
[ROCm performance issue](https://github.com/vllm-project/vllm/issues/50264)
reports that blindly widening the native ROCm selector is invalid because the
native kernel has no matching 256-wide-head/page instantiation.

The fork's explicit split-KV composition already follows that design and
improved same-model Qwen3.8 32K cached decode from `0.906` to `1.337 tok/s` in
Phase 31. That is the retained solution to the `0.238 tok/s` class of problem.
The K=256 GPTQ experiment is rejected: its 1.3--2.0% end-to-end change is below
the 5% retention gate and does not justify extra custom-op maintenance. The
trial implementation and image are not promoted; the reusable baseline
microbenchmark remains for a future independently justified GPTQ candidate.
