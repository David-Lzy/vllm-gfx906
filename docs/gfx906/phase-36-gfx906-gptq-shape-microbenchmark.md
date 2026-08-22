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
