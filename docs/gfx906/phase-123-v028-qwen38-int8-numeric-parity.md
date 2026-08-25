# Qwen3.8 packed-INT8 numeric parity and precision fix

## Scope

This phase followed the v0.28 loader-parity audit for the local Qwen3.8 27B
packed-INT8 embedding/LM-head profile. The checkpoint, model mapper, and
rank-local tensors had already been shown to agree with the validated v0.27
profile. The remaining question was whether the compute path preserved the
same numerical result.

The work is limited to the optional legacy gfx906 GPTQ `bit == 8` path. The
normal Qwen3.5 9B W4A16/AWQ production path is not selected by this change.

## Findings

The embedding gather was exact in both runtimes. For fixed token IDs, the
packed INT8 gather returned zero maximum and mean absolute error against a
direct dequantized reference.

The output-head projection was different. With identical packed tensors and
fixed FP16 hidden states, measured error against the same reference was:

| Runtime | Maximum absolute error | Mean absolute error |
| --- | ---: | ---: |
| v0.27 retained profile | 0.00133276 | 0.000542868 |
| v0.28 before fix | 0.00748682 | 0.00296130 |
| v0.28 with fix | 0.00133276 | 0.000542868 |

The v0.28 port had retained the generic FP16 partial-accumulation implementation
for the legacy INT8 GPTQ output head. The v0.27 gfx906 implementation instead
used FP32 group scales and FP32 partial sums before converting the final result
to FP16. In a large-vocabulary autoregressive model, a small logit difference
can alter greedy next-token selection and then cascade into malformed output.

## Fix

The retained source change restores the FP32 accumulation behavior under the
existing `VLLM_GFX906_LEGACY_QGEMM` compile option only when `bit == 8`. It
also zero-initializes the INT8 atomic-accumulation output before launch. The
W4 path remains the existing implementation.

The Qwen3.5 embedding constructor receives the configured quantization object
and its prefix so the optional compressed embedding table is recognized. The
Qwen3.8 GDN mapper remains enabled; removing it is known to prevent checkpoint
loading.

## Full-model validation

On two MI50 GPUs in tensor parallelism, the repaired v0.28 candidate passed:

- Text, one 256-square image, and two 256-square image requests with coherent
  output.
- JSON-constrained output `3/3`.
- No OOM, traceback, RCCL/NCCL fatal, xgrammar/FSM failure, or stranded
  running/waiting requests.
- Fixed 128-token text decode of `44.72 tok/s` C1 and `164.72 tok/s` C8
  aggregate across three measurements. These match the retained v0.27
  packed-INT8 reference (`45.34` and `165.96 tok/s`) within normal variation.

## Decision

Retain the narrow compatibility fix in the gfx906 source line and add it to
future Qwen3.8 packed-INT8 candidates. It resolves the v0.28 semantic
regression without claiming a general INT8 or W4 performance improvement.
The production deployment remains the separately validated Qwen3.5 9B AWQ
service until its own release/canary criteria are met.
