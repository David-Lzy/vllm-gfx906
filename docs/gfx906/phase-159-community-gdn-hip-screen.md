# Phase 159: Community GDN HIP source screen

## Status

Completed: source-level no-go. No image build, GPU workload, or production
maintenance window was justified.

## Question

Screen the current community gfx906 fork for a narrow implementation that can
improve Qwen3.5 9B AWQ or Qwen3.8 27B AWQ on MI50 beyond the retained v0.28
profiles.

## Source examined

- Fork: [`ttdxq/gfx906-vllm`](https://github.com/ttdxq/gfx906-vllm)
- Revision: `72dfb924f6d08c7b9d744f2e2676199cab0baf51` (2026-08-25)
- License: Apache-2.0, with the fork's own `NOTICE` attribution record

The source contains two gfx906-specific GDN/Mamba decode operations in
`csrc/mamba/gfx906_decode_kernels.cu`: a causal-convolution update and fused
sigmoid-gating delta-rule update. It also carries a GPTQ `q_gemm.cu` and an
opt-in AWQ-to-GPTQ-compatible route.

## Findings

### GPTQ and AWQ path

The community `q_gemm.cu` kernel bodies are materially the same as the
retained gfx906 legacy QGEMM source in this fork. The local changes are the
required v0.28 stable-operator API adaptation and current-stream/BLAS-handle
wrappers; they are not an alternative tile, wave, or quantized-GEMM algorithm.
Importing that file would therefore replace current integration work without
introducing a new performance hypothesis.

The community README also labels the AWQ/GPTQ route as partial and recommends
the conservative Triton path by default. It supplies no matched MI50 Qwen3.8
AWQ service benchmark or correctness evidence that could justify overriding
the current explicit `gfx906_gptq` profile.

### GDN/Mamba decode path

The GDN HIP implementation is a broad, fork-specific integration spanning
more than 3,000 lines and two custom operators. It is not a small standalone
kernel replacement on the current v0.28 model-executor interfaces.

More importantly, Phase 102 measured the actual retained Qwen3.8 TP4 short
decode profile. On the rank-0 GPU, GPTQ consumes 52.8% of C1 time and 38.4% of
C8 time, while RCCL consumes 24.1% and 35.2%. Hybrid Mamba/GDN consumes only
3.6% and 2.8%, respectively. Even an impossible zero-cost GDN implementation
could not meet the project's targeted service-gain threshold, and it would
not improve the dominant QGEMM or collective costs.

The project has also already rejected a narrower Qwen3.8 GPTQ custom-kernel
experiment after a two-image gate caused an illegal instruction and a
development-GPU RAS disable. A speculative 3,000-line HIP integration has an
unacceptable risk-to-upside ratio under that evidence.

## Decision

Do not port the community GDN/Mamba HIP implementation and do not build its
fork as a candidate image. The existing local legacy QGEMM implementation
already covers its only potentially relevant quantized-GEMM source.

The performance search remains focused on measured dominant categories:

1. gfx906 QGEMM only when a new safe, shape-specific source candidate exists;
2. TP topology and RCCL communication for the concurrent 27B profile;
3. upstream releases that explicitly include pre-CDNA ROCm/gfx906 coverage.

## Reopen gate

Reopen this route only when all of the following are available:

1. a fresh Qwen3.8 profile shows GDN/Mamba at least 10% of the target request
   class;
2. the community source has a minimal, reviewable v0.28 adaptation with a
   real MI50 Qwen3.8 AWQ correctness and throughput comparison; and
3. the candidate can be isolated with the Phase 75 HIP containment runner.
