# Phase 43: v0.27 PyTorch 2.11 HIP Graph Runtime Recovery

## Goal

Determine whether the retained PyTorch 2.11 ROCm 7.2 HIP graph runtime removes
the normal-concurrency Qwen3.5 9B AWQ v0.27 regression on MI50/gfx906 without
giving up v0.27 functionality. The candidate keeps v0.27.1 source, the
gfx906-specific W4A16 patches, and Triton 3.6; it rebuilds every native vLLM
extension against PyTorch 2.11 rather than swapping a wheel across ABIs.

## Evidence

The ordinary HTTP C8 comparison is stable: v0.23 reaches a median `216.703`
completion tok/s, while v0.27 reaches `157.900` tok/s with the explicit gfx906
GPTQ selector. Automatic ExLlama reaches `157.549` tok/s, so the selector is
not the cause.

Both images use HIP `7.2.53211`; the material runtime difference is PyTorch
`2.11.0a0+git70d99e9` plus Triton `3.6.0` in the retained worker versus
PyTorch `2.13.0+rocm7.2` plus Triton `3.7.1` in v0.27. Matching C8 profiler
captures show the model-forward mean rising from `14.86 ms` to `48.98 ms` and
host-visible `hipGraphLaunch` mean rising from `0.389 ms` to `39.16 ms`.

Two bounded v0.27 runtime controls are closed:

- disabling the ROCm CUDA-graph memory estimate changes C8 by only `+0.04%`;
- disabling async scheduling lowers C8 from `157.900` to `143.364` tok/s;
- forcing `PIECEWISE` reaches `155.630` tok/s and falls back from ROCm custom
  paged attention to Triton.

This is a compatibility experiment, not an upstream claim that PyTorch 2.11
is generally preferable. vLLM's ROCm source-build guidance explicitly requires
a source build for a non-validated PyTorch combination, and the v0.27 source
still has explicit compatibility guards for older compiler APIs.

## Candidate

`docker/Dockerfile.gfx906-v027-phase43-pytorch211` starts from the retained
gfx906 ROCm 7.2.1 image and creates a separate virtual environment that uses
its PyTorch 2.11 and Triton 3.6 through system site packages. It installs
v0.27 ROCm Python dependencies, compiles the native wheel with
`PYTORCH_ROCM_ARCH=gfx906`, and installs that wheel only in the temporary
environment.

The image does not contain model weights and must not replace production.

## Gates

1. Build and import must report vLLM v0.27, PyTorch 2.11, HIP 7.2.53211, and
   Triton 3.6.
2. GPU2 only: text, one 256px image, two 256px images, and JSON `3/3` pass.
3. Ordinary fixed-64 C8 is measured after warmup using the same routine
   Qwen3.5 9B AWQ fixture as Phase 41.
4. Retain only if C8 improves by at least 5% over v0.27 PyTorch 2.13 and all
   routine gates remain stable. It becomes a release candidate only if it
   reaches at least 95% of the v0.23 C8 baseline.
5. On build, import, output, or runtime failure, retain the error evidence,
   delete only disposable build cache, and leave production unchanged.

## Community Context

- [vLLM #48453](https://github.com/vllm-project/vllm/issues/48453) measured a
  separate 17--20% ROCm decode regression from CUDA-graph memory reservation.
  The local no-estimate test rules that specific mechanism out as the main
  cause here.
- [vLLM #43801](https://github.com/vllm-project/vllm/issues/43801) reports
  host blocking around `hipGraphLaunch` on AMD. The local traces exhibit the
  same symptom, although disabling async scheduling regresses our workload.
- [vLLM ROCm build guidance](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.rocm/) says vLLM extensions must be rebuilt when using a
  different PyTorch/ROCm stack; a binary-wheel swap is intentionally excluded.

## Result: Graph Runtime Restored, Release Throughput Not Recovered

The candidate built and imported with v0.27.1 source, PyTorch
`2.11.0a0+git70d99e9`, HIP `7.2.53211`, and Triton `3.6.0`. It passed the
complete routine gate on GPU2: text, one 256px image, two 256px images, and
JSON `3/3` all returned normal output. Final metrics were idle and the log
scan contained no OOM, HTTP 5xx, traceback, xgrammar/FSM, or RCCL/NCCL fatal
event.

The runtime swap does repair the profiler-visible graph symptom:

| C8 trace item | v0.27 PyTorch 2.13 | v0.27 PyTorch 2.11 control |
| --- | ---: | ---: |
| `hipGraphLaunch`, mean | 39.16 ms | 0.503 ms |
| `gpu_model_runner: forward`, mean | 48.98 ms | 21.77 ms |

However, the ordinary HTTP result remains below both the retained worker and
the phase's retention threshold:

| Routine scenario | v0.23 reference | v0.27 PyTorch 2.11 control | Relative |
| --- | ---: | ---: | ---: |
| Text C1 | 75.109 tok/s | 61.385 tok/s | 81.7% |
| Text C8 | 216.703 tok/s | 157.573 tok/s | 72.7% |
| One image C1 | 66.660 tok/s | 56.961 tok/s | 85.5% |
| Two images C1 | 62.974 tok/s | 52.937 tok/s | 84.1% |

The C8 result is statistically a tie with the PyTorch 2.13 v0.27 result
(`157.900 tok/s`). PyTorch 2.11/Triton 3.6 therefore removes a real
graph-replay penalty but does not restore end-to-end serving performance. The
same trace still assigns `2.041 s` over 8,192 calls to native GPTQ W4A16,
compared with `1.085 s` in the retained worker.

## Decision

Reject PyTorch 2.11 as a standalone release recovery. Keep this image as a
diagnostic base for the next isolated W4A16 kernel composition test; it does
not replace production.
