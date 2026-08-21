# Phase 22: Qwen3.8 27B AWQ/MTP on gfx906

## Scope

This isolated result evaluates `cyankiwi/Qwen3.8-27B-AWQ-INT4` on two AMD
MI50 (`gfx906`) GPUs with vLLM v0.27.1, tensor parallelism two, 100K context,
and the normal text plus one/two 256-square image and JSON routine gate. It
does not change a production deployment.

## Compatibility result

The checkpoint revision `63768c10df38c0395e12ef49edac1bd539eaeeea` declares
`compressed-tensors`, so vLLM was allowed to select the checkpoint quantizer
rather than being forced into an AWQ mode from the repository name.

The Qwen3.8 MRoPE interface change is required for startup. The PyTorch 2.13
wheel's bundled RCCL initialized on gfx906 but failed on the first TP2
collective. A retained ROCm 7.2.1 gfx906 RCCL shared library completed the
standalone two-rank all-reduce and the complete server test. The Phase 22
Dockerfile copies that library from the retained runtime in a multi-stage
build, so the result does not rely on a host bind mount.

Both no-MTP and native MTP1 passed health, model discovery, text, one-image,
two-image, JSON `3/3`, and idle request checks. No HTTP 500, OOM, traceback,
xgrammar/FSM, or RCCL/NCCL fatal was observed.

## Fixed-128 decode result

| Mode | Median completion throughput | 100K theoretical KV concurrency | Outcome |
| --- | ---: | ---: | --- |
| no-MTP | 0.441734 tok/s | 4.80x | Functional baseline |
| native MTP1 | 0.682798 tok/s | 4.34x | +54.57% decode, -9.49% capacity |

MTP1 has a stable improvement and should remain available as an experimental
Qwen3.8 option. It is not a production replacement for the existing smaller
Qwen deployment: cold start remains expensive and the 27B TP2 throughput is
far below the established 9B serving topology.

## Follow-up

The next performance work should isolate the v0.27 gfx906 cold Triton/LLVM
compile path. It must preserve the validated RCCL compatibility layer and use
the same bounded text, image, and JSON gate before any promotion discussion.
