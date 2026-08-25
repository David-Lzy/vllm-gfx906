# vLLM gfx906

Experimental vLLM maintenance work for AMD `gfx906` GPUs, including Radeon
VII, MI50, and MI60. This fork exists to make old-but-capable AMD hardware a
first-class, evidence-driven development target while remaining easy to compare
with upstream vLLM.

> [!WARNING]
> This is not an official vLLM release and is not a general ROCm distribution.
> Support claims apply only to the exact source revision, software stack, model,
> and hardware recorded in the accompanying benchmark evidence. Do not deploy
> `main` or an unpinned image to production without reproducing the required
> checks on your own hardware.

## Scope

- Maintain and test gfx906 compatibility on current vLLM integration lines.
- Measure changes on real MI50-class hardware before treating them as usable.
- Keep Qwen text and image workflows as the primary regression surface.
- Preserve focused patches and benchmark evidence, including negative results.
- Keep upstream-compatible changes reviewable; this fork is not a wholesale
  replacement for upstream ROCm support.

## Current Status

The public documentation describes an active, experimental v0.27-era gfx906
line. The validated small-model reference is Qwen3.5 9B AWQ with text and
image inputs. Qwen3.8 27B AWQ can be made functionally compatible in targeted
configurations, but it is an experimental performance path rather than a
production recommendation on MI50.

The project does not publish a floating `latest` image. Build and deployment
artifacts are only meaningful when accompanied by an immutable source commit,
image digest, model revision, and hardware result.

| Area | Position |
| --- | --- |
| Target GPUs | Radeon VII, MI50, MI60 (`gfx906`) |
| Primary validation model | Qwen3.5 9B AWQ, text plus image inputs |
| Secondary research model | Qwen3.8 27B AWQ and related W4A16 formats |
| Routine regression gate | text, one/two 256 x 256 images, JSON 3/3 |
| Large image or video workloads | specialized tests, not a generic claim |
| Production promotion | explicit canary and rollback review required |

The detailed status, known limits, and exact benchmark conditions live in
[`docs/gfx906/`](docs/gfx906/README.md). Start with the
[compatibility matrix](docs/gfx906/compatibility-matrix.md) and
[benchmark protocol](docs/gfx906/benchmark-protocol.md), not an old command
copied from a container page.

## Getting Started

Use a clean source checkout and keep model weights, build products, compiler
caches, credentials, and deployment configuration outside Git:

```bash
git clone --recursive https://github.com/David-Lzy/vllm-gfx906.git
cd vllm-gfx906
git remote add upstream https://github.com/vllm-project/vllm.git
git remote add mobydick https://github.com/ai-infos/vllm-gfx906-mobydick.git
git remote -v
```

Before building, confirm that the host ROCm kernel stack can see the device and
that the intended PyTorch, ROCm, and Triton revisions have an explicit gfx906
validation record. The release process documents the required source pinning,
image provenance, canary, and rollback gates:

- [gfx906 documentation index](docs/gfx906/README.md)
- [release process](docs/gfx906/release-process.md)
- [patch ledger](docs/gfx906/patch-ledger.md)
- [benchmark protocol](docs/gfx906/benchmark-protocol.md)

## What This Fork Does Not Promise

- Support for every vLLM model, quantization format, ROCm version, or AMD GPU.
- MI300-, RDNA3-, CDNA3-, or Blackwell-specific performance paths on gfx906.
- Fast FP8, NVFP4, MXFP4, or arbitrary compressed-weight inference on MI50.
- A production-safe response to unreviewed third-party model checkpoints.
- A container command that requires broad host mounts or privileged execution.

## Contributing

Keep each change narrow and reproducible. A contribution should identify the
exact gfx906 target, the upstream base, the affected model and quantization
path, correctness coverage, and before/after performance evidence. Do not
submit model weights, caches, compiled artifacts, credentials, or local
deployment files.

For a potentially upstreamable fix, extract the smallest portable patch and
benchmark it independently. A cross-fork GitHub "Compare & pull request"
banner is not evidence that an entire experimental branch belongs upstream.

## Relationship to Upstream

This repository is a downstream experimental fork of
[vllm-project/vllm](https://github.com/vllm-project/vllm). It also preserves
historical context from
[ai-infos/vllm-gfx906-mobydick](https://github.com/ai-infos/vllm-gfx906-mobydick).
Neither upstream project endorses every change or support claim made here.
