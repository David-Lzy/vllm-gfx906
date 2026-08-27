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

The current versioned release is `v0.28.0-gfx906.1`. Its validated production
profile is Qwen3.5 9B AWQ with text and image inputs. Qwen3.8 27B AWQ is
functionally compatible in targeted TP4 configurations, but remains a
development performance path rather than the default production model on MI50.

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

The target versioned image is:

```text
ghcr.io/david-lzy/vllm-gfx906:v0.28.0-gfx906.1
```

The source release and hardware validation are complete. Registry publication
is pending the repository owner's GitHub Packages authorization. Until the
release notes contain an immutable `sha256` digest and an anonymous-pull result,
build the image from the tagged source instead of assuming the tag is public.

See the [versioned GitHub release](https://github.com/David-Lzy/vllm-gfx906/releases/tag/v0.28.0-gfx906.1)
for the source tag, full Git recovery bundle, compact benchmark evidence, and
checksums. Registry publication status is recorded in the
[release notes](docs/gfx906/release-v0.28.0-gfx906.1.md).

There is deliberately no floating `latest` tag. The image is built from this
repository in one pass on top of the pinned ROCm 7.2.1/PyTorch 2.11 gfx906
base. Triton 3.6 is rebuilt from its pinned source commit with the retained
conditional-pointer patch; no local Phase image is part of the build chain.

### Single GPU

```bash
export HF_CACHE_DIR=/srv/vllm/huggingface
export VLLM_CACHE_DIR=/srv/vllm/cache
./deploy/gfx906/run-single-gpu.sh
curl http://127.0.0.1:8002/v1/models
```

### Four MI50 GPUs

The validated high-throughput topology runs one TP1 worker per GPU behind a
digest-pinned vLLM Router:

```bash
cd deploy/gfx906
cp .env.example .env
# Edit only the cache paths and model if required.
docker compose --env-file .env \
  -f docker-compose.tp1x4-router.yaml up -d
curl http://127.0.0.1:8002/health
```

The public Compose binds to `127.0.0.1` by default and does not require
`--privileged`. Initial model download and compiler warmup commonly take
10-20 minutes. Model weights are never included in the image.

### Optional gfx906 paths

The release contains several measured narrow paths, all disabled by default:

- `VLLM_ROCM_ENABLE_GFX906_SPLITKV=1` enables the head-256 Qwen SplitKV path.
- `VLLM_ROCM_GFX906_SPLITKV_MAX_SPLITS=32` and
  `VLLM_ROCM_GFX906_SPLITKV_FORCE_SPLITS=29` select the retained TP4
  long-context profile.
- `VLLM_ROCM_ENABLE_GFX906_QWEN36_FUSED_QK_ROPE_GATE=1` enables the Qwen3.6
  fused QK/RMSNorm/MRoPE/gate path.
- `VLLM_ROCM_ENABLE_GFX906_QWEN_GDN_OUTPUT_NORM=1` enables the measured GDN
  output-normalization reshape elision.

Do not combine optional paths without matching model-level evidence. The safe
Qwen3.5 production profile leaves all four variables unset.

### Build From Source

Use a clean source checkout and keep model weights, build products, compiler
caches, credentials, and deployment configuration outside Git:

```bash
git clone --recursive https://github.com/David-Lzy/vllm-gfx906.git
cd vllm-gfx906
git remote add upstream https://github.com/vllm-project/vllm.git
git remote add mobydick https://github.com/ai-infos/vllm-gfx906-mobydick.git
git remote -v
./tools/gfx906/build-release-image.sh local/vllm-gfx906:v0.28.0-gfx906.1
```

Before building, confirm that the host ROCm kernel stack can see the device and
that the intended PyTorch, ROCm, and Triton revisions have an explicit gfx906
validation record. The release process documents the required source pinning,
image provenance, canary, and rollback gates:

- [gfx906 documentation index](docs/gfx906/README.md)
- [release process](docs/gfx906/release-process.md)
- [patch ledger](docs/gfx906/patch-ledger.md)
- [benchmark protocol](docs/gfx906/benchmark-protocol.md)
- [evidence lifecycle](docs/gfx906/evidence-lifecycle.md)

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
