# v0.28.0 gfx906.1

This release rebases the maintained MI50/MI60 path on upstream vLLM `v0.28.0`
while retaining only patches that passed focused gfx906 validation.

## Runtime stack

- vLLM: `0.28.0+gfx906.1`
- ROCm runtime: 7.2.1 base image
- PyTorch: 2.11
- Triton: gfx906 3.6 at commit `82957a511217`, plus the tracked
  conditional-pointer intersection patch
- Primary hardware: four 32 GiB AMD MI50 GPUs

## Retained paths

- gfx906 classic GPTQ/AWQ QGEMM, including M8 row-4 dispatch and FP32
  accumulation for W4A16 and packed INT8.
- gfx906 output-head handling, deferred device detection, and the explicit
  `gfx906_gptq` linear backend.
- Qwen head-256 SplitKV decode with bounded 29-partition controls.
- Packed-INT8 numerical fixes and checkpoint conversion tool.
- Qwen3.6 fused QK/RMSNorm/MRoPE/gate and Qwen GDN output normalization.
- Qwen3 VL multimodal truncation guard.

SplitKV-29, Qwen3.6 fusion, and GDN output normalization remain explicit
opt-ins. The validated Qwen3.5 9B AWQ TP1x4 Router profile does not enable
them.

## Image

```text
ghcr.io/david-lzy/vllm-gfx906:v0.28.0-gfx906.1
```

The image was rebuilt in one pass from source commit
`314389675e1f7e86c1788c56a7a1eb335fd083ed`. Its OCI version is
`0.28.0+gfx906.1`, and its OCI revision points to that commit. GitHub Packages
authorization was not completed during the release window, so the image has
not yet been pushed. The immutable digest and anonymous-pull result must be
added here before this tag is described as publicly deployable. No `latest`
tag is created.

## Validation

The release gate covers Qwen3.5 9B AWQ text, one/two 256-square images, JSON
constraints, C1/C8/C16/C32 and a 30-minute soak. Qwen3.8 27B standard AWQ and
packed INT8 use TP4 with SplitKV-29 and include a 32K prefix-cache-hit decode.
Qwen3.6 27B, 35B-A3B MoE, TurboQuant, FP8 KV, MTP and DFlash are compatibility
smokes rather than blanket performance claims.

The hardware results, including exact throughput and optional-feature startup
times, are recorded in the [release validation](validation-v0.28.0-gfx906.1.md).
All seven optional compatibility smokes passed. They remain default-off unless
the compatibility matrix says otherwise.

## Release assets

The [GitHub release](https://github.com/David-Lzy/vllm-gfx906/releases/tag/v0.28.0-gfx906.1)
includes a verified full-reference Git bundle, compact experiment evidence,
and a SHA256 manifest. These assets preserve the deleted experiment heads
without keeping a branch farm in the active repository.
