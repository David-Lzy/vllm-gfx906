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

The final immutable digest and anonymous-pull verification are recorded here
after publication. No `latest` tag is created.

## Validation

The release gate covers Qwen3.5 9B AWQ text, one/two 256-square images, JSON
constraints, C1/C8/C16/C32 and a 30-minute soak. Qwen3.8 27B standard AWQ and
packed INT8 use TP4 with SplitKV-29 and include a 32K prefix-cache-hit decode.
Qwen3.6 27B, 35B-A3B MoE, TurboQuant, FP8 KV, MTP and DFlash are compatibility
smokes rather than blanket performance claims.
