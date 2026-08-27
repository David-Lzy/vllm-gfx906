# Triton 3.6 gfx906 patches

These patches target the retained `ai-infos/triton-gfx906` Triton 3.6 base at
commit `82957a5` (`Add gfx906 support to triton v3.6.0`). They are not applied
to an upstream Triton checkout automatically.

Apply the patch before building the Triton wheel:

```bash
git apply /path/to/0001-amd-scf-pointer-intersection.patch
```

`0001` ports Triton 3.7's conservative fat-pointer metadata intersection to
the gfx906 Triton 3.6 compiler. It is required to compile the fused
QK-RMSNorm/RoPE/gate kernels used by the Qwen3.6 MRoPE evaluation. The patch is
validated by direct BF16 parity tests for both 1D RoPE and interleaved MRoPE;
it does not by itself enable the feature in vLLM model dispatch.
