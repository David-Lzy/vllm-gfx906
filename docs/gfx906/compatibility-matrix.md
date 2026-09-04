# Compatibility matrix

Claims below apply to `v0.28.0-gfx906.1`, the recorded model revision and the
documented MI50 settings. `Supported` means that the release gate passed; it
does not imply support for every checkpoint using the same architecture.

| Area | Status | Default | Evidence or limit |
| --- | --- | --- | --- |
| MI50/MI60/Radeon VII (`gfx906`) | supported | yes | Real-hardware validation on four 32 GiB MI50 GPUs |
| OpenAI-compatible text serving | supported | yes | Text C1/C8/C16/C32 and soak |
| Qwen image input | supported | yes | One/two 256-square image release gate |
| JSON constrained output | supported | yes | Deterministic 3/3 gate |
| Qwen3.5 9B AWQ | supported with performance caveat | yes | Four TP1 workers and digest-pinned Router; v0.28 is functional but substantially slower than the v0.23 reference on the Phase 168 high-image, long-output replay |
| Qwen3.8 27B AWQ TP4 | development profile | opt-in | Text/image/JSON and 32K cache-hit decode |
| Qwen3.8 packed INT8 TP4 | development profile | opt-in | Numerical fix and Phase 153 comparison |
| Qwen3.6 27B AWQ TP4 | compatibility profile | opt-in | Text/image/JSON and SplitKV smoke |
| Qwen3.5 35B-A3B BF16 TP4 | compatibility profile | opt-in | Release text/image/JSON smoke passed; expert parallelism rejected |
| `gfx906_gptq` backend | supported | selected in examples | Classic W4A16 and packed INT8 paths; selection does not imply parity with the v0.23 ExLlama runtime on production-shaped multimodal load |
| SplitKV head-256 | supported | off | Enable with `VLLM_ROCM_ENABLE_GFX906_SPLITKV=1` |
| SplitKV-29 | long-context profile | off | Add max-splits 32 and force-splits 29 |
| Qwen3.6 fused QK/MRoPE/gate | experimental | off | Compiles with patched Triton; mixed traffic regressed |
| GDN output normalization | experimental | off | Small positive kernels, neutral/negative model results |
| FP8 E4M3 KV cache | capacity profile | off | Software path; MI50 has no native FP8 acceleration |
| TurboQuant KV | compatibility profile | off | Capacity increased; latency did not clear default gate |
| MTP | compatibility only | off | Qwen3.5 MTP1 release smoke passed; no production throughput claim |
| DFlash | compatibility only | off | Release smoke passed; draft model does not consume external multimodal embeddings |
| Expert parallelism | rejected on gfx906 | off | TP4+EP regressed; DP4+EP could not retain 100K KV |
| FP8/NVFP4/MXFP4 weights | out of scope | off | Fast paths target newer hardware |

The routine release gate deliberately uses text plus one/two 256-square images.
High image counts, 4096-square grids, and video are workload-specific capacity
tests and are not implied by this matrix.

Phase 168 additionally exercised a 40-request C32 replay containing 8--48
images per multimodal request and long generation budgets. The v0.23 reference
completed that workload 78.1% sooner than the selected v0.28 backend. See the
[Phase 168 report](phase-168-qwen35-v028-real-load-backend.md) before selecting
a runtime for a similar workload.
