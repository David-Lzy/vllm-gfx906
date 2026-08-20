# Compatibility matrix

Status meanings:

- `verified-current`: validated on the existing gfx906 production lineage
- `planned-v0.26`: requires validation or porting on v0.26
- `experimental`: useful research path, not a release default
- `out-of-scope`: not targeted by the first v0.26 release

| Area | Mode | Current lineage | v0.26 target | Notes |
| --- | --- | --- | --- | --- |
| Hardware | AMD MI50/MI60 (`gfx906`) | verified-current | planned-v0.26 | Real-hardware validation is mandatory |
| Primary model | Qwen3.5 9B AWQ 4-bit | verified-current | planned-v0.26 | Text and multimodal parity target |
| Serving | OpenAI-compatible API | verified-current | planned-v0.26 | Text, image URL/data URL, and JSON output |
| Topology | Four independent TP1 workers | verified-current | planned-v0.26 | Router-backed production topology |
| Context | 100K model length | verified-current | planned-v0.26 | Must retain high-context operation |
| Multimodal routine gate | 1/2 images at 256 x 256 | verified-current | planned-v0.26 | Required for each normal development and release gate; video remains disabled by default |
| Multimodal capacity | 8-64 images and 4096 x 4096 grid | verified-current capability | deferred-specialized | Run only when changing a capacity limit, scheduler, KV cache, or media-processing path |
| Quantization | Classic AWQ weights | verified-current | planned-v0.26 | Primary quantized weight path |
| Attention | gfx906 Triton attention | verified-current | planned-v0.26 | A fallback must remain available |
| Speculation | Qwen3.5 DFlash | verified-current (smoke) | experimental | Needs full throughput and quality evidence |
| MoE | 35B-A3B BF16 TP4 | verified-current (older experiments) | planned-v0.26 | No EP requirement for base support |
| KV cache | TurboQuant `k8v4` | verified-current (smoke) | experimental | Optional capacity mode |
| KV cache | TurboQuant `4bit_nc` | verified-current (smoke) | experimental | Optional capacity mode |
| KV cache | FP8 E4M3 | unverified | experimental | MI50 has no native FP8 fast path |
| MoE | Expert parallelism | unverified | experimental | Test after TP4 stability |
| Weights | FP8 weights | unverified | out-of-scope | Not a first-release optimization for MI50 |
| Weights | NVFP4/MXFP4 | unverified | out-of-scope | Hardware fast paths target newer GPUs |

Compatibility claims apply only to the exact image, model revision, settings,
and benchmark evidence recorded for a release.

## v0.27 exploration boundary

Upstream v0.27 is not currently a gfx906 runtime candidate. Its PyTorch 2.13
and Triton 3.7 dependency path rejects gfx906 in the compiler pipeline before
the server becomes healthy. The fork therefore treats v0.27 as two separate
research tracks:

- self-contained upstream backports may be evaluated on the v0.26 gfx906 base;
- a full v0.27 runtime requires a separately validated Triton 3.7 gfx906 port.

The selected-backport track covers Qwen3.5 9B AWQ multimodal preprocessing and
Qwen3.8 27B AWQ hybrid/Mamba execution independently. It excludes internal
data-parallel routing, video-only features, MI300-specific optimizations, and
startup-only improvements until they have a direct gfx906 use case.
