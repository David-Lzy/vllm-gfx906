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
| Secondary model | Qwen3.8 27B AWQ 4-bit | experimental | experimental | v0.27 TP2 at 100K passed text, image, JSON, no-MTP, and MTP1; MTP1 improved fixed-128 decode 54.6%, but it is not a production throughput candidate |
| Serving | OpenAI-compatible API | verified-current | planned-v0.26 | Text, image URL/data URL, and JSON output |
| Serving | Cost-aware Router sidecar | unverified | experimental-rejected | Isolated C16/C32 evaluation did not clear the tail-latency gate; retain current Router |
| Topology | Four independent TP1 workers | verified-current | planned-v0.26 | Router-backed production topology |
| Context | 100K model length | verified-current | planned-v0.26 | Must retain high-context operation |
| Multimodal routine gate | 1/2 images at 256 x 256 | verified-current | planned-v0.26 | Required for each normal development and release gate; video remains disabled by default |
| Multimodal capacity | 8-64 images and 4096 x 4096 grid | verified-current capability | deferred-specialized | Run only when changing a capacity limit, scheduler, KV cache, or media-processing path |
| Quantization | Classic AWQ weights | verified-current | planned-v0.26 | Primary quantized weight path |
| Attention | gfx906 Triton attention | verified-current | planned-v0.26 | A fallback must remain available |
| Speculation | Qwen3.5 DFlash | verified-current (smoke) | experimental | Needs full throughput and quality evidence |
| Speculation | Qwen3.8 native MTP1 | experimental | experimental | v0.27 TP2 fixed-128 decode: 0.6828 tok/s versus 0.4417 no-MTP (+54.6%); 100K capacity falls from 4.80x to 4.34x |
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

## v0.28 experimental screen

The active integration branch also contains an isolated v0.28 screen on the
same MI50 hardware. It is not a production recommendation.

- Qwen3.5 9B AWQ recovered the retained v0.27 fixed-128 reference after
  restoring narrow gfx906 output-head and W4A16 accumulation paths: 76.72
  tok/s C1 and 264.60 tok/s synchronized C8 median.
- Qwen3.6 27B AWQ and Qwen3.8 27B AWQ each passed GPU2/GPU3 TP2 text,
  one/two 256-square image, and JSON 3/3 gates with drained metrics and no
  fatal error signature.
- This establishes model compatibility, not long-context performance. The
  historical gfx906 SplitKV selector was removed by the v0.28 paged-decode
  refactor, so the 27B screen used the generic Triton paged-attention fallback.
  Restoring that opt-in path is tracked separately before making any throughput
  claim for 27B on v0.28.
