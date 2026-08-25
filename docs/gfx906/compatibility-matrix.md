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
| Secondary model | Qwen3.8 27B AWQ 4-bit | experimental | experimental | v0.28 TP4 standard weights passed text, image, JSON, C1/C8, and 32K cached decode; optional packed-INT8 embedding/head profile loads but fails output quality even after an isolated rule-order screen |
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
- A real two-worker Router canary then passed text, one/two 256-square image,
  JSON 3/3, idle-metric, and bounded fatal-log gates. It remained narrowly
  below the v0.27 production baseline: C1 `77.53 -> 75.71 tok/s`, C8 aggregate
  `420.26 -> 414.89 tok/s`, and C16 aggregate `527.99 -> 526.30 tok/s`.
  A follow-up same-GPU A-B-A direct-worker comparison reduced the attributable
  difference to `-0.11%` at C1 and `-0.94%` at C4, within ordinary worker/run
  variation. v0.28 stays an active development line rather than a Qwen3.5
  production promotion candidate because it has not shown a repeatable net
  gain over the retained v0.27 composition.
- Qwen3.6 27B AWQ and Qwen3.8 27B AWQ each passed GPU2/GPU3 TP2 text,
  one/two 256-square image, and JSON 3/3 gates with drained metrics and no
  fatal error signature. With the restored SplitKV path, Qwen3.6 matched its
  v0.27 reference or better: fixed-128 C1 `43.97 -> 44.74 tok/s` (+1.75%)
  and 32K cache-hit decode `11.84 -> 12.27 tok/s` (+3.67%).
- The v0.28 paged-decode refactor had removed the historical gfx906 SplitKV
  selector and left Qwen 27B on generic Triton paged attention. Phase 114
  restored a default-off, head-256-only SplitKV path for the compatible hybrid
  KV layout. On Qwen3.8 TP2, a matched 32K cache-hit fixed-128 decode improved
  from 2.55 to 12.28 tok/s (+382%). The same profile improved short fixed-128
  C1 from 42.24 to 44.08 tok/s (+4.4%) and C8 from 153.09 to 159.53 tok/s
  (+4.2%).
- The opt-in passed final text, one/two 256-square image, and JSON 3/3 gates.
  It is retained for v0.28 Qwen 27B development, not promoted as a production
  default until a separate model and serving canary is approved.
- Phase 118 then established the all-GPU v0.28 TP4/no-MTP standard-AWQ
  baseline. Qwen3.6 reached fixed-128 C1/C8/32K-cache medians of
  `52.65/216.53/17.21 tok/s`; standard Qwen3.8 reached
  `52.13/217.75/17.32 tok/s`. Both passed text, one/two 256-square image, and
  JSON 3/3 gates with drained metrics and empty fatal scans. The optional
  Qwen3.8 packed-INT8 embedding/head checkpoint is tracked separately. Phase
  119 restored its initial compressed-embedding construction wiring, but the
  loaded candidate produced malformed text, image, and JSON responses, so it
  is not a compatible v0.28 profile yet. Phase 120 then reordered the
  checkpoint's two packed-INT8 rules before its broad AWQ `Linear` rule without
  touching a weight or source line. It reached health but reproduced the same
  malformed outputs, rejecting rule precedence as the cause. Phase 121 then
  tested removal of the v0.28 Qwen3.5 weight mapper. It failed before health:
  the current fused GDN model has no destination for the checkpoint's separate
  `linear_attn.in_proj_a` key without that mapping. Mapper removal is therefore
  rejected as structurally incompatible, not merely slower.
