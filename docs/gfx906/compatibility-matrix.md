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
| Secondary models | Qwen3.6/Qwen3.8 27B AWQ 4-bit | experimental | experimental | v0.28 TP4 standard weights passed text, image, JSON, C1/C8, and 32K cached decode; the packed-INT8 embedding/head profile passed independent Qwen3.8 and Qwen3.6 gates after the FP32-accumulation repair, with small retained development-only throughput gains; the Qwen3.6 fused QK/RMSNorm/MRoPE/gate composition was provisionally positive at fixed-128 decode, while its SplitKV-29 composition is long-context-only after a +14.83% 32K result paired with a -6.77% C8 regression |
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
  JSON 3/3 gates with drained metrics and empty fatal scans. Phase 123 restored
  FP32 accumulation for gfx906 INT8 GPTQ components. Phase 124 then validated
  a packed-INT8 Qwen3.8 embedding/head profile at `+1.55%` C1 and `+2.10%` C8
  versus its standard TP4 baseline. Phase 125 independently ported that
  profile to Qwen3.6, passing the same multimodal/JSON gates and averaging
  `+0.20%` C1 and `+4.24%` C8 against the Qwen3.6 standard-TP4 baseline.
  Both packed profiles remain development-only until each has a separate
  production model and serving canary rationale.
- Phase 126 then applied the same copy-on-write packed-INT8 embedding/head
  profile to Qwen3.5 9B on independent TP1 workers. It passed text, one/two
  256-square-image, and JSON 3/3 gates. Matched throughput regressed at C1
  (`-4.2%` text) but improved at C8 (`+8.0%` text, `+9.4%` one image, and
  `+3.9%` two images). A temporary two-worker Router confirmation preserved
  that saturated direction. The profile is retained for development only; it
  is not a Qwen3.5 production candidate until an interleaved multi-worker
  control and a mixed-workload canary prove a net end-to-end gain.
- Phase 127 profiled the C1 component pair before considering another custom
  kernel. The selected-token INT8 embedding gather took only `13.4us` across
  four active steps, while the INT8 vocabulary head took `6.00ms` versus
  `9.37ms` for the standard FP16 logits path in the captured geometry. The
  C1 regression is therefore not attributed to either packed component; no
  bespoke embedding or lm-head implementation is retained.
- Phase 136 revalidated the earlier Qwen3.8 SplitKV 29-partition choice on the
  v0.28 TP4 standard-AWQ path. A bounded default-off selector improved a
  matched 32K prefix-cache-hit decode from `18.04` to `20.47 tok/s` (+13.51%),
  with C1 effectively unchanged and C8 +1.78%. The option remains limited to
  compatible Qwen3.8 TP4 development profiles pending a dedicated canary.
- Phase 137 ported that same bounded `cap=32, force=29` SplitKV profile to
  standard Qwen3.6 27B AWQ TP4. It passed text, one/two 256-square image,
  JSON `3/3`, drained-queue, and fatal-log gates. Fixed-128 C1 was effectively
  unchanged (`52.904 -> 52.868 tok/s`, -0.07%), C8 improved
  `206.923 -> 217.289 tok/s` (+5.01%), and paired 32K prefix-cache-hit decode
  improved `17.784 -> 20.157 tok/s` (+13.34%). Retain it only as a compatible
  Qwen3.6 TP4 development opt-in; it does not alter the Qwen3.5 production
  Router or generic SplitKV defaults.
- Phase 138 tested whether the Phase 135 packed/fused Qwen3.6 overlay and the
  Phase 137 SplitKV-29 selection compose. It passed the routine text,
  one/two-image, JSON, drained-queue, and fatal-log gates. The 32K
  prefix-cache-hit median improved `17.979 -> 20.644 tok/s` (+14.83%), but C8
  regressed `235.620 -> 219.663 tok/s` (-6.77%). Retain the composition only
  as a Qwen3.6 packed-INT8 TP4 long-context development option; it is not a
  common serving default and does not change Qwen3.5 production.
- Phase 139 screened upstream vLLM PR `#50465` batch-sharded sampling on
  Qwen3.6 27B standard AWQ TP4. It is functional only with explicitly forced
  Model Runner V2 for this hybrid architecture: text, one/two 256-square image,
  and JSON `3/3` gates passed with empty fatal scans. Against its matched
  forced-V2 control, however, it reduced fixed-128 C1 from `49.2431` to
  `48.7565 tok/s` (-0.99%) and C8 from `211.5570` to `199.6205 tok/s`
  (-5.64%). The batch-sharded path is therefore evidence-only, not a Qwen3.6,
  Qwen3.8, or production profile candidate.
- Phase 140 tested the generic Triton unified-attention long-prefill geometry
  proposed in upstream issue `#52585`. On exact Qwen3.5 head-256 GQA tensors,
  `BLOCK_M=64` improved 8K prefill latency from `463.9822` to `327.7984 ms`
  (+29.35%) with matching FP16 tolerance. In the production-equivalent Qwen3.5
  9B AWQ service, however, the selector was never invoked: Qwen3.5 uses hybrid
  GDN and ROCm attention dispatches for this workload. Cache-busted service
  prefill was effectively unchanged (`9.993731 -> 10.005731 s`, -0.12%). Text,
  one/two 256-square image, and JSON `3/3` gates passed, but the change is
  source-level evidence only and is not merged or promoted.
- Phase 141 tested a pure-Triton unified-attention substitution for the Qwen
  27B full-attention decode route at the exact head-256, GQA 6:1, 784-token
  hybrid-page, 32K, 29-split geometry. It was numerically compatible but its
  per-layer median was `13.961 ms` versus `0.865 ms` for the retained gfx906
  SplitKV control (`-1513.4%`). The generic replacement is rejected before any
  server integration; the compatible SplitKV route remains the development
  baseline for this path.
