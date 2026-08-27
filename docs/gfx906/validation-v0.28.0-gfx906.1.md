# v0.28.0 gfx906.1 validation

This record covers the release candidate built from commit `32ccc4a39a` and
tested on four 32 GiB AMD MI50 GPUs. Later release-only documentation and CI
changes do not alter the tested runtime paths. The published image is rebuilt
from the final clean release source before publication.

## Qwen3.5 9B production profile

The primary profile used four independent TP1 workers behind the pinned vLLM
Router. Text, one and two 256-square image requests, JSON constraints `3/3`,
and drained queues passed without HTTP 5xx, OOM, RCCL fatal, xgrammar, or FSM
errors.

| Workload | Release median | Historical production median | Change |
| --- | ---: | ---: | ---: |
| Text C1 | 77.35 tok/s | 77.68 tok/s | -0.4% |
| Text C8 | 516.61 tok/s | 499.61 tok/s | +3.4% |
| Mixed C16 | 705.03 tok/s | 691.58 tok/s | +1.9% |
| Mixed C32 | 852.29 tok/s | 823.42 tok/s | +3.5% |
| Text C32 | 1037.63 tok/s | 1018.05 tok/s | +1.9% |

The four workers received 46, 46, 46, and 45 requests in the release run. The
final production promotion adds a separate 30-minute soak to this release
candidate gate.

## Qwen3.8 27B development profiles

Both profiles used TP4, the explicit `gfx906_gptq` backend, and opt-in
SplitKV-29. They passed text, one/two image, JSON `3/3`, fatal-log, and drained
queue gates.

| Profile | Text C1 | Text C8 | 32K cache-hit decode | Startup |
| --- | ---: | ---: | ---: | ---: |
| Standard AWQ | 56.73 tok/s | 226.13 tok/s | 20.75 tok/s | 462 s |
| Packed INT8 | 57.61 tok/s | 245.11 tok/s | 20.84 tok/s | 462 s |

Relative to their matched historical results, the release candidate improved
standard AWQ by 6.1% at C1, 9.3% at C8, and 0.5% at the 32K cache-hit gate. The
packed-INT8 profile improved by 4.8%, 11.0%, and 1.5%, respectively. Packed
INT8 remains a development profile rather than the production default.

## Optional compatibility smokes

Each row passed health, text, one image, two images, JSON `3/3`, fatal-log
scan, and drained queues. These are compatibility results, not throughput
claims.

| Variant | Startup | Result | Release position |
| --- | ---: | --- | --- |
| TurboQuant `4bit_nc` KV | 422 s | passed | optional capacity mode |
| TurboQuant `k8v4` KV | 211 s | passed | optional capacity mode |
| FP8 E4M3 KV | 211 s | passed | optional software path |
| Qwen3.5 MTP1 | 241 s | passed | model-specific, default off |
| Qwen3.5 DFlash, 8 draft tokens | 311 s | passed | default off |
| Qwen3.6 fusion plus SplitKV-29 TP4 | 442 s | passed | explicit opt-in |
| Qwen3.5 35B-A3B BF16 MoE TP4 | 482 s | passed | compatibility profile |

DFlash logged that the draft model does not consume external multimodal
embeddings; image requests still completed through the target model. This is
why the row is a compatibility smoke rather than a multimodal acceleration
claim. The 35B MoE path selected the Triton unquantized MoE backend and logged
that no MI50-specific tuning file exists, so no MoE performance claim is made.

## Release decision

The release candidate clears the functional and performance gates for the
Qwen3.5 9B production profile and the two retained Qwen3.8 development
profiles. Narrow features stay opt-in. High image counts, 4096-square grids,
video, expert parallelism, and newer-hardware weight formats are outside this
release gate.
