# Phase 16: selected v0.27 backports

## Outcome

Completed with no retained source backport. Two narrowly scoped upstream
changes were independently tested on the gfx906 v0.26 base. Both preserved the
routine functional gate, but neither reached the required five percent target
path improvement.

This result does not reduce the importance of native MTP for Qwen3.8. MTP1 was
already available in the compatible v0.26 control and nearly doubled the fixed
128-token decode rate. It is not evidence that the selected v0.27 Mamba patch
adds further speed.

## Test boundary

- Hardware: two AMD MI50 GPUs in TP2, isolated from the production endpoint.
- Qwen3.5 routine gate: text, one 256-square image, two 256-square images,
  JSON constrained output, and small C8 request sets.
- Qwen3.8 routine gate: text, one/two 256-square images, JSON 3/3, and a
  deterministic 128-token completion measurement.
- All candidates retained the same model revision, quantization, context
  length, tensor parallelism, sampling, and request payloads within their A/B.

## Candidate A: multimodal preprocessing executor

Upstream PR 49524 isolates multimodal preprocessing from API event-loop
rendering. Its practical compatibility benefit is real: a generation model can
use more than one renderer worker with a multimodal processor cache.

| Qwen3.5 C8 shape | v0.26 control | Executor, one renderer | Executor, four renderers + cache |
| --- | ---: | ---: | ---: |
| One image mean | 16.138s | 16.531s | 16.176s |
| Two image mean | 81.620s | 81.753s | 81.711s |
| Text mean | 5.435s | 5.438s | 5.440s |

All functional checks passed without HTTP 5xx, OOM, RCCL failure, structured
output error, or stranded work. The closest result is still slightly slower
than the control, so the patch is not promoted as a performance backport.

## Candidate B: Mamba scalar-fill synchronization avoidance

Upstream PR 49736 replaces scalar tensor assignments with in-place `fill_`
calls during MRV2 Mamba state setup. The Qwen3.8 27B AWQ TP2 target formed a
healthy server in both forms and passed all routine text, image, and JSON
checks.

| Qwen3.8 shape | No-MTP control | MTP1 control | MTP1 + scalar fill |
| --- | ---: | ---: | ---: |
| Fixed 128-token decode | 0.4370 tok/s | 0.8511 tok/s | 0.8481 tok/s |
| Text gate | 13.897s | 7.552s | 7.556s |
| One image gate | 51.110s | 31.735s | 30.875s |
| Two image gate | 152.022s | 81.017s | 80.656s |
| JSON 3/3 mean | 29.52s | 18.51s | 18.59s |

Native MTP1 improves the deterministic decode measurement by 94.8 percent over
no-MTP. The scalar-fill candidate averages 0.8481 tok/s, however, versus
0.8511 tok/s for the matching MTP1 control: a 0.35 percent regression. It does
not meet the five percent keep gate and is not combined with other backports.

## Observations

- The initial TP2 server setup remains expensive on gfx906: the two MTP1
  variants completed engine initialization in roughly 13 minutes. This is
  tracked separately as a cold-start/compiler concern, not credited as a
  request-path speed improvement.
- Routine results were stable and did not expose an MTP or multimodal
  correctness regression.
- The next active work is CPU affinity and thread-budget evaluation, followed
  by the isolated cost-aware Router experiment. A full v0.27 runtime remains
  blocked on the independent Triton 3.7 gfx906 compiler port.
