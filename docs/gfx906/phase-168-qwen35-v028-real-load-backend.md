# Phase 168: Qwen3.5 v0.28 Backend Real-Load A/B

## Status

The isolated matrix and 30-minute winner canary are complete. The guarded
runner selected the v0.23 ExLlama arm, but its live-business canary was
interrupted by an independent intake hold. The fixed 60-minute throughput gate
therefore failed on an interval containing roughly 30 minutes without offered
business load. The runner restored the frozen v0.28 deployment and passed the
post-recovery contract checks. No runtime was promoted by this phase.

## Question

Production-shaped Phase1 measurements showed that the validated v0.23 lineage
could process the Qwen3.5 9B AWQ multimodal workload substantially faster than
the selected v0.28 build. Phase 168 separates the version effect from the
explicit linear-backend choice under matched topology, model, request bodies,
generation budgets, and Router policy.

## Compared Arms

| Arm | Runtime | Linear backend | Role |
| --- | --- | --- | --- |
| A | selected v0.28 gfx906 image | `gfx906_gptq` | production control |
| B | same v0.28 image | `exllama` | backend-only candidate |
| C | frozen v0.23 image | historical Exllama path | version reference |
| D | same v0.28 image | `auto` | dispatch identity only |

The measured arms retain four independent TP1 workers, eight sequences per
worker, 100K model length, FP16 KV cache, 32,768 batched tokens, renderer 1,
the same multimodal limits, and the same pinned round-robin Router.

## Protocol

1. Verify immutable image IDs and the SHA-256 of the 120-request replay.
2. Pause new business intake and wait for every business stage and vLLM worker
   to drain without cancellation.
3. Keep a separately verified fallback endpoint available while all MI50 GPUs
   are used by isolated loopback-only experiment deployments.
4. Require text, one/two 256-pixel image, and JSON-object 3/3 smoke tests.
5. Run fixed C1/C8 tests and a stratified 40-request C32 pilot. Preserve raw
   responses so output volume and response compatibility can be reviewed.
6. Exclude failed, unsafe, more than 15% slower, or materially truncated arms;
   run the full 120-request replay for the eligible leader(s).
7. Promote only a quality-safe candidate at least 5% faster than Arm A. A
   result inside 5% prefers v0.28, then the existing backend.
8. Run a loaded 30-minute loopback canary and a 60-minute production canary.
   Any failed gate restores the exact frozen production configuration.

Business request bodies, responses, host paths, credentials, and deployment
configuration are intentionally excluded from Git.

## Results

All three ranked arms passed text, one/two 256-pixel image, JSON-object 3/3,
and response-contract gates without HTTP failures, OOM, container restarts,
fatal signatures, or residual requests. The v0.28 automatic probe selected
`TritonW4A16LinearKernel`; the explicit arms selected the requested
`Gfx906GPTQWNA16LinearKernel` or `ExllamaLinearKernel` on every worker.

### Real-load pilot

The production-shaped pilot contained 40 frozen requests at C32. Ten were
text-only; the other 30 carried 8--48 images, including seven 48-image
requests. Every request retained its original body and `max_tokens=32768`.

| Arm | Success | Completion tokens | Makespan | Completion tok/s | p50 | p95 | p99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A: v0.28 `gfx906_gptq` | 40/40 | 268,965 | 5,806.8 s | 46.32 | 875.8 s | 4,920.3 s | 5,673.5 s |
| B: v0.28 `exllama` | 40/40 | 233,792 | 6,213.4 s | 37.63 | 953.0 s | 4,553.7 s | 5,796.0 s |
| C: v0.23 ExLlama | 40/40 | 254,014 | 1,273.9 s | 199.40 | 336.7 s | 1,147.7 s | 1,248.9 s |

Arm C reduced makespan by 78.1% and produced 4.31 times Arm A's completion
throughput. Its output-token total was 94.44% of Arm A, above the 90% quality
guard and below the 10% difference that requires a truncation review. Arm B
was both 7.0% slower by makespan and 18.8% slower by token throughput than Arm
A. It also produced only 86.92% of Arm A's tokens, so it is not promotion
eligible even apart from its performance loss.

The old runtime won despite less favorable Router-balance metrics. Its
`idle-while-queued` ratio was 23.28%, compared with 3.15% for Arm A, and its
worker queue time was 753.0 versus 487.7 seconds. Router skew therefore cannot
explain the v0.28 regression in this workload.

### Small requests do not predict the production result

The median fixed text C8 rates were 180.25 tok/s for Arm A, 172.74 tok/s for
Arm B, and 167.50 tok/s for Arm C. The v0.28 control therefore looked 7.6%
faster than v0.23 on that small test while taking 4.56 times as long on the
real-load pilot. Backend changes should not be promoted from short decode
micro-workloads alone.

### Attribution supported by this experiment

Forcing the same ExLlama backend name in v0.28 did not recover performance; it
made the real-load result worse. The dominant regression is consequently not
the top-level W4A16 backend selector alone. It lies elsewhere in the newer
runtime's long-output, large multimodal request path, or in an interaction
between that path and the linear kernels. A future source-level optimization
phase should profile matched long requests before changing another isolated
kernel.

The full 120-request stage was not run. After applying the 15% pilot cutoff,
only Arm C remained, and completing additional multi-hour arms would have
crossed the reserved restoration window. This limits the pilot's statistical
depth, but the 4.56-times makespan separation, clean contracts, 30-minute
loaded canary, and prior business A/B evidence all point in the same direction.

## Production Decision

Arm C was the only quality-safe candidate within 15% of the pilot leader and
cleared the required 5% improvement over Arm A. It completed 47 consecutive
C8 mixed canary rounds without a failed command during the isolated 30-minute
window. The guarded runner promoted the immutable v0.23 image while retaining
the four-TP1-worker topology, 100K model length, eight sequences per worker,
and the pinned round-robin Router.

The live canary remained runtime-safe: all four workers and the Router stayed
healthy with zero OOM events, restarts, preemptions, or fatal-log matches. It
recorded 328,819 generated tokens and 96 successful backend requests. However,
an independent Server2 workflow asserted an intake hold about five minutes
after business resume. The client then drained its in-flight work and exited
after about 30 minutes. The runner divided the token count by the complete
3,607.7-second observation interval and obtained 91.14 tok/s, below its fixed
90%-of-pilot gate.

That gate result is not a valid measurement of v0.23 under continuous offered
load. It is recorded as an externally interrupted canary, not as a v0.23
performance or stability failure. The recovery path restored the immutable
v0.28 control image and the pinned Router, then passed model discovery, text,
one-image, two-image, and JSON-object 3/3 smoke tests. All containers were
healthy with zero restarts and no OOM state after recovery.

Phase 168 supports two decisions:

1. Keep v0.23 ExLlama as the leading candidate for this high-image,
   long-output workload. Its isolated advantage is large and repeatable enough
   to justify another production canary.
2. Do not promote it from this run. A retry needs an exclusive business-load
   interval and a demand-aware gate that detects an external intake hold rather
   than averaging idle time into throughput.

The selected v0.28 release remains functionally supported, but Phase 168 shows
that its Qwen3.5 production-shaped throughput claim must be qualified. Short
C8 tests are not representative of this workload.
