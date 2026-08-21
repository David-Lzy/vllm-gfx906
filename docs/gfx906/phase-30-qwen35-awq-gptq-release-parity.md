# Phase 30: Qwen3.5 9B AWQ gfx906 GPTQ Release Parity

## Scope

Phase 28 introduced an explicit `gfx906_gptq` W4A16 adapter to avoid the
generic Triton path that made Qwen3.8 long-context decode unusable. This phase
tests whether the same adapter can make the current production checkpoint,
`cyankiwi/Qwen3.5-9B-AWQ-4bit`, a viable v0.27 replacement.

Both candidates used one isolated MI50 (GPU2), the same checkpoint, and the
current production serving settings: FP16 KV cache, 100K maximum context,
0.90 GPU memory utilization, eight sequences, chunked prefill, prefix cache,
and the current multimodal processor settings. Production GPU0/GPU1, Router,
and port 8002 were not changed.

The control was the retained v0.23 worker using its normal ExLlama selection.
The candidate was v0.27.1 with `--linear-backend gfx906_gptq`; its server log
confirmed `Gfx906GPTQWNA16LinearKernel` rather than a generic fallback.

## Results

| Scenario | v0.23 control tok/s | v0.27 gfx906 GPTQ tok/s | Ratio | Result |
| --- | ---: | ---: | ---: | --- |
| Text, C1 | 74.54 | 59.97 | 80.4% | Reject |
| Text, C8 | 223.37 | 125.27 | 56.1% | Reject |
| One 256-square image, C1 | 66.29 | 55.24 | 83.3% | Reject |
| Two 256-square images, C1 | 62.57 | 51.28 | 82.0% | Reject |

The candidate passed health, `/v1/models`, text and image responses, and JSON
constrained output `3/3`. Both benchmark error scans were empty; there were no
OOM, HTTP 5xx, xgrammar/FSM, or RCCL failures, and each run ended with no
running or waiting requests.

## Decision

Do not promote v0.27 or the explicit gfx906 GPTQ backend for the current Qwen
3.5 9B AWQ production workers. The release rule requires every routine path to
retain at least 95% of the control and at least one user-facing improvement;
this candidate misses the first condition by 16.6--43.9%.

This does not invalidate Phase 28. The backend is a targeted recovery for the
Qwen3.6/Qwen3.8 27B compressed-tensors long-decode case, where v0.27 had
routed to generic Triton W4A16. For the smaller production checkpoint, v0.23's
retained ExLlama path remains materially better end to end, especially under
concurrency. Retain the backend as an explicit development-only option and
continue the Qwen3.8 long-context work separately.

Raw artifacts are outside the repository under the Phase 30 results root;
their run identifier is `20260821T214158Z-qwen35-gptq-parity`.
