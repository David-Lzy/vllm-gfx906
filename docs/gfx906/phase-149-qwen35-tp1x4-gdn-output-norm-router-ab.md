# Phase 149: Qwen3.5 TP1x4 GDN Output-Norm Router A/B

## Decision

**Rejected for production default.** The retained Qwen GDN output-norm
reshape-elision overlay is correct and stable, but it does not produce a
repeatable end-to-end gain in the selected four independent TP1 workers plus
Router topology. It remains available behind its opt-in environment variable
for future, more targeted work.

## Contract

The three arms used the exact selected Qwen3.5 9B AWQ production topology:

- four MI50 TP1 workers and the vLLM Router with `round_robin`;
- 100K context, FP16 KV cache, `max-num-seqs=8` per worker, 32K batched
  tokens, image limit 64, `max_pixels=16777216`;
- the same image, model, Router image, CPU/thread configuration, and
  production-compatible OpenAI workload.

The candidate alone set:

```text
VLLM_ROCM_ENABLE_GFX906_QWEN_GDN_OUTPUT_NORM=1
```

The sequence was `control-a -> candidate -> control-b`; each arm used three
measured rounds after text, one-image, two-image, and JSON constrained smoke.
All test assets were cache-busted.

## Results

Median completion throughput in tokens/s:

| Scenario | Control A | Candidate | Control B | Candidate vs control mean |
| --- | ---: | ---: | ---: | ---: |
| C1 text | 77.19 | 77.54 | 77.40 | +0.32% |
| C8 text | 513.38 | 503.47 | 501.14 | -0.75% |
| C16 mixed text/image | 668.88 | 668.67 | 663.87 | +0.35% |
| C32 mixed text/image | 856.66 | 840.43 | 850.11 | -1.52% |
| C32 text | 1027.82 | 1035.60 | 1033.23 | +0.49% |

The candidate did not clear the retained-gain gate: no primary Router-load
aggregate improved by at least 1%, and the C32 mixed workload regressed.

## Correctness and Stability

- Text, one-image, two-image, and JSON constrained requests all succeeded.
- Each worker completed 68 or 69 requests in every arm, so Router imbalance
  did not explain the result.
- All request queues drained to zero after each arm.
- No OOM, traceback, xgrammar/FSM, RCCL/NCCL fatal, RAS event, or illegal
  instruction signature was found.
- The selected production Compose file remains unchanged; the environment
  variable stays unset in production.

## Evidence and Follow-up

Raw JSON/JSONL, resolved Compose configurations, worker metrics, and logs are
kept outside Git in the phase result store. The reproducible runner and its
candidate-only Compose overlay are maintained on the dedicated benchmark
branch. A follow-up should revisit this overlay only when a Qwen GDN kernel or
Router scheduling change alters the C16/C32 bottleneck; it is not a standalone
production optimization on the current line.
