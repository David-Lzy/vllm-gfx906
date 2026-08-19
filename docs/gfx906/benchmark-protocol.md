# Benchmark protocol

## Purpose

Use one stable protocol for release decisions. A result is comparable only
when the model revision, image, topology, prompts, assets, sampling settings,
cache state, and request concurrency are recorded.

## Core workloads

| Workload | Requests | Concurrency | Input | Max output tokens |
| --- | ---: | ---: | --- | ---: |
| Text | 16 | 8 | Fixed text prompt | 512 |
| 8 images | 16 | 8 | 8 non-reused 128px images/request | 128 |
| 32 images | 8 | 4 | 32 non-reused 128px images/request | 256 |
| 64 images | 4 | 4 | 64 non-reused 128px images/request | 128 |
| Large grid | At least 4 | 1 and 4 | Fixed 4096 x 4096 grid | 128 |
| JSON constrained | 3 | 1 | Fixed schema and prompt | Test-defined |

Cache-reuse runs may be reported separately, but do not replace cold or
non-reused image results.

## Measurements

- Success count and HTTP status
- Requests/second and end-to-end p50/p95/max latency
- Prompt and completion tokens/second
- Time to first token when available
- Prompt, completion, and total token usage
- Host CPU, RAM, GPU utilization, power, and VRAM
- KV cache utilization, running requests, and waiting requests per engine
- Startup time and post-test idle recovery
- Relevant warnings, tracebacks, OOMs, RCCL failures, xgrammar/FSM failures,
  empty output, and repetition

## Quality checks

- Text and multimodal responses are non-empty and relevant.
- Deterministic image sanity cases preserve object, count, color, ordering,
  and visible-text observations.
- JSON-constrained requests pass 3/3 and validate against the schema.
- Output does not expose hidden reasoning or enter uncontrolled repetition.

## Release acceptance

- Every core workload succeeds with no stranded running or waiting requests.
- Aggregate core throughput is at least 95% of the validated release baseline.
- No core workload regresses by more than 10% without an explicit, reviewed
  tradeoff.
- Quality remains equivalent within the documented sanity rubric.
- No OOM, HTTP 500, fatal RCCL, xgrammar/FSM, or reproducible engine crash is
  accepted.

Run variance must be addressed with warmup plus repeated measured runs. Report
median results and include raw machine-readable records.
