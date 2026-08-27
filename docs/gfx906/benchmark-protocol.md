# Benchmark protocol

## Purpose

Use one stable protocol for release decisions. A result is comparable only
when the model revision, image, topology, prompts, assets, sampling settings,
cache state, and request concurrency are recorded.

## Routine workloads

Routine validation is intentionally lightweight. It catches model loading,
multimodal mapping, vision encoding, JSON-constrained output, and idle recovery
without turning every kernel or dependency change into a long capacity run.

| Workload | Requests | Concurrency | Input | Max output tokens |
| --- | ---: | ---: | --- | ---: |
| Text | 1 smoke, then 4 warmed samples when performance is in scope | 1 | Fixed text prompt | 128 |
| One image | 1 smoke, then 4 warmed samples when performance is in scope | 1 | One fixed 256 x 256 image | 128 |
| Two images | 1 smoke, then 4 warmed samples when performance is in scope | 1 | Two fixed 256 x 256 images | 128 |
| JSON constrained | 3 | 1 | Fixed schema and prompt | Test-defined |

Each measured scenario records its own request count. A repeat contaminated by
unrelated traffic is excluded from a production comparison.

## Capacity and scheduler workloads

32-image, 64-image, and 4096 x 4096 grid payloads are not routine development
or release gates. Run them only in a separately declared capacity experiment
when changing a multimodal limit, scheduler policy, KV-cache representation, or
media-processing path. Cold assets in such an experiment must differ in decoded
pixel content, not only filenames or metadata. Cache-reuse results remain
diagnostic and do not replace cold results.

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

- Text plus one/two 256-square image smoke and JSON 3/3 succeed with no
  stranded running or waiting requests.
- When a release changes a performance-critical path, warmed text and two-image
  latency is at least 95% of the validated release baseline, unless an explicit
  reviewed tradeoff says otherwise.
- A capacity/scheduler release additionally runs its declared dedicated capacity
  suite; ordinary releases do not inherit that requirement.
- Quality remains equivalent within the documented sanity rubric.
- No OOM, HTTP 500, fatal RCCL, xgrammar/FSM, or reproducible engine crash is
  accepted.

Run variance must be addressed with warmup plus repeated measured runs. Report
median results and include raw machine-readable records.
