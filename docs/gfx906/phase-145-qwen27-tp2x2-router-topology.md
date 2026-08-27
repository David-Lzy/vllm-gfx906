# Phase 145: Qwen 27B TP4 versus TP2x2 Router topology

## Decision

Retain the two-TP2-engine topology as a high-concurrency development option for both standard Qwen3.6 and Qwen3.8 27B AWQ checkpoints. Do not use it as a default for isolated interactive requests: direct TP4 remains faster at C1. This evidence does not change the separate Qwen3.5 9B production service.

## Contract

The comparison used the same v0.28 gfx906 image, standard AWQ checkpoints, explicit `gfx906_gptq`, FP16 KV cache, no MTP, 100K context, eight sequences per engine, 8,192 batched tokens, image limit 64, and the routine 256-square image gate. The control was one TP4 engine on four MI50 GPUs. The candidate was two independent TP2 engines on GPU pairs 0/1 and 2/3 behind an isolated round-robin vLLM Router.

Every arm passed text, one/two-image, and JSON `3/3` smoke gates. The benchmark uses three fixed-128 rounds for C1, text C8, and mixed C16 (text plus one- and two-image requests). No OOM, HTTP 5xx, xgrammar/FSM, RCCL/NCCL, RAS, or illegal-instruction signature was found. The Router sent 40 and 41 completed requests to its two workers, respectively.

## Results

Aggregate completion throughput in tokens per second; deltas are against the same-checkpoint TP4 control.

| Checkpoint | Topology | C1 | C8 text | Mixed C16 |
| --- | --- | ---: | ---: | ---: |
| Qwen3.6 27B AWQ | TP4 direct | 52.91 | 205.47 | 187.83 |
| Qwen3.6 27B AWQ | TP2x2 Router | 43.21 (-18.3%) | 233.64 (+13.7%) | 277.77 (+47.9%) |
| Qwen3.8 27B AWQ | TP4 direct | 52.94 | 203.97 | 200.22 |
| Qwen3.8 27B AWQ | TP2x2 Router | 44.15 (-16.6%) | 235.70 (+15.6%) | 271.38 (+35.5%) |

The same split is visible in median end-to-end round time. For Qwen3.6, C8 fell from 4.98s to 4.38s and mixed C16 from 10.90s to 7.37s; Qwen3.8 C8 fell from 5.02s to 4.34s and mixed C16 from 10.23s to 7.55s. C1 rises because a single request loses TP4's four-way tensor-parallel execution and additionally pays Router dispatch overhead.

## Operational use

- Use direct TP4 for latency-sensitive single or lightly concurrent requests.
- Use two TP2 engines behind Router for batch or mixed concurrent Qwen 27B traffic. Both models clear the retained topology gate: Router C8 and C16 throughput improved by at least 10 percent with all routine gates passing.
- A future serving canary may expose the TP2x2 service separately. It must not replace the current Qwen3.5 9B Router by implication; model choice and topology are separate production decisions.
- Cold startup includes a long engine/shared-memory warmup. It is a maintenance consideration, not part of the steady-state score above.

## Reproduction

The harness creates isolated containers and networks, records per-worker metrics and logs, verifies drained queues, and restores the selected production service through an exit trap. It intentionally excludes high-image-count, 4096-square, MTP, KV-quantization, and kernel-selector sweeps so the result is attributable to serving topology.
