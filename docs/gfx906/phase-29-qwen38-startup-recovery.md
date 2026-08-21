# Phase 29: Qwen3.8 GPTQ Startup Recovery

## Scope

Phase 28 recovered the unusable gfx906 W4A16 decode path with the explicit
`gfx906_gptq` linear backend, but an unprepared restart still had a high
one-time cost. This phase tests upstream startup mechanisms before considering
a second on-disk representation of the quantized checkpoint.

All work used an isolated TP2 Qwen3.8 27B AWQ service on two MI50 GPUs. The
production workers, Router, model cache, and public endpoint were not changed.

## Configuration

The test server used the Phase 28 Qwen3.8 configuration: FP16 activations,
100K maximum model length, `--linear-backend gfx906_gptq`, and one persistent
vLLM cache directory. The routine gate was text, one and two 256-square images,
and JSON structured output three times. It intentionally does not repeat the
large-image capacity suite.

The runner now records the time to a healthy API endpoint and exposes the
following upstream controls for isolated startup tests:

- `VLLM_FORCE_AOT_LOAD=1` to assert reuse of the stored AOT artifact;
- safetensors strategy and prefetch tuning;
- `model_loader_extra_config` for the standard multithread loader; and
- an explicit `--kv-cache-memory-bytes` probe.

The underlying controls are documented by vLLM's
[load configuration](https://github.com/vllm-project/vllm/blob/main/vllm/config/load.py),
[default loader](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/model_loader/default_loader.py),
and [weight iterator](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/model_loader/weight_utils.py).

## Results

| Candidate | Health time | Model load | Engine init | KV capacity | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Initial cold control | 1167.277 s | 289.93 s | 620.68 s | 492,537 tokens | Baseline only |
| Persistent AOT, normal loader | 563.501 s | 285.38 s | 44.11 s | 464,179 tokens | Retain for development restarts |
| AOT plus `prefetch`, 8 threads, 16 MiB | 553.479 s | 284.06 s | 45.36 s | equivalent control path | Reject: only 1.8% faster |
| AOT plus multithread loader, 8 threads | 573.746 s | 289.29 s | 40.79 s | equivalent control path | Reject: 1.8% slower |
| AOT plus fixed KV allocation | 563.635 s | 283.52 s | 38.13 s | 437,313 tokens | Reject: 5.8% lower capacity |

The persistent AOT path is the clear win: it reduces health time by 51.7%.
Compilation falls from 232.65 seconds to 3.27 seconds and the initial
profiling/warmup step falls from 357.55 seconds to 14.22 seconds. The logs
show a direct load of both rank-specific AOT artifacts, rather than a fallback
compile.

The checkpoint sits on local ext4. vLLM correctly leaves automatic prefetch
off for that filesystem; forcing it placed the files into page cache in about
three seconds but did not materially change the roughly 284-second model-load
path. The multithread loader made some individual shard completions appear
earlier but had no end-to-end win. This is evidence that post-load W4A16
repacking and device materialization, rather than storage I/O, dominate this
checkpoint.

The fixed-KV probe skipped memory profiling as intended. It must not become a
default: the safe common allocation across the two workers reduced usable KV
capacity and vLLM explicitly warns that this bypasses normal
`gpu_memory_utilization` management.

An initial prefetch run intentionally used a different cache label while
`VLLM_FORCE_AOT_LOAD=1` was set. It failed with a missing AOT artifact instead
of silently compiling. That is the desired assertion behavior and establishes
that a loader comparison must retain the same persistent compile cache.

## Correctness And Retention

Each healthy candidate returned non-empty text and image results, passed JSON
`3/3`, and ended with zero running and waiting requests. There were no OOM,
RCCL fatal, xgrammar/FSM, or HTTP 5xx events. The recurring Transformers
warnings about undocumented Qwen video processor fields are upstream
documentation diagnostics; video input is disabled in this test and the
warnings did not affect requests.

Retain the persistent AOT cache for development image restarts, but do not make
`VLLM_FORCE_AOT_LOAD=1` a production default until cache invalidation and
deployment cache placement are documented. Do not retain forced prefetch,
multithread loading, or fixed KV allocation.

There is no general upstream or gfx906 community cache for the post-load GPTQ
shuffle/transpose representation. A local version would duplicate roughly 20
GiB of weights, still need source-validation reads, and would add a second
format to preserve. The official AOT reuse already removes the largest
avoidable portions of restart time, so a transformed-weight cache is deferred
rather than added speculatively.

## Exit

Phase 29 completes without any production promotion. Remove the temporary
Qwen3.8 checkpoint and compile cache after retaining the raw timing and gate
records outside the repository. Future work should focus on steady-state
decode, model coverage, and an explicitly approved canary rather than more
local-filesystem loader tuning.
