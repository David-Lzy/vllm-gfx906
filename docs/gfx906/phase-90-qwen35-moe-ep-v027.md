# Phase 90: Qwen3.5 35B-A3B v0.27 Expert Parallelism

## Purpose

Rebuild the deferred MoE expert-parallelism experiment on the MI50 gfx906
v0.27 lineage. The original Phase 09 inputs were deliberately deleted with the
retired v0.26 image, so this phase does not reuse its stale launcher or compare
across runtime generations.

The model is the public BF16 multimodal checkpoint `Qwen/Qwen3.5-35B-A3B`.
It was evaluated only on four MI50 GPUs during an authorized maintenance
window. The Qwen3.5 9B production Router remained a separate rollback service.

## Matrix

| Mode | TP | DP | EP | Purpose |
| --- | ---: | ---: | ---: | --- |
| `tp4` | 4 | 1 | off | Same-runtime BF16 MoE control. |
| `tp4-ep` | 4 | 1 | on | Compatibility probe. It shards experts, but does not form the intended DP4 serving topology. |
| `dp4-ep` | 1 | 4 | on | Primary EP candidate. Experts are sharded across four EP ranks while attention is replicated by DP. |

The test uses `allgather_reducescatter`, the ROCm-available general-purpose
all-to-all backend. DeepEP, DeepGEMM, FP8 weights, MTP, EPLB, and DBO are out
of scope: they require newer hardware-specific kernels or add variables before
the baseline topology is known to work on gfx906.

## Contract And Gates

All modes use FP16 runtime/KV cache, a 100K context limit, two sequences per
rank, 8,192 batched tokens, 64 images with video disabled, a 16 MiB image
limit, multimodal shared-memory cache, prefix caching, chunked prefill, and
thinking disabled. The exact same image asset and vLLM image are used for every
row.

Before benchmark collection each service must return text, one 256-square
image, two 256-square images, and JSON-constrained output three times. The
comparison records fixed-128 C1 text/image and C8 text aggregate throughput,
startup duration, KV capacity, final queue state, per-rank memory/power, and
fatal-log signatures.

## Results: 2026-08-24

All temporary modes used
`local/vllm-gfx906:v0.27.1-phase66-legacy-qgemm-c8-row4` and the public BF16
`Qwen/Qwen3.5-35B-A3B` snapshot. Production was stopped only for this
authorized four-GPU window, then restored and passed `/health`, model listing,
text, one-image, and JSON 3/3 smoke checks on port 8002.

### Functional Outcome

`tp4` and `tp4-ep` both reached health and passed text, one-image, two-image,
and JSON-constrained routine gates with no fatal-log signature. The EP log
shows 64 local experts out of 256 on each rank, so this is a real expert-shard
run, not an ignored command-line option.

`dp4-ep` also formed the four-rank expert group and selected
`AgRsAll2AllManager`, but all EngineCore processes failed before health with
`No available memory for the cache blocks`. At the required 100K context and
0.90 GPU-memory target, the most constrained rank calculated negative 1.6 GiB
available KV memory. This is a capacity rejection, not an RCCL failure. The
phase does not reduce the context contract merely to make this topology start.

### Fixed 128-Token Benchmark

| Scenario | TP4 | TP4 + EP | EP delta |
| --- | ---: | ---: | ---: |
| One 256-square image, C1 | 35.33 tok/s | 31.11 tok/s | -11.95% |
| Two 256-square images, C1 | 34.98 tok/s | 30.95 tok/s | -11.51% |
| Text, C1 | 36.04 tok/s | 31.64 tok/s | -12.19% |
| Text, C8 aggregate | 58.82 tok/s | 49.34 tok/s | -16.13% |

The two successful rows have comparable EngineCore initialization time
(396.05 s for TP4, 393.95 s for TP4 + EP), so the steady-state regression is
not explained by one-time compilation. EP also reduces configured KV capacity
from 860,714 tokens to 355,102 tokens, a 58.74% reduction.

## Current Re-screen: 2026-08-27

Current vLLM documentation continues to identify
`allgather_reducescatter` as the general EP backend. Its newer high-throughput
alternatives require DeepEP, FlashInfer/NVLink, or AITER-oriented paths. The
current Qwen ROCm recipe documents its verified AMD deployment on MI300-class
hardware, not gfx906. The local v0.28 source still selects the same AgRs path
for a portable EP deployment; it contains no new pre-CDNA MI50 EP fast path.

Accordingly, a v0.28 all-GPU retest would repeat the identical communication
class while retaining the same 100K KV constraint. It is not justified until a
new gfx906-capable EP kernel or a measurable reduction in the DP4 EP memory
footprint exists.

## Decision

Keep ordinary TP4 as the current 35B-A3B MoE reference. Do not promote either
EP geometry on this MI50 configuration: TP4 + EP is materially slower and
leaves much less KV capacity, while DP4 + EP cannot satisfy the 100K context
contract. The retained production service remains the separate Qwen3.5 9B
Router deployment; this phase makes no production model change.

Raw startup, gate, benchmark, final-state, and DP4 failure evidence is kept
under the Phase 90 result root on disk2 and intentionally excluded from Git.
