# Patch ledger

This ledger prevents the v0.26 effort from becoming an unreviewed bulk merge.
Each feature is classified independently as retained, rewritten, replaced by
upstream, deferred, or dropped.

## Initial audit

- Baseline lineage: `gfx906/v0.23.1rc0.x`
- Integration base: upstream `v0.26.0`
- Initial merge-tree audit: 22 conflicting paths between the two lineages
- Rule: old branches are evidence and patch sources, not merge targets

| Patch area | Existing purpose | v0.26 starting point | Decision | Required evidence |
| --- | --- | --- | --- | --- |
| Build target and device detection | Compile and select gfx906 kernels | Upstream recognizes gfx906 in parts of CMake; container defaults still need review | pending | Clean image build and device smoke |
| gfx906 Triton toolchain | Provide usable Triton kernels on MI50 | Reconcile the validated gfx906 toolchain with v0.26 dependencies | pending | Kernel smoke and attention correctness |
| Classic AWQ kernels | Run Qwen AWQ weights efficiently | Audit v0.26 quantization dispatch before porting | pending | Qwen3.5 AWQ text and image parity |
| Attention fallback | Avoid unsupported newer-architecture paths | Prefer upstream path when valid; retain a narrow fallback | pending | Short, long, and chunked-prefill tests |
| Vision transformer path | Run Qwen multimodal encoder on gfx906 | Audit attention and processor changes in v0.26 | pending | 1/8/32/64 image and grid tests |
| Qwen3.5 multimodal integration | Serve the primary production model | Rebase behavior onto v0.26 model interfaces | pending | API, quality, and usage parity |
| DFlash | Speculative decode acceleration | Isolate from base release | deferred | At least 10% completion or 5% E2E gain |
| MoE TP | Run 35B-A3B across four GPUs | Start from BF16 TP4 without EP | deferred | Text, image, context, and RCCL stability |
| TurboQuant KV | Reduce KV capacity pressure | Port each cache format independently | deferred | 32K/64K/100K quality and load tests |
| FP8 KV | Optional KV compression | Software path only on MI50 | deferred | Demonstrated E2E benefit without quality loss |
| Expert parallelism | Distribute MoE experts | Test after ordinary TP4 | deferred | Stable communication and competitive throughput |

## Entry requirements

Every retained or rewritten patch must record:

1. Source commit or upstream issue.
2. Files and public behavior changed.
3. Why upstream v0.26 is insufficient on gfx906.
4. Unit, smoke, benchmark, and hardware evidence.
5. Removal criteria for a future upstream version.
