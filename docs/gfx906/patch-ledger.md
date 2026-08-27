# v0.28 gfx906 patch ledger

The release starts at upstream vLLM `v0.28.0` and keeps each gfx906 delta
independently attributable.

| Area | Source | Release decision | Default |
| --- | --- | --- | --- |
| Legacy gfx906 QGEMM | `6bd2e79477` | retained | build-time enabled |
| Deferred gfx906 device probe | `c761e656f5` | retained | enabled |
| `gfx906_gptq` CLI backend | `4fa5416905` | retained | explicit in examples |
| gfx906 output-head fast path | `96e8c8a425` | retained | enabled when eligible |
| W4A16 FP32 accumulation | `7d91f7de16` | retained | enabled when eligible |
| Qwen SplitKV restore | `a43e219209` | retained | off |
| Packed-INT8 GPTQ fix | `e7593a1689` | retained | checkpoint dependent |
| Packed-INT8 conversion tool | `d46b03a172` | retained | manual tool |
| Qwen3.6 fused QK/MRoPE/gate | `56ed1e5ec1`, `b6ac45dc84` | retained | off on ROCm |
| SplitKV-29 controls | `bf56406197` | retained | off |
| GDN output norm | `bd84b5192c` | retained | off |
| Qwen3 VL truncation guard | `e903d47981`, `32d469cfe7` | retained | enabled |
| Triton pointer intersection | `c25477975a` | retained in rebuilt Triton | enabled |

Rejected experiments remain documentation or archive evidence only. The public
image does not contain the rejected custom attention kernels, cost-aware Router,
CPU pinning policy, automatic Qwen3.6 fusion, MTP/DFlash defaults, or expert
parallel defaults.
