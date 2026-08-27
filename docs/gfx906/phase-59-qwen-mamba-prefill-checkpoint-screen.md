# Qwen Mamba Prefill-Checkpoint Screen

**Status:** not applicable to the current Qwen release composition.

We screened upstream vLLM change
[`9eb9d9d395`](https://github.com/vllm-project/vllm/commit/9eb9d9d3953959695108600c8ed33d36bc6a1e5f)
([PR 52789](https://github.com/vllm-project/vllm/pull/52789)), which reports
9--25% TTFT improvement from internal prefill checkpoints for Mamba prefix
caching.

The generic scheduler and Mamba cache-manager changes are gated by
`MambaSpec.num_prefill_checkpoint_blocks`. In the upstream implementation,
only the Kimi K3 FlashKDA backend sets that field to a nonzero value. Qwen3.5
and Qwen3.8 retain the default of zero, so they do not enter the new checkpoint
path.

The commit also conflicts with the fork's older KDA and KV-cache contracts.
Cherry-picking it produced conflicts in FlashKDA registration, Kimi metadata,
prefix-cache tests, and `kv_cache_interface.py`. A partial port would add
maintenance risk without activating Qwen behavior, so no image or GPU test was
justified. The screen leaves production and the v0.27 candidate unchanged.
