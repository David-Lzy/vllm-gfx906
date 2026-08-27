# Qwen3.8 Hybrid Mamba State-Copy Tiling Screen

**Status:** rejected before build; not a safe v0.27 gfx906 backport.

We screened vLLM upstream commit
[`fac808b36f`](https://github.com/vllm-project/vllm/commit/fac808b36f502d0d992509a187dca94c68b0360a),
which uses a 3D Triton launch to divide each large temporal Mamba-state copy
across 16 CTAs. Its intended path is a hybrid model using speculative decode,
so Qwen3.8 MTP would satisfy the feature predicates.

It is not an isolated patch for this fork. The commit depends on a newer
Mamba-worker structure and does not apply to the v0.27 gfx906 worker. More
importantly, matched local evidence shows that Qwen3.8 MTP1 is 29.9% slower
than the repaired no-MTP path, while the relevant GDN component is only 0.55%
of a warmed 32K decode's GPU time. Attention, GPTQ W4A16, and communication
are materially larger costs.

We therefore did not build an image, download a model, or run a GPU benchmark.
A manual port cannot plausibly recover the required end-to-end throughput and
would create a high-maintenance divergence. The experiment can be reopened
only after a trace shows state-copy/GDN is a material cost on the intended
workload.
