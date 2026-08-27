# Qwen3.5 Multimodal Prefix-Cache Transport Screen

**Status:** rejected for the gfx906 release composition.

We screened vLLM upstream commit
[`db92053e97`](https://github.com/vllm-project/vllm/commit/db92053e97),
the implementation proposed in [vLLM pull request 52041](https://github.com/vllm-project/vllm/pull/52041).
It avoids sending multimodal tensors to a worker when an already-cached prefix
contains the corresponding image features. The change preserves Qwen3.5
M-RoPE `image_grid_thw` metadata while omitting the cached `pixel_values`.

This is a workload-specific optimization: a fresh image request still must
send its pixel tensor, so it cannot improve ordinary cache-busted image
traffic. The local control validated a real cached prefix on Qwen3.5's Mamba
alignment geometry: prefix-cache hit rate reached 55.8% and multimodal-cache
hit rate reached 98.8% for repeated one/two 256-square images with a
block-aligned text prefix.

The backport is not safe on the current v0.27 gfx906 composition. Its narrow
unit test passed, but a patched GPU2 worker returned HTTP 500 for its first
multimodal request. A clean retry then failed to reach `/health` after model
load and `torch.compile`, exceeding the matched control startup window while
the `VLLM::EngineCore` remained stuck. Production GPU0/GPU1, the Router, and
port 8002 were never changed.

The source change is therefore not retained. Reopen only after rebasing onto
a vLLM revision containing the complete scheduler/output contract for this
feature, and only for a workload with sustained exact image-prefix reuse.
