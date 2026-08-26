# vLLM gfx906

An experimental vLLM fork and evidence archive for AMD gfx906 GPUs, including
Radeon VII, MI50, and MI60.

This repository exists to make old-but-capable CDNA1 and Vega20 hardware useful
for modern vLLM workloads. It is maintained as a focused engineering project,
not as a drop-in replacement for upstream vLLM or a general ROCm distribution.

## Scope and current status

- `main` is intentionally low-churn. It receives only reviewed, reproducible
  gfx906 work that is suitable as a public integration point.
- Performance and compatibility experiments live on focused `perf/`, `port/`,
  `backport/`, and `integration/` branches. Their benchmark evidence is kept so
  regressions can be investigated rather than rediscovered.
- The active local reference deployment uses an independently built gfx906
  runtime to serve Qwen3.5 9B AWQ multimodal requests. It is not a published
  production image from this repository.
- Qwen 27B, MoE, speculative decoding, KV-cache quantization, and newer
  runtime ports are experimental. A successful branch is not a support claim
  until it has a documented benchmark and release gate.

No model weights, Hugging Face caches, build caches, credentials, or production
Compose files are stored in this repository.

## Start here

1. Confirm that the host exposes `/dev/kfd` and `/dev/dri` and that ROCm can see
   the gfx906 GPU before attempting a build.
2. Clone this repository with submodules and choose an evidence branch only when
   its linked benchmark matches the workload you want to reproduce.
3. Build in a disposable container or isolated virtual environment. Do not use
   `--privileged`, mount an entire home directory, or rely on a floating
   `latest` image merely to test a model server.
4. Run a small text and multimodal smoke test before applying any performance
   setting to a real service.

The upstream vLLM documentation remains the source of truth for the OpenAI API,
supported model interfaces, and general serving behavior:
<https://docs.vllm.ai/>.

## Documentation and evidence

The maintained gfx906 engineering documentation is currently published on the
[`docs/gfx906-roadmap-v027`](https://github.com/David-Lzy/vllm-gfx906/tree/docs/gfx906-roadmap-v027/docs/gfx906)
branch while its individual records are prepared for selective integration.

Useful starting points:

- [gfx906 documentation index](https://github.com/David-Lzy/vllm-gfx906/blob/docs/gfx906-roadmap-v027/docs/gfx906/README.md)
- [v0.27 roadmap](https://github.com/David-Lzy/vllm-gfx906/blob/docs/gfx906-roadmap-v027/docs/gfx906/roadmap-v027.md)
- [compatibility matrix](https://github.com/David-Lzy/vllm-gfx906/blob/docs/gfx906-roadmap-v027/docs/gfx906/compatibility-matrix.md)
- [benchmark protocol](https://github.com/David-Lzy/vllm-gfx906/blob/docs/gfx906-roadmap-v027/docs/gfx906/benchmark-protocol.md)

## Branch policy

`main` is the public integration surface. Integration branches track a specific
upstream vLLM line. Feature branches carry one narrowly scoped experiment or
port and should contain their measurements and a rollback note. Evidence
branches are retained until their result has been incorporated into a stable
release record or explicitly archived.

When proposing a change, keep it small, include the exact test command and
hardware/runtime versions, and report both wins and regressions. Do not combine
unrelated kernel, routing, model, and documentation changes in one pull request.

## License

This fork retains the upstream vLLM license and notices. See [LICENSE](LICENSE)
for the applicable terms.
