# gfx906 release benchmarks

These benchmarks support hardware validation for the gfx906 maintenance
branch. They use the OpenAI-compatible HTTP API and do not import vLLM, model
code, or GPU libraries.

## Production baseline

`production_baseline.py` fixes the core release workloads described in
`docs/gfx906/benchmark-protocol.md`:

- text, 16 requests at concurrency 8
- 8 images, 16 requests at concurrency 8
- 32 images, 8 requests at concurrency 4
- 64 images, 4 requests at concurrency 4
- one 4096 x 4096 grid at concurrency 1 and 4
- three JSON-schema constrained image requests

The runner generates deterministic PNG inputs in memory. Unique scenarios
change image bytes per request; reuse scenarios intentionally submit identical
bytes. Results are appended to JSONL and summarized in Markdown.

Example:

```bash
.venv/bin/python benchmarks/gfx906/production_baseline.py \
  --candidate current-gfx906 \
  --base-url http://127.0.0.1:8002/v1 \
  --model example/model \
  --jsonl /path/to/results.jsonl \
  --markdown /path/to/summary.md \
  --metrics-url worker0=http://127.0.0.1:18000/metrics
```

Pass one `--metrics-url NAME=URL` option for each worker and, when available,
the Router. Use repeated `--scenario NAME` options for a focused rerun. Raw
output paths should be outside the repository.
