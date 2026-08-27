# Repository cleanup record, 2026-08-28

This release replaces the experimental branch farm with one reviewed `main`
branch and annotated archive tags. Before deletion, all 137 local experiment
heads and 113 worktrees were captured in a verified Git bundle. The two dirty
worktrees were exported as binary patches, and the bundle plus checksums are
attached to the GitHub release.

The cleanup preserves source history through the release bundle and four
annotated tags: v0.23 production rollback, retained v0.27, v0.28 integration,
and the Qwen3.8 packed-INT8/SplitKV-29 frontier. Model weights, compiler output,
caches, credentials, and machine-specific deployment files are not stored in
Git or release assets.

## Archived recovery material

- Full-reference bundle: 259,702,083 bytes, SHA256
  `b72dc1dd7c6c76c7cea417f9c738e98978dd47765be987c84ca92b4478d8ff5c`.
- Compact experiment evidence: 33,508,592 bytes, SHA256
  `4dce3add171293259fb361d288dbef2d822b481555d233f9cfd1e0d5d1e85e37`.
- Both dirty worktrees were exported as patches before deletion.
- The release assets are available from
  [`v0.28.0-gfx906.1`](https://github.com/David-Lzy/vllm-gfx906/releases/tag/v0.28.0-gfx906.1).

## Repository result

The cleanup reduced 137 experiment heads and 113 worktrees to one local
`main` branch, one canonical worktree, and one branch on the fork (`main`).
There are no open pull requests. The only retained workflow is `pre-commit`,
and its latest `main` run passed. `main` requires a pull request and the
`pre-commit` check, rejects force-push and deletion, and automatically deletes
merged branches.

## Storage result

The vLLM model set was reduced to the production Qwen3.5 9B AWQ snapshot,
Qwen3.8 27B AWQ, and the retained Qwen3.8 packed-INT8 derivative. Independent
embedding and OCR service caches were not treated as vLLM weights. Phase HF,
vLLM, torch, Triton, profiler, duplicate-model, and obsolete build trees were
removed after their compact evidence was archived.

Before cleanup, disk2 had 81,587,912,704 bytes available. After cleanup it had
338,614,611,968 bytes available, an increase of 257,026,699,264 bytes (about
239.4 GiB). The cleanup removed the old v0.23 runtime image, 24 Phase image
tags and their unshared layers, the obsolete production compiler cache, and
all Docker build cache. Docker build cache fell from 42.98 GB to zero. The only
retained vLLM runtime images are the versioned v0.28 release image and the
digest-pinned Router; the final model compiler cache is about 1.3 GiB.

The local recovery archive was reduced to about 306 MiB by deleting a duplicate
pre-tag bundle and an accidentally retained 930 MiB pre-commit environment.
The public checksum asset was regenerated with relative names only. Independent
containers, images, and caches belonging to other services were not removed.
