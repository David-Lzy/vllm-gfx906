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

Final branch counts, image digests, disk-space deltas, weight-retention paths,
and cache-removal totals are appended after the release and production canary
complete.
