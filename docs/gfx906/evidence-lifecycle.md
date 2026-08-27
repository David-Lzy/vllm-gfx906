# Evidence lifecycle

This fork keeps narrowly scoped experimental branches because each branch can
carry a real-hardware result, a rejected kernel path, or a compatibility
finding that prevents later work from repeating the same mistake. A large
branch count is not, by itself, evidence that branches are disposable.

## Durable documentation

`docs/gfx906-roadmap-v027` is the single integration line for public fork
documentation. It carries the root README and the public compatibility,
benchmark, release, and phase-result records. Experimental branches remain
separate until their outcome is represented in that documentation or in a
versioned release tag.

## Branch lifecycle

| Lifecycle | Meaning | Normal action |
| --- | --- | --- |
| `retain-active` | Default history or the current documentation integration line. | Keep reviewable and protected from accidental deletion. |
| `retain-evidence` | A source, benchmark, port, or historical-release record with retained value. | Keep until a public result and any required release tag make archival safe. |
| `tag-then-archive` | A fully summarized, inactive line that should first receive immutable provenance. | Propose a tag and archive plan for review. |
| `review-for-delete` | A fully summarized, unmerged branch with no worktree or open pull request. | Present an exact ref list and evidence links for explicit approval. |

The audit ledger intentionally has no `review-for-delete` rows while phase
worktrees or unsummarized evidence remain. No branch deletion is implicit in a
report, a phase completion, or GitHub's branch suggestions.

## Pull requests

The draft documentation pull request is the review point for this lifecycle.
Earlier stacked drafts remain historical until that documentation line is
merged with explicit approval. Only then may they be closed as superseded;
their branch heads remain intact until a separate cleanup review approves a
specific deletion batch.

## External comparisons

GitHub's `Compare & pull request` banner can appear when a branch has recently
been pushed. For this fork, a banner targeting an upstream or mobydick branch
often compares unrelated, long-lived histories. It is not a request to submit
the whole branch upstream.

An external contribution must instead be a small patch rebased on the intended
upstream base, with independent correctness and gfx906 benchmark evidence.

## Ledger

The current non-destructive snapshot is
[`branch-ledger-20260826.json`](branch-ledger-20260826.json). It contains one
record per remote head, including its tip, last update, evidence class,
published worktree state, open pull-request state, merge base, and proposed
lifecycle. Local filesystem paths are intentionally not published.

Regenerate a later snapshot from a clean checkout with:

```bash
python3 tools/gfx906_branch_ledger.py \
  --github-repo David-Lzy/vllm-gfx906 \
  --as-of YYYY-MM-DD \
  --output docs/gfx906/branch-ledger-YYYYMMDD.json
```
