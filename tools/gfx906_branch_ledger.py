#!/usr/bin/env python3
"""Generate a non-destructive lifecycle ledger for a gfx906 fork.

The ledger intentionally records branch purpose and retention policy rather
than deleting refs. It omits local worktree paths so the JSON can be published
with the repository documentation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

PullRequests = dict[str, list[dict[str, object]]]


def run(command: list[str], repo: Path) -> str:
    completed = subprocess.run(
        command,
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout


def try_run(command: list[str], repo: Path) -> str | None:
    try:
        return run(command, repo)
    except subprocess.CalledProcessError:
        return None


def worktree_branches(repo: Path) -> set[str]:
    branches: set[str] = set()
    for line in run(["git", "worktree", "list", "--porcelain"], repo).splitlines():
        prefix = "branch refs/heads/"
        if line.startswith(prefix):
            branches.add(line.removeprefix(prefix))
    return branches


def open_pull_requests(repo: Path, github_repo: str | None) -> PullRequests:
    if not github_repo:
        return {}

    output = try_run(
        [
            "gh",
            "pr",
            "list",
            "--repo",
            github_repo,
            "--state",
            "open",
            "--limit",
            "100",
            "--json",
            "number,headRefName,baseRefName,isDraft,title,url",
        ],
        repo,
    )
    if output is None:
        raise RuntimeError(
            "Could not query open pull requests. Retry with authenticated gh or "
            "omit --github-repo to generate a Git-only ledger."
        )

    by_head: PullRequests = {}
    for pull_request in json.loads(output):
        head = pull_request.pop("headRefName")
        pull_request["base"] = pull_request.pop("baseRefName")
        by_head.setdefault(head, []).append(pull_request)
    return by_head


def evidence_class(branch: str, default_branch: str) -> str:
    if branch == default_branch:
        return "default-release-history"
    if branch == "docs/gfx906-roadmap-v027":
        return "canonical-documentation-integration"
    if branch.startswith("agent/"):
        return "bootstrap-documentation-evidence"
    if branch.startswith("bench/"):
        return "baseline-harness-evidence"
    if branch.startswith("gfx906/v"):
        return "historical-release-provenance"
    if branch.startswith("integration/"):
        return "upstream-integration-evidence"
    if branch.startswith("port/"):
        return "compatibility-port-evidence"
    if branch.startswith("backport/"):
        return "selected-backport-evidence"
    if branch.startswith("perf/"):
        return "performance-or-benchmark-evidence"
    if branch.startswith("feat/") or branch.startswith("fix/"):
        return "retained-source-change-evidence"
    if branch.startswith("research/"):
        return "research-evidence"
    return "unclassified-retained-evidence"


def lifecycle(branch: str, default_branch: str) -> str:
    if branch in {default_branch, "docs/gfx906-roadmap-v027"}:
        return "retain-active"
    return "retain-evidence"


def ancestry(repo: Path, default_ref: str, ref: str) -> dict[str, object]:
    merge_base = try_run(["git", "merge-base", default_ref, ref], repo)
    counts = try_run(
        ["git", "rev-list", "--left-right", "--count", f"{default_ref}...{ref}"],
        repo,
    )
    if counts is None:
        ahead = behind = None
    else:
        behind_text, ahead_text = counts.split()
        behind, ahead = int(behind_text), int(ahead_text)
    return {
        "merge_base": merge_base.strip() if merge_base else None,
        "ahead_of_default": ahead,
        "behind_default": behind,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--default-branch", default="main")
    parser.add_argument(
        "--github-repo", help="Optional owner/repository for open PR metadata."
    )
    parser.add_argument(
        "--as-of", help="Audit date in YYYY-MM-DD form; defaults to UTC today."
    )
    parser.add_argument(
        "--output", type=Path, help="Write JSON here instead of stdout."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    remote_prefix = f"{args.remote}/"
    default_ref = f"{args.remote}/{args.default_branch}"
    worktrees = worktree_branches(repo)
    pulls = open_pull_requests(repo, args.github_repo)
    format_string = (
        "%(refname:short)\t%(objectname)\t%(committerdate:iso-strict)\t"
        "%(subject)"
    )
    ref_command = [
        "git",
        "for-each-ref",
        f"refs/remotes/{args.remote}",
        f"--format={format_string}",
    ]
    entries: list[dict[str, object]] = []

    for line in run(ref_command, repo).splitlines():
        remote_ref, tip, updated_at, subject = line.split("\t", maxsplit=3)
        if remote_ref in {args.remote, f"{args.remote}/HEAD"}:
            continue
        if not remote_ref.startswith(remote_prefix):
            continue
        branch = remote_ref.removeprefix(remote_prefix)
        entry: dict[str, object] = {
            "ref": branch,
            "tip": tip,
            "last_update": updated_at,
            "subject": subject,
            "evidence_class": evidence_class(branch, args.default_branch),
            "worktree": "attached" if branch in worktrees else "not-attached",
            "open_pull_requests": pulls.get(branch, []),
            "proposed_lifecycle": lifecycle(branch, args.default_branch),
            "evidence_reference": "docs/gfx906/README.md",
        }
        entry.update(ancestry(repo, default_ref, remote_ref))
        entries.append(entry)

    audit_date = args.as_of or dt.datetime.now(dt.timezone.utc).date().isoformat()
    summary: dict[str, int] = {}
    for entry in entries:
        key = str(entry["proposed_lifecycle"])
        summary[key] = summary.get(key, 0) + 1
    payload = {
        "schema_version": 1,
        "audit_date": audit_date,
        "remote": args.remote,
        "default_branch": args.default_branch,
        "scope": (
            "non-destructive lifecycle audit; local worktree paths are "
            "intentionally omitted"
        ),
        "summary": {"remote_heads": len(entries), "lifecycle_counts": summary},
        "entries": entries,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
