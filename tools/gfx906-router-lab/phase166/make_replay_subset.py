#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Freeze a deterministic 40-request Phase1 replay subset by image/payload strata."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

STRATA = (
    ("image-0", 0, 0, 10),
    ("image-3-8", 3, 8, 2),
    ("image-9-16", 9, 16, 6),
    ("image-17-24", 17, 24, 5),
    ("image-25-32", 25, 32, 4),
    ("image-33-40", 33, 40, 5),
    ("image-41-48", 41, 48, 8),
)


def evenly_spaced(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(rows) <= count:
        return rows
    if count == 1:
        return [rows[len(rows) // 2]]
    indices = [round(index * (len(rows) - 1) / (count - 1)) for index in range(count)]
    return [rows[index] for index in indices]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.input.resolve().parent
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected: list[dict[str, Any]] = []
    stratum_counts: dict[str, int] = {}
    for name, minimum, maximum, quota in STRATA:
        candidates = sorted(
            (
                row
                for row in rows
                if minimum <= int(row.get("image_count", -1)) <= maximum
            ),
            key=lambda row: (int(row["payload_bytes"]), int(row["request_index"])),
        )
        if len(candidates) < quota:
            raise SystemExit(
                f"stratum {name} has {len(candidates)} records, needs {quota}"
            )
        chosen = evenly_spaced(candidates, quota)
        selected.extend(chosen)
        stratum_counts[name] = len(chosen)

    if len(selected) != 40 or len({row["request_index"] for row in selected}) != 40:
        raise SystemExit("subset must contain exactly 40 unique requests")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    normalized: list[dict[str, Any]] = []
    for row in sorted(selected, key=lambda item: int(item["request_index"])):
        payload = (source_root / str(row["payload_relpath"])).resolve()
        if not payload.is_relative_to(source_root) or not payload.is_file():
            raise SystemExit(f"invalid payload path: {payload}")
        item = dict(row)
        item["payload_relpath"] = os.path.relpath(payload, args.output.parent.resolve())
        item["phase166_subset_stratum"] = next(
            name
            for name, minimum, maximum, _ in STRATA
            if minimum <= int(item["image_count"]) <= maximum
        )
        normalized.append(item)

    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in normalized),
        encoding="utf-8",
    )
    summary = {
        "output": str(args.output.resolve()),
        "requests": len(normalized),
        "strata": stratum_counts,
        "stages": Counter(str(row.get("stage", "unknown")) for row in normalized),
        "image_count_min": min(int(row["image_count"]) for row in normalized),
        "image_count_max": max(int(row["image_count"]) for row in normalized),
        "payload_bytes_min": min(int(row["payload_bytes"]) for row in normalized),
        "payload_bytes_max": max(int(row["payload_bytes"]) for row in normalized),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
