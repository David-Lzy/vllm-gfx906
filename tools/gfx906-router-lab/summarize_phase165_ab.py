#!/usr/bin/env python3
"""Aggregate Phase 165 per-round summaries and evaluate promotion gates."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any


ARM_PATTERN = re.compile(
    r"^(official-rr|patched-rr|least-inflight|queue-aware)-"
    r"(text-c16|image1-c16|image2-c16|text-c32|image1-c32|image2-c32|phase1-c32)-"
    r"r([123])$"
)

FIXED_SCENARIOS = (
    "text-c16",
    "image1-c16",
    "image2-c16",
    "text-c32",
    "image1-c32",
    "image2-c32",
)
ALL_SCENARIOS = FIXED_SCENARIOS + ("phase1-c32",)


def median(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.median(present) if present else None


def percent_change(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline in (None, 0):
        return None
    return (candidate / baseline - 1.0) * 100.0


def collect(directory: Path) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for path in sorted(directory.glob("*-summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        match = ARM_PATTERN.match(data.get("arm", ""))
        if not match:
            continue
        arm, scenario, _round = match.groups()
        grouped.setdefault(arm, {}).setdefault(scenario, []).append(data)

    result: dict[str, dict[str, dict[str, Any]]] = {}
    fields = (
        "median_requests_per_second",
        "median_completion_tokens_per_second",
        "median_makespan_seconds",
        "median_latency_p95_seconds",
        "median_latency_p99_seconds",
        "median_idle_while_queued_ratio",
        "median_router_dispatch_p99_bound_seconds",
        "median_ttft_p50_bound_seconds",
        "median_ttft_p95_bound_seconds",
        "median_ttft_p99_bound_seconds",
        "median_router_cpu_percent_mean",
        "median_router_cpu_percent_p95",
        "median_router_rss_peak_bytes",
        "max_residual_router_local_inflight",
        "max_residual_worker_running",
        "max_residual_worker_waiting",
    )
    for arm, scenarios in grouped.items():
        result[arm] = {}
        for scenario, rows in scenarios.items():
            result[arm][scenario] = {
                field: median([row.get(field) for row in rows]) for field in fields
            }
            result[arm][scenario]["rounds"] = len(rows)
            result[arm][scenario]["failures"] = sum(row.get("total_failures", 0) for row in rows)
            result[arm][scenario]["backend_request_count_matches"] = all(
                row.get("all_backend_request_count_matches", False) for row in rows
            )
    return result


def evaluate(data: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    baseline = data.get("official-rr", {})
    candidate = data.get("queue-aware", {})
    patched = data.get("patched-rr", {})
    replay_base = baseline.get("phase1-c32", {})
    replay = candidate.get("phase1-c32", {})
    p95_delta = percent_change(
        replay.get("median_latency_p95_seconds"), replay_base.get("median_latency_p95_seconds")
    )
    p99_delta = percent_change(
        replay.get("median_latency_p99_seconds"), replay_base.get("median_latency_p99_seconds")
    )
    throughput_delta = percent_change(
        replay.get("median_requests_per_second"), replay_base.get("median_requests_per_second")
    )
    makespan_delta = percent_change(
        replay.get("median_makespan_seconds"), replay_base.get("median_makespan_seconds")
    )
    idle_base = replay_base.get("median_idle_while_queued_ratio")
    idle = replay.get("median_idle_while_queued_ratio")
    idle_reduction = None
    if idle is not None and idle_base not in (None, 0):
        idle_reduction = (1.0 - idle / idle_base) * 100.0
    fixed_deltas = [
        percent_change(
            candidate.get(scenario, {}).get("median_requests_per_second"),
            baseline.get(scenario, {}).get("median_requests_per_second"),
        )
        for scenario in FIXED_SCENARIOS
    ]
    fixed_present = [value for value in fixed_deltas if value is not None]
    queue_dispatch = median(
        [
            candidate.get(scenario, {}).get("median_router_dispatch_p99_bound_seconds")
            for scenario in ALL_SCENARIOS
        ]
    )
    patched_dispatch = median(
        [
            patched.get(scenario, {}).get("median_router_dispatch_p99_bound_seconds")
            for scenario in ALL_SCENARIOS
        ]
    )
    dispatch_overhead = (
        queue_dispatch - patched_dispatch
        if queue_dispatch is not None and patched_dispatch is not None
        else None
    )
    all_failures = sum(
        scenario.get("failures", 0)
        for scenarios in data.values()
        for scenario in scenarios.values()
    )
    backend_counts_match = all(
        scenario.get("backend_request_count_matches", False)
        for scenarios in data.values()
        for scenario in scenarios.values()
    )
    residual_values = [
        scenario.get(field)
        for scenarios in data.values()
        for scenario in scenarios.values()
        for field in ("max_residual_worker_running", "max_residual_worker_waiting")
    ]
    residual_values.extend(
        scenario.get("max_residual_router_local_inflight")
        for arm, scenarios in data.items()
        if arm != "official-rr"
        for scenario in scenarios.values()
    )
    gates = {
        "three_rounds_each": all(
            data.get(arm, {}).get(scenario, {}).get("rounds") == 3
            for arm in ("official-rr", "patched-rr", "least-inflight", "queue-aware")
            for scenario in ALL_SCENARIOS
        ),
        "zero_failures": all_failures == 0,
        "no_duplicate_backend_requests": backend_counts_match,
        "zero_residual": bool(residual_values)
        and all(value is not None and value <= 0 for value in residual_values),
        "idle_reduction": idle_reduction is not None and idle_reduction >= 50.0,
        "idle_absolute": idle is not None and idle <= 0.25,
        "phase1_tail": any(value is not None and value <= -10.0 for value in (p95_delta, p99_delta)),
        "phase1_throughput_or_makespan": (
            throughput_delta is not None
            and makespan_delta is not None
            and (throughput_delta >= 5.0 or makespan_delta <= -5.0)
        ),
        "fixed_regression": bool(fixed_present) and min(fixed_present) >= -3.0,
        "dispatch_overhead": dispatch_overhead is not None and dispatch_overhead <= 0.002,
    }
    return {
        "deltas_percent": {
            "phase1_p95": p95_delta,
            "phase1_p99": p99_delta,
            "phase1_throughput": throughput_delta,
            "phase1_makespan": makespan_delta,
            "phase1_idle_reduction": idle_reduction,
            "fixed_throughput": fixed_deltas,
        },
        "phase1_idle_absolute": idle,
        "dispatch_overhead_seconds": dispatch_overhead,
        "total_failures": all_failures,
        "gates": gates,
        "eligible_for_phase166": all(gates.values()),
    }


def render_markdown(data: dict[str, Any], evaluation: dict[str, Any]) -> str:
    lines = [
        "# Phase 165 Router A/B scorecard",
        "",
        "| Policy | Scenario | req/s | p95 s | p99 s | TTFT p95 bound s | idle while queued | failures |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ("official-rr", "patched-rr", "least-inflight", "queue-aware"):
        for scenario in ALL_SCENARIOS:
            row = data.get(arm, {}).get(scenario, {})
            fmt = lambda value: "n/a" if value is None else f"{value:.4f}"
            lines.append(
                f"| {arm} | {scenario} | {fmt(row.get('median_requests_per_second'))} "
                f"| {fmt(row.get('median_latency_p95_seconds'))} "
                f"| {fmt(row.get('median_latency_p99_seconds'))} "
                f"| {fmt(row.get('median_ttft_p95_bound_seconds'))} "
                f"| {fmt(row.get('median_idle_while_queued_ratio'))} "
                f"| {row.get('failures', 'n/a')} |"
            )
    lines.extend(("", "## Gates", ""))
    for gate, passed in evaluation["gates"].items():
        lines.append(f"- [{'x' if passed else ' '}] `{gate}`")
    lines.extend(
        (
            "",
            f"Phase 166 eligible: **{'yes' if evaluation['eligible_for_phase166'] else 'no'}**",
            "",
        )
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    data = collect(args.results)
    evaluation = evaluate(data)
    payload = {"results": data, "evaluation": evaluation}
    args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(data, evaluation), encoding="utf-8")
    print(json.dumps(evaluation, indent=2, sort_keys=True))
    return 0 if evaluation["eligible_for_phase166"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
