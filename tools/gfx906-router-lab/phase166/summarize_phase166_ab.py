#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Aggregate one Phase 166 Router A/B stage and evaluate its gates."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

ARMS = ("official-rr", "patched-rr", "least-inflight", "global-fifo")
QUICK_SCENARIOS = tuple(
    f"{request_class}-c{concurrency}"
    for concurrency in (16, 32, 40, 64)
    for request_class in ("text", "image1", "image2")
)
STAGE_SCENARIOS = {
    "quick": QUICK_SCENARIOS,
    "subset": ("phase1-subset-c40",),
    "full": ("phase1-full-c32",),
}
PATTERN = re.compile(
    r"^(official-rr|patched-rr|least-inflight|global-fifo)-(.+)-r([123])$"
)


def median(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.median(present) if present else None


def percent_change(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline in (None, 0):
        return None
    return (candidate / baseline - 1.0) * 100.0


def reduction(candidate: float | None, baseline: float | None) -> float | None:
    if baseline == 0:
        return 100.0 if candidate == 0 else 0.0 if candidate is not None else None
    change = percent_change(candidate, baseline)
    return -change if change is not None else None


def collect(directory: Path, scenarios: tuple[str, ...]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for path in sorted(directory.glob("*-summary.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        match = PATTERN.match(str(row.get("arm", "")))
        if not match:
            continue
        arm, scenario, _round = match.groups()
        if scenario not in scenarios:
            continue
        grouped.setdefault(arm, {}).setdefault(scenario, []).append(row)

    fields = (
        "median_requests_per_second",
        "median_completion_tokens_per_second",
        "median_makespan_seconds",
        "median_latency_p95_seconds",
        "median_latency_p99_seconds",
        "median_idle_while_queued_ratio",
        "median_worker_queue_seconds_total",
        "median_admission_backlog_with_unused_slot_ratio",
        "median_router_dispatch_p99_bound_seconds",
        "median_ttft_p95_bound_seconds",
        "median_router_cpu_percent_mean",
        "median_router_rss_peak_bytes",
        "max_residual_router_local_inflight",
        "max_residual_admission_slots",
        "max_residual_admission_queue",
        "max_residual_worker_running",
        "max_residual_worker_waiting",
        "max_admission_invariant_violations",
    )
    result: dict[str, Any] = {}
    for arm, arm_scenarios in grouped.items():
        result[arm] = {}
        for scenario, rows in arm_scenarios.items():
            summary = {
                field: median([row.get(field) for row in rows]) for field in fields
            }
            summary["rounds"] = len(rows)
            summary["failures"] = sum(row.get("total_failures", 0) for row in rows)
            summary["backend_request_count_matches"] = all(
                row.get("all_backend_request_count_matches", False) for row in rows
            )
            result[arm][scenario] = summary
    return result


def evaluate(stage: str, data: dict[str, Any]) -> dict[str, Any]:
    scenarios = STAGE_SCENARIOS[stage]
    required_arms = ARMS if stage != "full" else ("official-rr", "global-fifo")
    baseline = data.get("official-rr", {})
    candidate = data.get("global-fifo", {})
    patched = data.get("patched-rr", {})

    throughput_deltas = [
        percent_change(
            candidate.get(scenario, {}).get("median_requests_per_second"),
            baseline.get(scenario, {}).get("median_requests_per_second"),
        )
        for scenario in scenarios
    ]
    p95_deltas = [
        percent_change(
            candidate.get(scenario, {}).get("median_latency_p95_seconds"),
            baseline.get(scenario, {}).get("median_latency_p95_seconds"),
        )
        for scenario in scenarios
    ]
    p99_deltas = [
        percent_change(
            candidate.get(scenario, {}).get("median_latency_p99_seconds"),
            baseline.get(scenario, {}).get("median_latency_p99_seconds"),
        )
        for scenario in scenarios
    ]
    makespan_deltas = [
        percent_change(
            candidate.get(scenario, {}).get("median_makespan_seconds"),
            baseline.get(scenario, {}).get("median_makespan_seconds"),
        )
        for scenario in scenarios
    ]
    idle_reductions = [
        reduction(
            candidate.get(scenario, {}).get("median_idle_while_queued_ratio"),
            baseline.get(scenario, {}).get("median_idle_while_queued_ratio"),
        )
        for scenario in scenarios
    ]
    queue_reductions = [
        reduction(
            candidate.get(scenario, {}).get("median_worker_queue_seconds_total"),
            baseline.get(scenario, {}).get("median_worker_queue_seconds_total"),
        )
        for scenario in scenarios
    ]
    admission_unused = [
        candidate.get(scenario, {}).get(
            "median_admission_backlog_with_unused_slot_ratio"
        )
        for scenario in scenarios
    ]
    dispatch_overheads = []
    for scenario in scenarios:
        candidate_dispatch = candidate.get(scenario, {}).get(
            "median_router_dispatch_p99_bound_seconds"
        )
        patched_dispatch = patched.get(scenario, {}).get(
            "median_router_dispatch_p99_bound_seconds"
        )
        dispatch_overheads.append(
            candidate_dispatch - patched_dispatch
            if candidate_dispatch is not None and patched_dispatch is not None
            else None
        )

    rows = [
        data.get(arm, {}).get(scenario, {})
        for arm in required_arms
        for scenario in scenarios
    ]
    candidate_rows = [candidate.get(scenario, {}) for scenario in scenarios]
    worker_residual_fields = (
        "max_residual_worker_running",
        "max_residual_worker_waiting",
    )
    admission_residual_fields = (
        "max_residual_admission_slots",
        "max_residual_admission_queue",
    )
    gates: dict[str, bool] = {
        "three_rounds_each": all(row.get("rounds") == 3 for row in rows),
        "zero_failures": all(row.get("failures") == 0 for row in rows),
        "no_duplicate_backend_requests": all(
            row.get("backend_request_count_matches") is True for row in rows
        ),
        "zero_worker_residual": all(
            row.get(field) in (0, 0.0)
            for row in candidate_rows
            for field in worker_residual_fields
        ),
        "zero_admission_residual": all(
            row.get(field) in (0, 0.0)
            for row in candidate_rows
            for field in admission_residual_fields
        ),
        "zero_admission_invariants": all(
            row.get("max_admission_invariant_violations") in (None, 0, 0.0)
            for row in candidate_rows
        ),
        "admission_slots_not_idle": bool(admission_unused)
        and all(value is not None and value <= 0.01 for value in admission_unused),
    }

    if stage == "quick":
        c16_c32 = [
            value
            for scenario, value in zip(scenarios, throughput_deltas, strict=True)
            if scenario.endswith("c16") or scenario.endswith("c32")
        ]
        heavy_indices = [
            index
            for index, scenario in enumerate(scenarios)
            if scenario.endswith("c40") or scenario.endswith("c64")
        ]
        gates.update(
            {
                "fixed_c16_c32_regression": bool(c16_c32)
                and all(value is not None and value >= -3.0 for value in c16_c32),
                "idle_reduction": all(
                    idle_reductions[index] is not None
                    and idle_reductions[index] >= 80.0
                    for index in heavy_indices
                ),
                "idle_absolute": all(
                    candidate.get(scenarios[index], {}).get(
                        "median_idle_while_queued_ratio"
                    )
                    is not None
                    and candidate[scenarios[index]]["median_idle_while_queued_ratio"]
                    <= 0.10
                    for index in heavy_indices
                ),
                "worker_waiting_reduction": all(
                    queue_reductions[index] is not None
                    and queue_reductions[index] >= 50.0
                    for index in heavy_indices
                ),
                "dispatch_overhead": bool(dispatch_overheads)
                and all(
                    value is not None and value <= 0.002 for value in dispatch_overheads
                ),
            }
        )
    elif stage == "subset":
        gates.update(
            {
                "idle_reduction": idle_reductions[0] is not None
                and idle_reductions[0] >= 80.0,
                "idle_absolute": candidate.get(scenarios[0], {}).get(
                    "median_idle_while_queued_ratio"
                )
                is not None
                and candidate[scenarios[0]]["median_idle_while_queued_ratio"] <= 0.10,
                "worker_waiting_reduction": queue_reductions[0] is not None
                and queue_reductions[0] >= 50.0,
                "no_material_throughput_regression": throughput_deltas[0] is not None
                and throughput_deltas[0] >= -3.0,
            }
        )
    else:
        gates.update(
            {
                "tail_improvement": any(
                    value is not None and value <= -10.0
                    for value in (p95_deltas[0], p99_deltas[0])
                ),
                "throughput_or_makespan": (
                    throughput_deltas[0] is not None
                    and makespan_deltas[0] is not None
                    and (throughput_deltas[0] >= 5.0 or makespan_deltas[0] <= -5.0)
                ),
                "idle_reduction": idle_reductions[0] is not None
                and idle_reductions[0] >= 80.0,
                "idle_absolute": candidate.get(scenarios[0], {}).get(
                    "median_idle_while_queued_ratio"
                )
                is not None
                and candidate[scenarios[0]]["median_idle_while_queued_ratio"] <= 0.10,
                "worker_waiting_reduction": queue_reductions[0] is not None
                and queue_reductions[0] >= 50.0,
            }
        )

    return {
        "stage": stage,
        "deltas_percent": {
            "throughput": throughput_deltas,
            "p95": p95_deltas,
            "p99": p99_deltas,
            "makespan": makespan_deltas,
            "idle_reduction": idle_reductions,
            "worker_waiting_reduction": queue_reductions,
        },
        "admission_backlog_with_unused_slot_ratio": admission_unused,
        "dispatch_overhead_seconds": dispatch_overheads,
        "gates": gates,
        "eligible_for_next_stage": all(gates.values()),
    }


def render_markdown(
    stage: str, data: dict[str, Any], evaluation: dict[str, Any]
) -> str:
    scenarios = STAGE_SCENARIOS[stage]
    arms = ARMS if stage != "full" else ("official-rr", "global-fifo")
    lines = [
        f"# Phase 166 Router A/B scorecard: {stage}",
        "",
        (
            "| Policy | Scenario | req/s | p95 s | p99 s | idle while queued "
            "| worker queue-s | admission gap | failures |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in arms:
        for scenario in scenarios:
            row = data.get(arm, {}).get(scenario, {})

            def fmt(value: Any) -> str:
                return "n/a" if value is None else f"{value:.4f}"

            lines.append(
                f"| {arm} | {scenario} | {fmt(row.get('median_requests_per_second'))} "
                f"| {fmt(row.get('median_latency_p95_seconds'))} "
                f"| {fmt(row.get('median_latency_p99_seconds'))} "
                f"| {fmt(row.get('median_idle_while_queued_ratio'))} "
                f"| {fmt(row.get('median_worker_queue_seconds_total'))} "
                f"| {fmt(row.get('median_admission_backlog_with_unused_slot_ratio'))} "
                f"| {row.get('failures', 'n/a')} |"
            )
    lines.extend(("", "## Gates", ""))
    for gate, passed in evaluation["gates"].items():
        lines.append(f"- [{'x' if passed else ' '}] `{gate}`")
    lines.extend(
        (
            "",
            (
                "Eligible for next stage: "
                f"**{'yes' if evaluation['eligible_for_next_stage'] else 'no'}**"
            ),
            "",
        )
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--stage", choices=tuple(STAGE_SCENARIOS), required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    data = collect(args.results, STAGE_SCENARIOS[args.stage])
    evaluation = evaluate(args.stage, data)
    payload = {"results": data, "evaluation": evaluation}
    args.json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.markdown.write_text(
        render_markdown(args.stage, data, evaluation), encoding="utf-8"
    )
    print(json.dumps(evaluation, indent=2, sort_keys=True))
    return 0 if evaluation["eligible_for_next_stage"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
