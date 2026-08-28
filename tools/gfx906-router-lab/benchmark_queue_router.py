#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark one Router policy while sampling Router and vLLM metrics."""

from __future__ import annotations

import argparse
import asyncio
import binascii
import concurrent.futures
import contextlib
import hashlib
import json
import math
import os
import statistics
import struct
import subprocess
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import regex as re

SAMPLE_METRICS = {
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
}

WORKER_HISTOGRAM_METRICS = {
    "vllm:time_to_first_token_seconds",
}

ROUTER_SAMPLE_METRICS = {
    "vllm_router_policy_decisions_total",
    "vllm_router_processed_requests_total",
    "vllm_router_queue_policy_choices_total",
    "vllm_router_queue_fallback_total",
    "vllm_router_dispatch_duration_seconds_bucket",
    "vllm_router_dispatch_duration_seconds_sum",
    "vllm_router_dispatch_duration_seconds_count",
    "vllm_router_worker_request_duration_seconds_bucket",
    "vllm_router_worker_request_duration_seconds_sum",
    "vllm_router_worker_request_duration_seconds_count",
    "vllm_router_worker_local_inflight",
    "vllm_router_worker_observed_running",
    "vllm_router_worker_observed_waiting",
    "vllm_router_worker_effective_depth",
    "vllm_router_worker_telemetry_age_seconds",
}

DEFAULT_MODEL = "cyankiwi/Qwen3.5-9B-AWQ-4bit"


@dataclass(frozen=True)
class WorkItem:
    request_id: str
    request_class: str
    body: dict[str, Any]
    wire_body: bytes | None = None


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile_value
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def median_present(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.median(present) if present else None


def max_present(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def png_data_url(seed: int, size: int = 256) -> str:
    """Create a deterministic, cache-distinct RGB PNG without third-party deps."""
    rows = bytearray()
    for y in range(size):
        rows.append(0)
        for x in range(size):
            rows.extend(
                (
                    (x + seed * 17) % 256,
                    (y * 3 + seed * 29) % 256,
                    ((x // 16) ^ (y // 16) ^ seed) % 256,
                )
            )

    def chunk(kind: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", checksum)
        )

    data = b"\x89PNG\r\n\x1a\n"
    data += chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
    data += chunk(b"IDAT", zlib.compress(bytes(rows), level=6))
    data += chunk(b"IEND", b"")
    encoded = binascii.b2a_base64(data, newline=False).decode("ascii")
    return "data:image/png;base64," + encoded


def fixed_workload(
    model: str,
    count: int,
    max_tokens: int,
    fixed_class: str,
) -> list[WorkItem]:
    items: list[WorkItem] = []
    classes = ("text", "image1", "image2")
    for index in range(count):
        request_class = (
            classes[index % len(classes)] if fixed_class == "mixed" else fixed_class
        )
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Reply in one concise sentence. Describe the supplied visual "
                    f"pattern if present. Benchmark request {index:04d}."
                ),
            }
        ]
        image_count = 0 if request_class == "text" else int(request_class[-1])
        for image_index in range(image_count):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": png_data_url(index * 2 + image_index)},
                }
            )
        items.append(
            WorkItem(
                request_id=f"fixed-{index:04d}",
                request_class=request_class,
                body={
                    "model": model,
                    "messages": [{"role": "user", "content": content}],
                    "temperature": 0,
                    "max_tokens": max_tokens,
                    "stream": False,
                },
            )
        )
    return items


def replay_workload(path: Path, model: str | None) -> list[WorkItem]:
    items: list[WorkItem] = []
    replay_root = path.resolve().parent
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        wire_body: bytes | None = None
        if "payload_relpath" in record:
            payload_path = (replay_root / str(record["payload_relpath"])).resolve()
            if not payload_path.is_relative_to(replay_root):
                raise ValueError(
                    f"{path}:{line_number}: payload path escapes replay root"
                )
            payload_bytes = payload_path.read_bytes()
            expected_sha = record.get("payload_sha256")
            actual_sha = hashlib.sha256(payload_bytes).hexdigest()
            if expected_sha and actual_sha != expected_sha:
                raise ValueError(
                    f"{path}:{line_number}: payload SHA256 mismatch for {payload_path}"
                )
            body = json.loads(payload_bytes)
            wire_body = payload_bytes
        else:
            body = record.get("body", record)
        if not isinstance(body, dict):
            raise ValueError(f"{path}:{line_number}: body is not an object")
        body = dict(body)
        if body.get("stream") is True:
            raise ValueError(f"{path}:{line_number}: streaming replay is not supported")
        if model:
            body["model"] = model
            wire_body = None
        request_id = str(
            record.get(
                "id", f"replay-{int(record.get('request_index', line_number)):04d}"
            )
        )
        request_class = str(record.get("class", record.get("stage", "phase1")))
        items.append(WorkItem(request_id, request_class, body, wire_body))
    if not items:
        raise ValueError(f"replay file is empty: {path}")
    return items


def parse_labels(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    labels: dict[str, str] = {}
    for match in re.finditer(r'(\w+)="((?:\\.|[^"\\])*)"', raw):
        labels[match.group(1)] = bytes(match.group(2), "utf-8").decode("unicode_escape")
    return labels


def parse_prometheus(text: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    pattern = re.compile(r"^([^\s{]+)(?:\{(.*)\})?\s+([^\s]+)$")
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = pattern.match(line)
        if not match:
            continue
        try:
            value = float(match.group(3))
        except ValueError:
            continue
        samples.append(
            {
                "name": match.group(1),
                "labels": parse_labels(match.group(2)),
                "value": value,
            }
        )
    return samples


def metric_value(samples: list[dict[str, Any]], name: str) -> float | None:
    values = [sample["value"] for sample in samples if sample["name"] == name]
    return sum(values) if values else None


def fetch_text(url: str, timeout: float) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "gfx906-router-lab/1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def container_pid(name: str | None) -> int | None:
    if not name:
        return None
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Pid}}", name],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    pid = int(result.stdout.strip())
    if pid <= 0:
        raise RuntimeError(f"Router container is not running: {name}")
    return pid


def process_sample(pid: int | None) -> dict[str, float] | None:
    if pid is None:
        return None
    stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    fields = stat_text[stat_text.rfind(")") + 2 :].split()
    ticks = float(fields[11]) + float(fields[12])
    rss_kib = 0.0
    for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            rss_kib = float(line.split()[1])
            break
    return {
        "cpu_seconds": ticks / os.sysconf("SC_CLK_TCK"),
        "rss_bytes": rss_kib * 1024.0,
    }


def post_json(url: str, body: bytes, timeout: float) -> tuple[int, bytes]:
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "User-Agent": "gfx906-router-lab/1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def response_summary(payload: bytes) -> tuple[int, int, bool, str]:
    try:
        decoded = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return 0, 0, False, hashlib.sha256(payload).hexdigest()
    usage = decoded.get("usage") or {}
    completion_tokens = int(usage.get("completion_tokens") or 0)
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    content = ""
    choices = decoded.get("choices") or []
    if choices:
        content = str((choices[0].get("message") or {}).get("content") or "")
    return (
        completion_tokens,
        prompt_tokens,
        bool(content.strip()),
        hashlib.sha256(payload).hexdigest(),
    )


async def fetch_metrics(
    executor: concurrent.futures.Executor,
    url: str,
    timeout: float,
) -> list[dict[str, Any]] | None:
    loop = asyncio.get_running_loop()
    try:
        text = await loop.run_in_executor(executor, fetch_text, url, timeout)
    except Exception:
        return None
    return parse_prometheus(text)


async def sample_metrics(
    stop: asyncio.Event,
    executor: concurrent.futures.Executor,
    router_metrics_url: str,
    worker_metrics_urls: list[str],
    router_pid: int | None,
    interval: float,
    sink: list[dict[str, Any]],
) -> None:
    while True:
        started = time.monotonic()
        router_task = fetch_metrics(executor, router_metrics_url, min(interval, 1.0))
        worker_tasks = [
            fetch_metrics(executor, url, min(interval, 1.0))
            for url in worker_metrics_urls
        ]
        router_samples, *worker_samples = await asyncio.gather(
            router_task, *worker_tasks
        )
        if router_samples is not None:
            router_samples = [
                sample
                for sample in router_samples
                if sample["name"] in ROUTER_SAMPLE_METRICS
            ]
        workers: list[dict[str, Any]] = []
        for index, samples in enumerate(worker_samples):
            values: dict[str, Any] = {"worker": index, "available": samples is not None}
            if samples is not None:
                for name in SAMPLE_METRICS:
                    values[name] = metric_value(samples, name)
                values["histograms"] = [
                    sample
                    for sample in samples
                    if any(
                        sample["name"].startswith(name + "_")
                        for name in WORKER_HISTOGRAM_METRICS
                    )
                ]
            workers.append(values)
        sink.append(
            {
                "monotonic": started,
                "wall_time": time.time(),
                "workers": workers,
                "router": router_samples,
                "router_process": process_sample(router_pid),
            }
        )
        if stop.is_set():
            return
        remaining = interval - (time.monotonic() - started)
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=max(0.01, remaining))


async def issue_request(
    item: WorkItem,
    endpoint: str,
    timeout: float,
    semaphore: asyncio.Semaphore,
    executor: concurrent.futures.Executor,
    round_number: int,
) -> dict[str, Any]:
    encoded = item.wire_body or json.dumps(
        item.body, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    async with semaphore:
        started = time.monotonic()
        try:
            loop = asyncio.get_running_loop()
            status, response = await loop.run_in_executor(
                executor, post_json, endpoint, encoded, timeout
            )
            error = None
        except Exception as exc:
            status, response, error = 0, b"", f"{type(exc).__name__}: {exc}"
        latency = time.monotonic() - started
    completion_tokens, prompt_tokens, nonempty, response_sha = response_summary(
        response
    )
    return {
        "round": round_number,
        "request_id": item.request_id,
        "class": item.request_class,
        "status": status,
        "latency_seconds": latency,
        "payload_bytes": len(encoded),
        "response_bytes": len(response),
        "response_sha256": response_sha,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "nonempty": nonempty,
        "error": error,
    }


def counter_delta(
    first: list[dict[str, Any]] | None,
    last: list[dict[str, Any]] | None,
    name: str,
    label_key: str | None = None,
) -> dict[str, float] | float | None:
    if first is None or last is None:
        return None

    def collect(samples: list[dict[str, Any]]) -> dict[str, float]:
        values: dict[str, float] = {}
        for sample in samples:
            if sample["name"] != name:
                continue
            key = sample["labels"].get(label_key, "all") if label_key else "all"
            values[key] = values.get(key, 0.0) + sample["value"]
        return values

    before = collect(first)
    after = collect(last)
    delta = {
        key: after.get(key, 0.0) - before.get(key, 0.0)
        for key in set(before) | set(after)
    }
    return delta if label_key else delta.get("all", 0.0)


def histogram_quantile_bound(
    first: list[dict[str, Any]] | None,
    last: list[dict[str, Any]] | None,
    name: str,
    quantile: float,
) -> float | None:
    if first is None or last is None:
        return None

    def buckets(samples: list[dict[str, Any]]) -> dict[float, float]:
        result: dict[float, float] = {}
        for sample in samples:
            if sample["name"] != name + "_bucket":
                continue
            raw = sample["labels"].get("le")
            if raw is None:
                continue
            bound = math.inf if raw == "+Inf" else float(raw)
            result[bound] = result.get(bound, 0.0) + sample["value"]
        return result

    before = buckets(first)
    after = buckets(last)
    deltas = {
        bound: after.get(bound, 0.0) - before.get(bound, 0.0)
        for bound in set(before) | set(after)
    }
    total = deltas.get(math.inf, 0.0)
    if total <= 0:
        return None
    target = total * quantile
    for bound in sorted(deltas):
        if deltas[bound] >= target:
            return bound
    return None


def histogram_quantile_by_label(
    first: list[dict[str, Any]] | None,
    last: list[dict[str, Any]] | None,
    name: str,
    quantile: float,
    label_key: str,
) -> dict[str, float] | None:
    if first is None or last is None:
        return None

    labels = {
        sample["labels"].get(label_key, "unknown")
        for sample in first + last
        if sample["name"] == name + "_bucket"
    }
    result: dict[str, float] = {}
    for label in labels:
        before = [
            sample for sample in first if sample["labels"].get(label_key) == label
        ]
        after = [sample for sample in last if sample["labels"].get(label_key) == label]
        bound = histogram_quantile_bound(before, after, name, quantile)
        if bound is not None:
            result[label] = bound
    return result or None


def worker_histogram_samples(sample: dict[str, Any], name: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for worker in sample["workers"]:
        selected.extend(
            metric
            for metric in worker.get("histograms", [])
            if metric["name"].startswith(name + "_")
        )
    return selected


def summarize_samples(
    samples: list[dict[str, Any]], max_num_seqs: int
) -> dict[str, Any]:
    usable = [
        sample
        for sample in samples
        if all(worker["available"] for worker in sample["workers"])
    ]
    idle_while_queued = 0
    queue_seconds = [0.0, 0.0, 0.0, 0.0]
    max_running = [0.0, 0.0, 0.0, 0.0]
    max_waiting = [0.0, 0.0, 0.0, 0.0]
    for index, sample in enumerate(usable):
        running = [
            float(worker.get("vllm:num_requests_running") or 0.0)
            for worker in sample["workers"]
        ]
        waiting = [
            float(worker.get("vllm:num_requests_waiting") or 0.0)
            for worker in sample["workers"]
        ]
        if any(value > 0 for value in waiting) and any(
            running[i] < max_num_seqs and waiting[i] == 0 for i in range(len(running))
        ):
            idle_while_queued += 1
        for worker_index in range(4):
            max_running[worker_index] = max(
                max_running[worker_index], running[worker_index]
            )
            max_waiting[worker_index] = max(
                max_waiting[worker_index], waiting[worker_index]
            )
            if index:
                elapsed = sample["monotonic"] - usable[index - 1]["monotonic"]
                queue_seconds[worker_index] += waiting[worker_index] * elapsed
    router_first = usable[0]["router"] if usable else None
    router_last = usable[-1]["router"] if usable else None
    ttft_first = (
        worker_histogram_samples(usable[0], "vllm:time_to_first_token_seconds")
        if usable
        else None
    )
    ttft_last = (
        worker_histogram_samples(usable[-1], "vllm:time_to_first_token_seconds")
        if usable
        else None
    )
    router_cpu_percent: list[float] = []
    router_rss_bytes: list[float] = []
    for previous, current in zip(samples, samples[1:]):
        before = previous.get("router_process")
        after = current.get("router_process")
        elapsed = current["monotonic"] - previous["monotonic"]
        if before and after and elapsed > 0:
            router_cpu_percent.append(
                max(0.0, after["cpu_seconds"] - before["cpu_seconds"]) / elapsed * 100.0
            )
        if after:
            router_rss_bytes.append(after["rss_bytes"])

    def worker_deltas(name: str) -> list[float] | None:
        if not usable:
            return None
        deltas: list[float] = []
        for worker_index in range(4):
            before = usable[0]["workers"][worker_index].get(name)
            after = usable[-1]["workers"][worker_index].get(name)
            if before is None or after is None:
                return None
            deltas.append(max(0.0, float(after) - float(before)))
        return deltas

    return {
        "samples": len(samples),
        "usable_samples": len(usable),
        "idle_while_queued_samples": idle_while_queued,
        "idle_while_queued_ratio": idle_while_queued / len(usable) if usable else None,
        "queue_seconds": queue_seconds,
        "max_running": max_running,
        "max_waiting": max_waiting,
        "worker_prompt_token_deltas": worker_deltas("vllm:prompt_tokens_total"),
        "worker_generation_token_deltas": worker_deltas("vllm:generation_tokens_total"),
        "worker_request_success_deltas": worker_deltas("vllm:request_success_total"),
        "router_policy_decisions": counter_delta(
            router_first, router_last, "vllm_router_policy_decisions_total", "worker"
        ),
        "router_fallbacks": counter_delta(
            router_first, router_last, "vllm_router_queue_fallback_total", "reason"
        ),
        "residual_router_local_inflight": metric_value(
            router_last or [], "vllm_router_worker_local_inflight"
        ),
        "residual_worker_running": (
            sum(
                float(worker.get("vllm:num_requests_running") or 0.0)
                for worker in usable[-1]["workers"]
            )
            if usable
            else None
        ),
        "residual_worker_waiting": (
            sum(
                float(worker.get("vllm:num_requests_waiting") or 0.0)
                for worker in usable[-1]["workers"]
            )
            if usable
            else None
        ),
        "router_dispatch_p99_bound_seconds": histogram_quantile_bound(
            router_first, router_last, "vllm_router_dispatch_duration_seconds", 0.99
        ),
        "router_worker_request_p95_bound_seconds": histogram_quantile_by_label(
            router_first,
            router_last,
            "vllm_router_worker_request_duration_seconds",
            0.95,
            "worker",
        ),
        "ttft_p50_bound_seconds": histogram_quantile_bound(
            ttft_first, ttft_last, "vllm:time_to_first_token_seconds", 0.50
        ),
        "ttft_p95_bound_seconds": histogram_quantile_bound(
            ttft_first, ttft_last, "vllm:time_to_first_token_seconds", 0.95
        ),
        "ttft_p99_bound_seconds": histogram_quantile_bound(
            ttft_first, ttft_last, "vllm:time_to_first_token_seconds", 0.99
        ),
        "router_cpu_percent_mean": (
            statistics.fmean(router_cpu_percent) if router_cpu_percent else None
        ),
        "router_cpu_percent_p95": percentile(router_cpu_percent, 0.95),
        "router_rss_peak_bytes": max(router_rss_bytes) if router_rss_bytes else None,
    }


async def run_round(
    args: argparse.Namespace,
    items: list[WorkItem],
    round_number: int,
    executor: concurrent.futures.Executor,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    metric_samples: list[dict[str, Any]] = []
    stop = asyncio.Event()
    sampler = asyncio.create_task(
        sample_metrics(
            stop,
            executor,
            args.router_metrics,
            args.worker_metrics,
            args.router_pid,
            args.sample_interval,
            metric_samples,
        )
    )
    await asyncio.sleep(args.sample_interval)
    started = time.monotonic()
    semaphore = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(
        *[
            issue_request(
                item,
                args.endpoint.rstrip("/") + "/v1/chat/completions",
                args.timeout,
                semaphore,
                executor,
                round_number,
            )
            for item in items
        ]
    )
    makespan = time.monotonic() - started
    await asyncio.sleep(args.sample_interval)
    stop.set()
    await sampler

    latencies = [record["latency_seconds"] for record in results]
    successes = [
        record for record in results if record["status"] == 200 and record["nonempty"]
    ]
    completion_tokens = sum(record["completion_tokens"] for record in successes)
    metrics_summary = summarize_samples(metric_samples, args.max_num_seqs)
    worker_success_deltas = metrics_summary["worker_request_success_deltas"]
    backend_successes = (
        sum(worker_success_deltas) if worker_success_deltas is not None else None
    )
    round_summary = {
        "round": round_number,
        "requests": len(results),
        "successes": len(successes),
        "failures": len(results) - len(successes),
        "makespan_seconds": makespan,
        "requests_per_second": len(successes) / makespan if makespan else None,
        "completion_tokens_per_second": completion_tokens / makespan
        if makespan
        else None,
        "latency_p50_seconds": percentile(latencies, 0.50),
        "latency_p95_seconds": percentile(latencies, 0.95),
        "latency_p99_seconds": percentile(latencies, 0.99),
        "backend_request_count_matches": (
            backend_successes is not None
            and abs(backend_successes - len(successes)) < 0.5
        ),
        "backend_successes": backend_successes,
        "metrics": metrics_summary,
    }
    return results, metric_samples, round_summary


async def async_main(args: argparse.Namespace) -> int:
    if args.replay:
        items = replay_workload(args.replay, args.model)
    else:
        items = fixed_workload(
            args.model or DEFAULT_MODEL,
            args.requests,
            args.max_tokens,
            args.fixed_class,
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    request_path = output_dir / f"{args.arm}-requests.jsonl"
    metrics_path = output_dir / f"{args.arm}-metrics.jsonl"
    summary_path = output_dir / f"{args.arm}-summary.json"
    warmup_request_path = output_dir / f"{args.arm}-warmup-requests.jsonl"
    warmup_metrics_path = output_dir / f"{args.arm}-warmup-metrics.jsonl"
    warmup_summary_path = output_dir / f"{args.arm}-warmup-summary.json"
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=max(16, args.concurrency + len(args.worker_metrics) + 8),
        thread_name_prefix="router-lab",
    )
    all_results: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    try:
        if args.warmup > 0:
            warmup_items = items[: min(args.warmup, len(items))]
            warmup_results, warmup_metrics, warmup_summary = await run_round(
                args, warmup_items, 0, executor
            )
            warmup_request_path.write_text(
                "".join(
                    json.dumps(record, sort_keys=True) + "\n"
                    for record in warmup_results
                ),
                encoding="utf-8",
            )
            warmup_metrics_path.write_text(
                "".join(
                    json.dumps(record, sort_keys=True) + "\n"
                    for record in warmup_metrics
                ),
                encoding="utf-8",
            )
            warmup_summary_path.write_text(
                json.dumps(warmup_summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if warmup_summary["failures"]:
                raise RuntimeError(f"warmup failed: {warmup_summary}")
        for round_number in range(1, args.rounds + 1):
            results, metrics, summary = await run_round(
                args, items, round_number, executor
            )
            all_results.extend(results)
            for sample in metrics:
                sample["round"] = round_number
            all_metrics.extend(metrics)
            summaries.append(summary)
            print(json.dumps(summary, sort_keys=True), flush=True)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    request_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in all_results),
        encoding="utf-8",
    )
    metrics_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in all_metrics),
        encoding="utf-8",
    )
    valid_rounds = [summary for summary in summaries if summary["failures"] == 0]
    arm_summary = {
        "arm": args.arm,
        "endpoint": args.endpoint,
        "concurrency": args.concurrency,
        "rounds": summaries,
        "median_requests_per_second": median_present(
            [summary["requests_per_second"] for summary in valid_rounds]
        ),
        "median_completion_tokens_per_second": median_present(
            [summary["completion_tokens_per_second"] for summary in valid_rounds]
        ),
        "median_makespan_seconds": median_present(
            [summary["makespan_seconds"] for summary in valid_rounds]
        ),
        "median_latency_p95_seconds": median_present(
            [summary["latency_p95_seconds"] for summary in valid_rounds]
        ),
        "median_latency_p99_seconds": median_present(
            [summary["latency_p99_seconds"] for summary in valid_rounds]
        ),
        "median_idle_while_queued_ratio": median_present(
            [summary["metrics"]["idle_while_queued_ratio"] for summary in valid_rounds]
        ),
        "median_router_dispatch_p99_bound_seconds": median_present(
            [
                summary["metrics"]["router_dispatch_p99_bound_seconds"]
                for summary in valid_rounds
            ]
        ),
        "median_ttft_p50_bound_seconds": median_present(
            [summary["metrics"]["ttft_p50_bound_seconds"] for summary in valid_rounds]
        ),
        "median_ttft_p95_bound_seconds": median_present(
            [summary["metrics"]["ttft_p95_bound_seconds"] for summary in valid_rounds]
        ),
        "median_ttft_p99_bound_seconds": median_present(
            [summary["metrics"]["ttft_p99_bound_seconds"] for summary in valid_rounds]
        ),
        "median_router_cpu_percent_mean": median_present(
            [summary["metrics"]["router_cpu_percent_mean"] for summary in valid_rounds]
        ),
        "median_router_cpu_percent_p95": median_present(
            [summary["metrics"]["router_cpu_percent_p95"] for summary in valid_rounds]
        ),
        "median_router_rss_peak_bytes": median_present(
            [summary["metrics"]["router_rss_peak_bytes"] for summary in valid_rounds]
        ),
        "max_residual_router_local_inflight": max_present(
            [
                summary["metrics"]["residual_router_local_inflight"]
                for summary in summaries
            ]
        ),
        "max_residual_worker_running": max_present(
            [summary["metrics"]["residual_worker_running"] for summary in summaries]
        ),
        "max_residual_worker_waiting": max_present(
            [summary["metrics"]["residual_worker_waiting"] for summary in summaries]
        ),
        "total_failures": sum(summary["failures"] for summary in summaries),
        "all_backend_request_count_matches": bool(summaries)
        and all(summary["backend_request_count_matches"] for summary in summaries),
    }
    summary_path.write_text(
        json.dumps(arm_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(arm_summary, indent=2, sort_keys=True))
    return 0 if arm_summary["total_failures"] == 0 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", required=True, help="Stable label used for output filenames"
    )
    parser.add_argument(
        "--endpoint",
        required=True,
        help="Router API root, for example http://127.0.0.1:8003",
    )
    parser.add_argument(
        "--router-metrics", required=True, help="Router Prometheus metrics URL"
    )
    parser.add_argument(
        "--router-container", help="Router container name for CPU/RSS sampling"
    )
    parser.add_argument(
        "--worker-metrics",
        action="append",
        required=True,
        help="Worker metrics URL; pass four times",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        help=f"Override replay model ID; fixed workloads default to {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--replay", type=Path, help="JSONL containing direct bodies or {id,class,body}"
    )
    parser.add_argument("--requests", type=int, default=96)
    parser.add_argument(
        "--fixed-class",
        choices=("mixed", "text", "image1", "image2"),
        default="mixed",
        help="Homogeneous fixed workload class; ignored for replay input",
    )
    parser.add_argument("--concurrency", type=int, choices=(16, 32), required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()
    if len(args.worker_metrics) != 4:
        parser.error("--worker-metrics must be supplied exactly four times")
    if args.replay and args.requests != 96:
        parser.error("--requests is not used with --replay")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.arm):
        parser.error(
            "--arm may contain only letters, digits, dot, underscore, and dash"
        )
    args.router_pid = container_pid(args.router_container)
    return args


if __name__ == "__main__":
    raise SystemExit(asyncio.run(async_main(parse_args())))
