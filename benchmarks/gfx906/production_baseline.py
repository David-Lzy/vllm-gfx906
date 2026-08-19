#!/usr/bin/env python3
"""Run the fixed gfx906 text and multimodal production baseline."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import statistics
import struct
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Scenario:
    """One fixed benchmark workload."""

    name: str
    requests: int
    concurrency: int
    images: int
    image_size: int
    max_tokens: int
    reuse: bool
    grid: bool
    prompt: str


@dataclass
class RequestResult:
    """Result from one OpenAI-compatible request."""

    ok: bool
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    text: str
    error: str = ""


SCENARIOS = (
    Scenario(
        "text",
        16,
        8,
        0,
        0,
        512,
        False,
        False,
        "Explain three practical tradeoffs in continuous batching.",
    ),
    Scenario(
        "image8_unique",
        16,
        8,
        8,
        128,
        128,
        False,
        False,
        "Summarize the colors and visual patterns across all eight images.",
    ),
    Scenario(
        "image8_reuse",
        16,
        8,
        8,
        128,
        128,
        True,
        False,
        "Summarize the colors and visual patterns across all eight images.",
    ),
    Scenario(
        "image32_unique",
        8,
        4,
        32,
        128,
        256,
        False,
        False,
        "Summarize the visual sequence across all 32 images compactly.",
    ),
    Scenario(
        "image32_reuse",
        8,
        4,
        32,
        128,
        256,
        True,
        False,
        "Summarize the visual sequence across all 32 images compactly.",
    ),
    Scenario(
        "image64_unique",
        4,
        4,
        64,
        128,
        128,
        False,
        False,
        "Describe the dominant colors and changes across all 64 images.",
    ),
    Scenario(
        "image64_reuse",
        4,
        4,
        64,
        128,
        128,
        True,
        False,
        "Describe the dominant colors and changes across all 64 images.",
    ),
    Scenario(
        "grid4096_unique_c1",
        4,
        1,
        1,
        4096,
        128,
        False,
        True,
        "Describe the 4 by 4 colored grid and its dominant color layout.",
    ),
    Scenario(
        "grid4096_unique_c4",
        4,
        4,
        1,
        4096,
        128,
        False,
        True,
        "Describe the 4 by 4 colored grid and its dominant color layout.",
    ),
    Scenario(
        "grid4096_reuse_c4",
        4,
        4,
        1,
        4096,
        128,
        True,
        True,
        "Describe the 4 by 4 colored grid and its dominant color layout.",
    ),
)

SELECTED_METRICS = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
    "vllm_router_processed_requests_total",
    "vllm_router_requests_total",
    "vllm_router_active_workers",
    "vllm_router_retries_exhausted_total",
)

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "image_count": {"type": "integer"},
        "has_images": {"type": "boolean"},
        "observations": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 4,
        },
    },
    "required": ["summary", "image_count", "has_images", "observations"],
    "additionalProperties": False,
}


def png_chunk(kind: bytes, data: bytes) -> bytes:
    """Create one PNG chunk."""
    checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


@lru_cache(maxsize=1024)
def make_small_png(size: int, seed: int) -> bytes:
    """Create a deterministic RGB PNG for small-image tests."""
    raw = bytearray()
    for y in range(size):
        raw.append(0)
        for x in range(size):
            raw.extend(
                (
                    (x * 3 + seed * 17) & 255,
                    (y * 5 + seed * 31) & 255,
                    ((x ^ y) + seed * 13) & 255,
                )
            )
    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", header)
        + png_chunk(b"IDAT", zlib.compress(bytes(raw), level=3))
        + png_chunk(b"IEND", b"")
    )


@lru_cache(maxsize=16)
def make_grid_png(size: int, seed: int) -> bytes:
    """Create a compressible 4x4 RGB grid without third-party libraries."""
    palette = (
        (220, 45, 55),
        (35, 125, 220),
        (45, 180, 90),
        (235, 190, 45),
        (175, 70, 205),
        (30, 190, 195),
        (235, 120, 40),
        (125, 125, 135),
    )
    tile = size // 4
    compressor = zlib.compressobj(level=3)
    compressed: list[bytes] = []
    for y in range(size):
        tile_y = min(3, y // tile)
        row = bytearray(b"\x00")
        for tile_x in range(4):
            base = palette[(tile_y * 4 + tile_x + seed) % len(palette)]
            delta = seed % 251
            color = (
                (base[0] + delta) & 255,
                (base[1] + delta * 3) & 255,
                (base[2] + delta * 7) & 255,
            )
            width = tile if tile_x < 3 else size - tile * 3
            row.extend(bytes(color) * width)
        compressed.append(compressor.compress(bytes(row)))
    compressed.append(compressor.flush())
    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", header)
        + png_chunk(b"IDAT", b"".join(compressed))
        + png_chunk(b"IEND", b"")
    )


def data_uri(data: bytes) -> str:
    """Encode PNG bytes as an OpenAI image data URI."""
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def endpoint_from_base(base_url: str) -> str:
    """Return the chat-completions endpoint for a base URL."""
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def content_for(scenario: Scenario, request_id: int) -> list[dict[str, Any]]:
    """Build one deterministic multimodal message content list."""
    content: list[dict[str, Any]] = []
    scenario_seed = sum(
        (index + 1) * ord(character) for index, character in enumerate(scenario.name)
    )
    seed_base = 0 if scenario.reuse else scenario_seed * 1_000_000 + request_id * 1000
    for index in range(scenario.images):
        if scenario.grid:
            image = make_grid_png(scenario.image_size, seed_base + index)
        else:
            image = make_small_png(scenario.image_size, seed_base + index)
        content.append({"type": "image_url", "image_url": {"url": data_uri(image)}})
    content.append({"type": "text", "text": scenario.prompt})
    return content


def send_request(
    endpoint: str,
    model: str,
    content: list[dict[str, Any]],
    max_tokens: int,
    timeout: float,
    response_format: dict[str, Any] | None = None,
) -> RequestResult:
    """Send one non-streaming OpenAI-compatible chat request."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if response_format is not None:
        payload["response_format"] = response_format
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer EMPTY",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read())
        latency = time.perf_counter() - started
        usage = body.get("usage") or {}
        message = body.get("choices", [{}])[0].get("message", {})
        text = message.get("content") or ""
        return RequestResult(
            ok=bool(text),
            latency_s=latency,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            text=text,
            error="" if text else "empty response content",
        )
    except urllib.error.HTTPError as error:
        try:
            detail = error.read().decode("utf-8", errors="replace")[:2000]
        except Exception:
            detail = str(error)
        return RequestResult(False, time.perf_counter() - started, 0, 0, "", detail)
    except Exception as error:
        return RequestResult(
            False, time.perf_counter() - started, 0, 0, "", repr(error)
        )


def percentile(values: list[float], fraction: float) -> float:
    """Return a nearest-rank percentile for a small sample."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[min(len(ordered) - 1, max(0, index))]


def repetition_detected(text: str) -> bool:
    """Flag obvious repeated eight-word spans in longer output."""
    words = text.lower().split()
    if len(words) < 40:
        return False
    spans = [tuple(words[index : index + 8]) for index in range(len(words) - 7)]
    return max((spans.count(span) for span in set(spans)), default=0) >= 4


def parse_metrics(text: str) -> dict[str, float]:
    """Keep release-relevant scalar Prometheus samples."""
    result: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.rsplit(None, 1)
        if len(fields) != 2:
            continue
        metric = fields[0].split("{", 1)[0]
        if metric not in SELECTED_METRICS:
            continue
        try:
            result[fields[0]] = float(fields[1])
        except ValueError:
            continue
    return result


def fetch_text(url: str, timeout: float) -> str:
    """Fetch a UTF-8 HTTP response."""
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def capture_metrics(metric_urls: dict[str, str], timeout: float) -> dict[str, Any]:
    """Capture selected metrics from all configured endpoints."""
    snapshots: dict[str, Any] = {}
    for name, url in metric_urls.items():
        try:
            text = fetch_text(url, timeout)
            snapshots[name] = {"ok": True, "values": parse_metrics(text)}
        except Exception as error:
            snapshots[name] = {"ok": False, "error": repr(error), "values": {}}
    return snapshots


def router_request_total(snapshot: dict[str, Any]) -> float | None:
    """Return the Router chat request total from a metrics snapshot."""
    values = [
        value
        for source in snapshot.values()
        for key, value in source.get("values", {}).items()
        if key.startswith("vllm_router_requests_total")
    ]
    return sum(values) if values else None


def summarize_results(
    scenario: Scenario,
    repeat: int,
    results: list[RequestResult],
    wall_s: float,
) -> dict[str, Any]:
    """Summarize one measured scenario repeat."""
    successes = [result for result in results if result.ok]
    failures = [result for result in results if not result.ok]
    latencies = [result.latency_s for result in successes]
    prompt_tokens = sum(result.prompt_tokens for result in successes)
    completion_tokens = sum(result.completion_tokens for result in successes)
    return {
        "record": "scenario_repeat",
        "scenario": asdict(scenario),
        "repeat": repeat,
        "successes": len(successes),
        "failures": len(failures),
        "wall_s": round(wall_s, 4),
        "request_per_s": round(len(successes) / wall_s, 6),
        "prompt_tok_per_s": round(prompt_tokens / wall_s, 3),
        "completion_tok_per_s": round(completion_tokens / wall_s, 3),
        "latency_p50_s": round(percentile(latencies, 0.50), 4),
        "latency_p95_s": round(percentile(latencies, 0.95), 4),
        "latency_mean_s": round(statistics.mean(latencies), 4) if latencies else 0.0,
        "quality_ok": all(
            result.text and not repetition_detected(result.text) for result in successes
        ),
        "sample_response": successes[0].text[:1000] if successes else "",
        "first_error": failures[0].error if failures else "",
    }


def run_scenario(
    endpoint: str,
    model: str,
    scenario: Scenario,
    repeat: int,
    timeout: float,
) -> dict[str, Any]:
    """Warm and execute one measured scenario repeat."""
    warm_content = content_for(scenario, -(repeat + 1))
    warmup = send_request(
        endpoint,
        model,
        warm_content,
        min(32, scenario.max_tokens),
        timeout,
    )
    if not warmup.ok:
        return {
            "record": "scenario_repeat",
            "scenario": asdict(scenario),
            "repeat": repeat,
            "successes": 0,
            "failures": 1,
            "quality_ok": False,
            "first_error": f"warmup failed: {warmup.error}",
        }

    started = time.perf_counter()
    results: list[RequestResult] = []
    request_offset = repeat * 100_000
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=scenario.concurrency
    ) as pool:
        futures = [
            pool.submit(
                send_request,
                endpoint,
                model,
                content_for(scenario, request_offset + request_id),
                scenario.max_tokens,
                timeout,
            )
            for request_id in range(scenario.requests)
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return summarize_results(scenario, repeat, results, time.perf_counter() - started)


def run_json_tests(endpoint: str, model: str, timeout: float) -> dict[str, Any]:
    """Run three deterministic multimodal JSON-schema requests."""
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "gfx906_baseline",
            "schema": JSON_SCHEMA,
            "strict": True,
        },
    }
    results: list[dict[str, Any]] = []
    scenario = Scenario(
        "json_image1",
        1,
        1,
        1,
        128,
        256,
        False,
        False,
        "Return the required JSON. There is exactly one image.",
    )
    for request_id in range(3):
        result = send_request(
            endpoint,
            model,
            content_for(scenario, request_id),
            scenario.max_tokens,
            timeout,
            response_format,
        )
        valid = False
        parsed: Any = None
        if result.ok:
            try:
                parsed = json.loads(result.text)
                valid = (
                    isinstance(parsed, dict)
                    and parsed.get("image_count") == 1
                    and parsed.get("has_images") is True
                    and isinstance(parsed.get("summary"), str)
                    and isinstance(parsed.get("observations"), list)
                    and 1 <= len(parsed["observations"]) <= 4
                    and set(parsed) == set(JSON_SCHEMA["required"])
                )
            except json.JSONDecodeError:
                valid = False
        results.append(
            {
                "request": request_id,
                "ok": result.ok,
                "valid": valid,
                "latency_s": round(result.latency_s, 4),
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "response": parsed if valid else result.text[:1000],
                "error": result.error,
            }
        )
    return {
        "record": "json_tests",
        "successes": sum(item["ok"] and item["valid"] for item in results),
        "failures": sum(not (item["ok"] and item["valid"]) for item in results),
        "results": results,
    }


def median_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize repeated scenario records using medians."""
    numeric = (
        "request_per_s",
        "prompt_tok_per_s",
        "completion_tok_per_s",
        "latency_p50_s",
        "latency_p95_s",
        "latency_mean_s",
    )
    summary: dict[str, Any] = {
        "name": records[0]["scenario"]["name"],
        "repeats": len(records),
        "successes": sum(record.get("successes", 0) for record in records),
        "failures": sum(record.get("failures", 0) for record in records),
        "quality_ok": all(record.get("quality_ok", False) for record in records),
        "traffic_clean": all(
            record.get("traffic_clean") is not False for record in records
        ),
        "external_requests": sum(
            record.get("external_requests", 0) for record in records
        ),
    }
    for key in numeric:
        values = [record[key] for record in records if key in record]
        summary[key] = round(statistics.median(values), 6) if values else 0.0
    return summary


def write_markdown(
    path: Path,
    candidate: str,
    model: str,
    summaries: list[dict[str, Any]],
    json_result: dict[str, Any],
) -> None:
    """Write a compact local Markdown scorecard."""
    lines = [
        f"# {candidate}",
        "",
        f"- Timestamp: `{datetime.now(UTC).isoformat()}`",
        f"- Model: `{model}`",
        "",
        (
            "| Scenario | Repeats | OK | Fail | Req/s | Prompt tok/s | "
            "Completion tok/s | p50 s | p95 s | Quality | Traffic clean |"
        ),
        ("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"),
    ]
    for item in summaries:
        lines.append(
            "| {name} | {repeats} | {successes} | {failures} | "
            "{request_per_s:.4f} | {prompt_tok_per_s:.2f} | "
            "{completion_tok_per_s:.2f} | {latency_p50_s:.2f} | "
            "{latency_p95_s:.2f} | {quality_ok} | {traffic_clean} |".format(**item)
        )
    lines.extend(
        [
            "",
            f"JSON schema: {json_result['successes']}/3 passed.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_named_urls(values: list[str]) -> dict[str, str]:
    """Parse repeated NAME=URL command-line values."""
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"metrics URL must be NAME=URL: {value}")
        name, url = value.split("=", 1)
        result[name] = url
    return result


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def main() -> int:
    """Run the baseline and return nonzero when a release gate fails."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--metrics-url", action="append", default=[])
    parser.add_argument(
        "--scenario",
        action="append",
        choices=[scenario.name for scenario in SCENARIOS],
        help="run only the selected scenario; may be repeated",
    )
    args = parser.parse_args()

    endpoint = endpoint_from_base(args.base_url)
    base = args.base_url.rstrip("/")
    root = base[:-3] if base.endswith("/v1") else base
    metric_urls = parse_named_urls(args.metrics_url)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    start_record = {
        "record": "run_start",
        "run_id": run_id,
        "candidate": args.candidate,
        "model": args.model,
        "base_url": args.base_url,
        "repeats": args.repeats,
        "health": fetch_text(root + "/health", 10),
        "models": json.loads(fetch_text(base + "/models", 10)),
        "metrics": capture_metrics(metric_urls, 10),
    }
    append_jsonl(args.jsonl, start_record)

    smoke = Scenario(
        "image1_smoke",
        1,
        1,
        1,
        128,
        64,
        False,
        False,
        "Briefly describe this image.",
    )
    smoke_result = run_scenario(endpoint, args.model, smoke, 0, args.timeout)
    append_jsonl(args.jsonl, smoke_result)
    if smoke_result.get("failures", 1):
        return 2

    selected_names = set(args.scenario or [])
    selected_scenarios = [
        scenario
        for scenario in SCENARIOS
        if not selected_names or scenario.name in selected_names
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for scenario in selected_scenarios:
        grouped[scenario.name] = []
        for repeat in range(args.repeats):
            before = capture_metrics(metric_urls, 10)
            record = run_scenario(endpoint, args.model, scenario, repeat, args.timeout)
            record["run_id"] = run_id
            record["metrics_before"] = before
            after = capture_metrics(metric_urls, 10)
            record["metrics_after"] = after
            before_total = router_request_total(before)
            after_total = router_request_total(after)
            if before_total is not None and after_total is not None:
                actual = int(after_total - before_total)
                expected = scenario.requests + 1
                record["router_request_delta"] = actual
                record["expected_request_delta"] = expected
                record["external_requests"] = max(0, actual - expected)
                record["traffic_clean"] = actual == expected
            grouped[scenario.name].append(record)
            append_jsonl(args.jsonl, record)
            print(
                f"{scenario.name} repeat={repeat + 1}/{args.repeats} "
                f"ok={record.get('successes', 0)} "
                f"fail={record.get('failures', 0)} "
                f"req/s={record.get('request_per_s', 0)}",
                flush=True,
            )

    json_result = run_json_tests(endpoint, args.model, args.timeout)
    json_result["run_id"] = run_id
    append_jsonl(args.jsonl, json_result)
    summaries = [median_summary(records) for records in grouped.values()]
    end_record = {
        "record": "run_end",
        "run_id": run_id,
        "candidate": args.candidate,
        "summaries": summaries,
        "json_tests": {
            "successes": json_result["successes"],
            "failures": json_result["failures"],
        },
        "metrics": capture_metrics(metric_urls, 10),
        "timestamp": datetime.now(UTC).isoformat(),
    }
    append_jsonl(args.jsonl, end_record)
    write_markdown(args.markdown, args.candidate, args.model, summaries, json_result)
    print(json.dumps(end_record, indent=2, sort_keys=True), flush=True)

    scenario_failed = any(
        item["failures"] or not item["quality_ok"] or not item["traffic_clean"]
        for item in summaries
    )
    return 1 if scenario_failed or json_result["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
