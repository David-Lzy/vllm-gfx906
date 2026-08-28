#!/usr/bin/env python3
"""Run the Phase 165 side-by-side Router matrix during an approved window."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


OFFICIAL_IMAGE = (
    "vllm/vllm-router@sha256:"
    "4bd8f436cb578a29cc28b1bbf75c9a3a2c2cb4a353f999dc3bfa6adfeebe6dd5"
)
PATCHED_IMAGE = "local/vllm-router:0.1.14-queue-aware-phase165"
PRODUCTION_ROUTER = "qwen35-9b-vllm-router"
OFFICIAL_CONTAINER = "phase165-router-official-rr"
CANDIDATE_CONTAINER = "phase165-router-candidate"
ROTATIONS = (
    ("official-rr", "patched-rr", "least-inflight", "queue-aware"),
    ("queue-aware", "least-inflight", "patched-rr", "official-rr"),
    ("least-inflight", "official-rr", "queue-aware", "patched-rr"),
)
SAFETY_PATTERN = re.compile(
    r"out of memory|\bOOM\b|Failed to advance FSM|"
    r"(?:xgrammar.*(?:error|failed|exception))|"
    r"(?:(?:error|failed|exception).*xgrammar)|"
    r"(?:RCCL|NCCL).*fatal|Traceback \(most recent call last\)",
    re.IGNORECASE,
)


def run(command: list[str], *, capture: bool = False) -> str:
    result = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return result.stdout.strip() if capture else ""


def container_running(name: str) -> bool:
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Running}}", name],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def remove_container(name: str) -> None:
    if container_running(name):
        subprocess.run(["docker", "stop", "--timeout", "15", name], check=False)


def scan_worker_logs(since_epoch: float) -> dict[str, object]:
    result: dict[str, object] = {"since_epoch": since_epoch, "workers": {}, "matches": 0}
    for index in range(4):
        name = f"qwen35-9b-vllm-gpu{index}"
        completed = subprocess.run(
            ["docker", "logs", "--since", str(int(since_epoch)), name],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        matches = [line[-2000:] for line in completed.stdout.splitlines() if SAFETY_PATTERN.search(line)]
        result["workers"][name] = {
            "docker_logs_exit_code": completed.returncode,
            "matches": matches[:50],
            "match_count": len(matches),
        }
        result["matches"] += len(matches)
    return result


def worker_metric_urls() -> list[str]:
    urls: list[str] = []
    for index in range(4):
        name = f"qwen35-9b-vllm-gpu{index}"
        ip = run(
            [
                "docker",
                "inspect",
                "--format",
                "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
                name,
            ],
            capture=True,
        )
        if not ip:
            raise RuntimeError(f"no network address for {name}")
        urls.append(f"http://{ip}:8000/metrics")
    return urls


def queue_depth(url: str) -> tuple[int, int]:
    with urllib.request.urlopen(url, timeout=2.0) as response:
        text = response.read().decode("utf-8", errors="replace")
    running = sum(
        float(value)
        for value in re.findall(r"^vllm:num_requests_running(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", text, re.M)
    )
    waiting = sum(
        float(value)
        for value in re.findall(r"^vllm:num_requests_waiting(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", text, re.M)
    )
    return int(running), int(waiting)


def wait_drained(urls: list[str], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        depths = [queue_depth(url) for url in urls]
        if all(running == 0 and waiting == 0 for running, waiting in depths):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"workers did not drain: {depths}")
        time.sleep(1.0)


def validate_replay(path: Path, expected_requests: int) -> dict[str, object]:
    replay_root = path.resolve().parent
    request_indices: set[int] = set()
    stage_counts: dict[str, int] = {}
    payload_bytes = 0
    max_tokens: list[int] = []
    records = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if "payload_relpath" not in record or "payload_sha256" not in record:
            raise ValueError(f"{path}:{line_number}: exact payload reference and SHA are required")
        payload_path = (replay_root / str(record["payload_relpath"])).resolve()
        if not payload_path.is_relative_to(replay_root):
            raise ValueError(f"{path}:{line_number}: payload path escapes replay root")
        payload = payload_path.read_bytes()
        actual_sha = hashlib.sha256(payload).hexdigest()
        if actual_sha != record["payload_sha256"]:
            raise ValueError(f"{path}:{line_number}: payload SHA256 mismatch")
        body = json.loads(payload)
        if not isinstance(body, dict):
            raise ValueError(f"{path}:{line_number}: payload is not a JSON object")
        if body.get("stream") is True:
            raise ValueError(f"{path}:{line_number}: streaming replay is not supported")
        request_index = int(record.get("request_index", line_number))
        if request_index in request_indices:
            raise ValueError(f"{path}:{line_number}: duplicate request_index {request_index}")
        request_indices.add(request_index)
        stage = str(record.get("stage", "unknown"))
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
        if body.get("max_tokens") is not None:
            max_tokens.append(int(body["max_tokens"]))
        payload_bytes += len(payload)
        records += 1
    if records != expected_requests:
        raise ValueError(f"expected {expected_requests} replay requests, found {records}")
    return {
        "requests": records,
        "payload_bytes": payload_bytes,
        "stage_counts": stage_counts,
        "max_tokens_min": min(max_tokens) if max_tokens else None,
        "max_tokens_max": max(max_tokens) if max_tokens else None,
    }


def start_router(
    helper: Path,
    arm: str,
    official_image: str,
    patched_image: str,
) -> tuple[str, str, str]:
    if arm == "official-rr":
        name, image, policy, api_port, metrics_port = (
            OFFICIAL_CONTAINER,
            official_image,
            "round_robin",
            8004,
            29002,
        )
    else:
        policy = {
            "patched-rr": "round_robin",
            "least-inflight": "least_inflight",
            "queue-aware": "queue_aware",
        }[arm]
        name, image, api_port, metrics_port = CANDIDATE_CONTAINER, patched_image, 8003, 29001
    run([str(helper), name, image, policy, str(api_port), str(metrics_port)])
    return (
        f"http://127.0.0.1:{api_port}",
        f"http://127.0.0.1:{metrics_port}/metrics",
        name,
    )


def benchmark_command(
    benchmark: Path,
    label: str,
    endpoint: str,
    router_metrics: str,
    router_container: str,
    workers: list[str],
    output_dir: Path,
    concurrency: int,
    warmup: int,
    replay: Path | None,
    requests: int,
    fixed_class: str | None,
) -> list[str]:
    command = [
        sys.executable,
        str(benchmark),
        "--arm",
        label,
        "--endpoint",
        endpoint,
        "--router-metrics",
        router_metrics,
        "--router-container",
        router_container,
        "--output-dir",
        str(output_dir),
        "--concurrency",
        str(concurrency),
        "--rounds",
        "1",
        "--warmup",
        str(warmup),
        "--timeout",
        "3600" if replay else "900",
    ]
    for worker in workers:
        command.extend(("--worker-metrics", worker))
    if replay:
        command.extend(("--replay", str(replay)))
    else:
        command.extend(
            (
                "--requests",
                str(requests),
                "--max-tokens",
                "128",
                "--fixed-class",
                fixed_class or "mixed",
            )
        )
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--official-image", default=OFFICIAL_IMAGE)
    parser.add_argument("--patched-image", default=PATCHED_IMAGE)
    parser.add_argument("--fixed-requests-per-class", type=int, default=32)
    parser.add_argument("--expected-replay-requests", type=int, default=120)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate the frozen replay while production is still running, then exit",
    )
    parser.add_argument("--drain-timeout", type=float, default=1800.0)
    parser.add_argument(
        "--maintenance-ready",
        action="store_true",
        help="Assert Phase1 is paused, fallback owns 8002, and worker queues are draining",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.replay.is_file():
        raise SystemExit(f"replay file does not exist: {args.replay}")
    replay_validation = validate_replay(args.replay, args.expected_replay_requests)
    if args.preflight_only:
        print(json.dumps(replay_validation, indent=2, sort_keys=True))
        return 0
    if not args.maintenance_ready:
        raise SystemExit("refusing to run without --maintenance-ready")
    if container_running(PRODUCTION_ROUTER):
        raise SystemExit(f"refusing while production Router {PRODUCTION_ROUTER} is running")

    script_dir = Path(__file__).resolve().parent
    helper = script_dir / "run-router.sh"
    benchmark = script_dir / "benchmark_queue_router.py"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    workers = worker_metric_urls()
    wait_drained(workers, args.drain_timeout)
    scenarios = tuple(
        (f"{fixed_class}-c{concurrency}", concurrency, None, args.fixed_requests_per_class, fixed_class)
        for concurrency in (16, 32)
        for fixed_class in ("text", "image1", "image2")
    ) + (("phase1-c32", 32, args.replay, 0, None),)
    manifest: dict[str, object] = {
        "official_image": args.official_image,
        "patched_image": args.patched_image,
        "replay": str(args.replay),
        "replay_sha256": hashlib_file(args.replay),
        "replay_validation": replay_validation,
        "workers": workers,
        "runs": [],
    }
    bundle_manifest = args.replay.parent / "MANIFEST.sha256"
    if bundle_manifest.is_file():
        manifest["bundle_manifest_sha256"] = hashlib_file(bundle_manifest)
    warmed: set[tuple[str, str]] = set()
    try:
        for scenario, concurrency, replay, request_count, fixed_class in scenarios:
            for rotation_index, rotation in enumerate(ROTATIONS, 1):
                for arm in rotation:
                    wait_drained(workers, args.drain_timeout)
                    endpoint, router_metrics, container = start_router(
                        helper, arm, args.official_image, args.patched_image
                    )
                    key = (scenario, arm)
                    # Warm the complete scenario exactly once per policy. In
                    # particular, do not let the first replay arm pay the cold
                    # processor-cache cost while later arms inherit all 120
                    # cached payloads.
                    warmup = 1_000_000 if key not in warmed else 0
                    warmed.add(key)
                    label = f"{arm}-{scenario}-r{rotation_index}"
                    started = time.time()
                    run(
                        benchmark_command(
                            benchmark,
                            label,
                            endpoint,
                            router_metrics,
                            container,
                            workers,
                            args.output_dir,
                            concurrency,
                            warmup,
                            replay,
                            request_count,
                            fixed_class,
                        )
                    )
                    wait_drained(workers, args.drain_timeout)
                    safety = scan_worker_logs(started)
                    (args.output_dir / f"{label}-safety.json").write_text(
                        json.dumps(safety, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                    )
                    manifest["runs"].append(
                        {
                            "label": label,
                            "container": container,
                            "started_at": started,
                            "completed_at": time.time(),
                            "safety_matches": safety["matches"],
                        }
                    )
                    (args.output_dir / "manifest.json").write_text(
                        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                    )
                    if safety["matches"]:
                        raise RuntimeError(f"worker safety log gate failed for {label}")
    finally:
        remove_container(CANDIDATE_CONTAINER)
        remove_container(OFFICIAL_CONTAINER)
    return 0


def hashlib_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
