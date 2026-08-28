#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU-only HTTP contract for Phase 166 global FIFO admission."""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import http.client
import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import regex as re

MODEL_ID = "mock-qwen35"
WORKER_PORTS = (18210, 18211, 18212, 18213)
ROUTER_PORT = 18220
METRICS_PORT = 29166
CONTAINER_NAME = "gfx906-router-phase166-contract"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


@dataclass
class WorkerState:
    name: str
    healthy: bool = True
    active: int = 0
    max_active: int = 0
    requests: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def begin(self, marker: str) -> None:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.requests.append(marker)

    def end(self) -> None:
        with self.lock:
            self.active -= 1
            require(self.active >= 0, f"negative mock active count on {self.name}")

    def snapshot(self) -> tuple[int, int, list[str]]:
        with self.lock:
            return self.active, self.max_active, list(self.requests)

    def reset_observations(self) -> None:
        with self.lock:
            require(self.active == 0, f"cannot reset active worker {self.name}")
            self.max_active = 0
            self.requests.clear()


@dataclass
class ClusterState:
    block_event: threading.Event = field(default_factory=threading.Event)
    failed_markers: set[str] = field(default_factory=set)
    dispatches: list[tuple[str, str, float]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, worker: str, marker: str) -> None:
        with self.lock:
            self.dispatches.append((worker, marker, time.monotonic()))

    def consume_failure(self, marker: str) -> bool:
        with self.lock:
            if marker in self.failed_markers:
                return False
            self.failed_markers.add(marker)
            return True

    def reset(self) -> None:
        with self.lock:
            self.failed_markers.clear()
            self.dispatches.clear()
        self.block_event.clear()


class MockWorkerServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 256

    def __init__(
        self,
        address: tuple[str, int],
        state: WorkerState,
        cluster: ClusterState,
    ) -> None:
        super().__init__(address, MockWorkerHandler)
        self.state = state
        self.cluster = cluster


class MockWorkerHandler(BaseHTTPRequestHandler):
    server: MockWorkerServer
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            status = 200 if self.server.state.healthy else 503
            self._send_json(status, {"status": "ok" if status == 200 else "down"})
            return
        if self.path == "/v1/models":
            self._send_json(
                200,
                {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]},
            )
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self._send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        marker = str(payload.get("metadata", {}).get("marker", "unmarked"))
        self.server.state.begin(marker)
        self.server.cluster.record(self.server.state.name, marker)
        active = True
        try:
            if marker.startswith("fail-once:") and self.server.cluster.consume_failure(
                marker
            ):
                self.server.state.end()
                active = False
                self._send_json(500, {"error": "injected first-attempt failure"})
                return

            if marker.startswith("block:"):
                self.server.cluster.block_event.wait(timeout=20)

            delay = float(payload.get("metadata", {}).get("delay", 0.0))
            if delay:
                time.sleep(delay)

            if payload.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                for index in range(60):
                    chunk = {
                        "id": "mock-stream",
                        "object": "chat.completion.chunk",
                        "model": MODEL_ID,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "content": self.server.state.name
                                    if index == 0
                                    else "."
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(0.05)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return

            # A non-streaming model invocation has finished before its buffered
            # HTTP response is written. Track compute occupancy, not the tiny
            # post-compute socket-write tail, when checking the admission cap.
            self.server.state.end()
            active = False
            self._send_json(
                200,
                {
                    "id": "mock-completion",
                    "object": "chat.completion",
                    "model": MODEL_ID,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": self.server.state.name,
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            )
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            if active:
                self.server.state.end()


def request_body(
    marker: str,
    *,
    delay: float = 0.0,
    stream: bool = False,
    padding: int = 0,
) -> bytes:
    return json.dumps(
        {
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "x" * max(1, padding)}],
            "temperature": 0,
            "max_tokens": 8,
            "stream": stream,
            "metadata": {"marker": marker, "delay": delay},
        }
    ).encode()


def post(
    marker: str,
    *,
    delay: float = 0.0,
    stream: bool = False,
    padding: int = 0,
    timeout: float = 30,
) -> tuple[int, bytes, dict[str, str]]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{ROUTER_PORT}/v1/chat/completions",
        data=request_body(marker, delay=delay, stream=stream, padding=padding),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read(), dict(response.headers.items())
    except urllib.error.HTTPError as error:
        return error.code, error.read(), dict(error.headers.items())


def header_value(headers: dict[str, str], name: str) -> str | None:
    expected = name.lower()
    return next(
        (value for key, value in headers.items() if key.lower() == expected), None
    )


def get_text(port: int, path: str, timeout: float = 3) -> str:
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}{path}", timeout=timeout
    ) as response:
        return response.read().decode()


def metric_samples(text: str, name: str) -> list[tuple[dict[str, str], float]]:
    samples: list[tuple[dict[str, str], float]] = []
    pattern = re.compile(rf"^{re.escape(name)}(?:\{{([^}}]*)\}})?\s+([0-9.eE+-]+)$")
    for line in text.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        labels = dict(re.findall(r'(\w+)="([^"]*)"', match.group(1) or ""))
        samples.append((labels, float(match.group(2))))
    return samples


def metric_value(name: str) -> float:
    samples = metric_samples(get_text(METRICS_PORT, "/metrics"), name)
    return sum(value for _, value in samples)


def wait_until(
    predicate: Callable[[], bool],
    message: str,
    *,
    timeout: float = 10,
    interval: float = 0.02,
) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            if predicate():
                return
        except Exception as error:  # noqa: BLE001
            last_error = error
        time.sleep(interval)
    suffix = f"; last error: {last_error}" if last_error else ""
    raise AssertionError(f"{message}{suffix}")


def start_router(
    image: str,
    log_path: Path,
    *,
    worker_capacity: int = 8,
    queue_size: int = 96,
    queue_bytes: int = 1_073_741_824,
    queue_timeout: int = 30,
) -> subprocess.Popen[bytes]:
    subprocess.run(
        ["docker", "rm", "-f", CONTAINER_NAME],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        CONTAINER_NAME,
        "--network",
        "host",
        image,
        "vllm-router",
        "--host",
        "127.0.0.1",
        "--port",
        str(ROUTER_PORT),
        "--worker-urls",
        *[f"http://127.0.0.1:{port}" for port in WORKER_PORTS],
        "--policy",
        "global_fifo",
        "--global-fifo-worker-capacity",
        str(worker_capacity),
        "--global-fifo-queue-size",
        str(queue_size),
        "--global-fifo-queue-bytes",
        str(queue_bytes),
        "--global-fifo-queue-timeout-secs",
        str(queue_timeout),
        "--max-concurrent-requests",
        str(worker_capacity * len(WORKER_PORTS) + queue_size),
        "--max-payload-size",
        "1048576",
        "--request-timeout-secs",
        str(max(60, queue_timeout + 5)),
        "--retry-max-retries",
        "1",
        "--retry-initial-backoff-ms",
        "1",
        "--retry-max-backoff-ms",
        "2",
        "--health-check-interval-secs",
        "1",
        "--health-check-timeout-secs",
        "1",
        "--health-failure-threshold",
        "1",
        "--health-success-threshold",
        "1",
        "--worker-startup-timeout-secs",
        "30",
        "--worker-startup-check-interval",
        "1",
        "--prometheus-host",
        "127.0.0.1",
        "--prometheus-port",
        str(METRICS_PORT),
    ]
    log_handle = log_path.open("ab")
    process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT)
    process.log_handle = log_handle  # type: ignore[attr-defined]
    wait_until(
        lambda: "healthy" in get_text(ROUTER_PORT, "/health").lower(),
        "Router did not become healthy",
        timeout=30,
        interval=0.2,
    )
    models = json.loads(get_text(ROUTER_PORT, "/v1/models"))
    require(any(item.get("id") == MODEL_ID for item in models["data"]), "model missing")
    return process


def stop_router(process: subprocess.Popen[bytes] | None) -> None:
    subprocess.run(
        ["docker", "rm", "-f", CONTAINER_NAME],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if process is not None:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        process.log_handle.close()  # type: ignore[attr-defined]


def wait_for_zero(states: list[WorkerState]) -> None:
    wait_until(
        lambda: all(state.snapshot()[0] == 0 for state in states),
        "mock workers did not drain",
        timeout=20,
    )
    wait_until(
        lambda: (
            metric_value("vllm_router_admission_queue_depth_total") == 0
            and metric_value("vllm_router_admission_worker_slots_used") == 0
        ),
        "Router admission state did not drain",
        timeout=10,
    )


def reset_states(states: list[WorkerState], cluster: ClusterState) -> None:
    wait_for_zero(states)
    cluster.reset()
    for state in states:
        state.reset_observations()


def run_contract(image: str, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "router.log"
    log_path.unlink(missing_ok=True)
    cluster = ClusterState()
    states = [WorkerState(f"worker-{index}") for index in range(4)]
    servers = [
        MockWorkerServer(("127.0.0.1", port), state, cluster)
        for port, state in zip(WORKER_PORTS, states, strict=True)
    ]
    threads = [
        threading.Thread(target=server.serve_forever, daemon=True) for server in servers
    ]
    for thread in threads:
        thread.start()

    process: subprocess.Popen[bytes] | None = None
    results: dict[str, Any] = {}
    try:
        process = start_router(image, log_path)

        # Burst 128: 32 are dispatched and the other 96 wait at the Router.
        peak_queue = 0.0
        with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
            futures = [
                executor.submit(post, f"burst:{index}", delay=0.4, timeout=30)
                for index in range(128)
            ]
            while any(not future.done() for future in futures):
                peak_queue = max(
                    peak_queue,
                    metric_value("vllm_router_admission_queue_depth_total"),
                )
                time.sleep(0.01)
            responses = [future.result() for future in futures]
        status_counts = collections.Counter(status for status, _, _ in responses)
        error_bodies = [
            body.decode(errors="replace")[:300]
            for status, body, _ in responses
            if status != 200
        ][:5]
        require(
            all(status == 200 for status, _, _ in responses),
            (
                "burst returned errors: "
                f"statuses={dict(status_counts)} bodies={error_bodies}"
            ),
        )
        maxima = [state.snapshot()[1] for state in states]
        require(all(maximum <= 8 for maximum in maxima), f"worker exceeded 8: {maxima}")
        require(peak_queue >= 64, f"burst did not exercise Router queue: {peak_queue}")
        results["burst_128"] = {"max_active": maxima, "peak_queue": peak_queue}
        reset_states(states, cluster)

        # Keep one capacity-1 worker healthy so HTTP arrival order is a direct
        # observation of Router FIFO order rather than host socket scheduling.
        stop_router(process)
        process = start_router(image, log_path, worker_capacity=1, queue_size=16)
        for state in states[1:]:
            state.healthy = False
        wait_until(
            lambda: (
                sum(
                    value
                    for _, value in metric_samples(
                        get_text(METRICS_PORT, "/metrics"), "vllm_router_worker_health"
                    )
                )
                == 1
            ),
            "Router did not converge to one healthy FIFO worker",
            timeout=5,
        )
        cluster.block_event.clear()
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            blockers = [executor.submit(post, "block:0")]
            wait_until(
                lambda: sum(state.snapshot()[0] for state in states) == 1,
                "capacity-one blocker did not occupy the FIFO worker",
            )
            queued = []
            for index in range(8):
                queued.append(executor.submit(post, f"fifo:{index}", delay=0.02))
                expected = index + 1
                wait_until(
                    lambda expected=expected: (
                        metric_value("vllm_router_admission_queue_depth_total")
                        >= expected
                    ),
                    f"FIFO request {index} was not queued",
                )
            cluster.block_event.set()
            responses = [future.result() for future in blockers + queued]
        require(
            all(status == 200 for status, _, _ in responses), "FIFO requests failed"
        )
        fifo_arrivals = [
            int(marker.split(":", 1)[1])
            for _, marker, _ in cluster.dispatches
            if marker.startswith("fifo:")
        ]
        require(fifo_arrivals == list(range(8)), f"FIFO order changed: {fifo_arrivals}")
        results["fifo"] = {"arrival_order": fifo_arrivals}
        reset_states(states, cluster)
        for state in states[1:]:
            state.healthy = True
        wait_until(
            lambda: (
                sum(
                    value
                    for _, value in metric_samples(
                        get_text(METRICS_PORT, "/metrics"), "vllm_router_worker_health"
                    )
                )
                == 4
            ),
            "Router did not restore all workers after FIFO test",
            timeout=5,
        )

        # Client cancellation removes the queued body and request.
        cluster.block_event.clear()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            blockers = [
                executor.submit(post, f"block:cancel:{index}") for index in range(4)
            ]
            wait_until(
                lambda: sum(state.snapshot()[0] for state in states) == 4,
                "cancel blockers did not occupy slots",
            )
            connection = http.client.HTTPConnection("127.0.0.1", ROUTER_PORT, timeout=5)
            body = request_body("cancel:queued", padding=2048)
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=body,
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(len(body)),
                },
            )
            wait_until(
                lambda: metric_value("vllm_router_admission_queue_depth_total") == 1,
                "cancel test request did not queue",
            )
            connection.close()
            wait_until(
                lambda: metric_value("vllm_router_admission_queue_depth_total") == 0,
                "client disconnect did not remove queued request",
            )
            cluster.block_event.set()
            require(
                all(future.result()[0] == 200 for future in blockers), "blocker failed"
            )
        results["queued_cancel"] = "pass"
        reset_states(states, cluster)

        # Queue request limit returns OpenAI 429 and Retry-After.
        stop_router(process)
        process = start_router(image, log_path, worker_capacity=1, queue_size=2)
        cluster.block_event.clear()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            blockers = [
                executor.submit(post, f"block:limit:{index}") for index in range(4)
            ]
            wait_until(
                lambda: sum(state.snapshot()[0] for state in states) == 4,
                "limit blockers did not occupy slots",
            )
            queued = [
                executor.submit(post, f"limit:queued:{index}") for index in range(2)
            ]
            wait_until(
                lambda: metric_value("vllm_router_admission_queue_depth_total") == 2,
                "request-limit queue did not fill",
            )
            status, body, headers = post("limit:rejected")
            require(status == 429, f"queue limit returned {status}")
            require(
                header_value(headers, "Retry-After") == "30",
                "429 lacked Retry-After: 30",
            )
            require(
                json.loads(body)["error"]["code"] == "router_admission_queue_full",
                "bad 429",
            )
            cluster.block_event.set()
            require(
                all(future.result()[0] == 200 for future in blockers + queued),
                "limit cleanup requests failed",
            )
        results["request_limit"] = "pass"
        reset_states(states, cluster)

        # Queue byte limit is independent of request count.
        stop_router(process)
        process = start_router(
            image,
            log_path,
            worker_capacity=1,
            queue_size=4,
            queue_bytes=1400,
        )
        cluster.block_event.clear()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            blockers = [
                executor.submit(post, f"block:bytes:{index}") for index in range(4)
            ]
            wait_until(
                lambda: sum(state.snapshot()[0] for state in states) == 4,
                "byte blockers did not occupy slots",
            )
            queued = executor.submit(post, "bytes:queued", padding=900)
            wait_until(
                lambda: metric_value("vllm_router_admission_queue_depth_total") == 1,
                "byte-limit request did not queue",
            )
            status, body, headers = post("bytes:rejected", padding=900)
            require(status == 429, f"byte limit returned {status}")
            require(
                header_value(headers, "Retry-After") == "30",
                "byte 429 lacked Retry-After",
            )
            require(
                json.loads(body)["error"]["code"] == "router_admission_bytes_full",
                "bad byte 429",
            )
            cluster.block_event.set()
            require(
                all(future.result()[0] == 200 for future in blockers + [queued]),
                "byte-limit cleanup requests failed",
            )
        results["byte_limit"] = "pass"
        reset_states(states, cluster)

        # Short contract timeout proves 408 cleanup without waiting 12 hours.
        stop_router(process)
        process = start_router(
            image,
            log_path,
            worker_capacity=1,
            queue_size=4,
            queue_timeout=1,
        )
        cluster.block_event.clear()
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            blockers = [
                executor.submit(post, f"block:timeout:{index}") for index in range(4)
            ]
            wait_until(
                lambda: sum(state.snapshot()[0] for state in states) == 4,
                "timeout blockers did not occupy slots",
            )
            started = time.monotonic()
            status, body, _ = post("timeout:queued", timeout=5)
            elapsed = time.monotonic() - started
            require(status == 408, f"queue timeout returned {status}")
            require(
                json.loads(body)["error"]["code"] == "router_admission_timeout",
                "bad 408",
            )
            require(
                0.7 <= elapsed <= 3.0, f"unexpected timeout duration {elapsed:.3f}s"
            )
            cluster.block_event.set()
            require(
                all(future.result()[0] == 200 for future in blockers), "blocker failed"
            )
        results["queue_timeout"] = {"elapsed_seconds": elapsed}
        reset_states(states, cluster)

        # Streaming client disconnect must release its lease immediately.
        stop_router(process)
        process = start_router(image, log_path)
        connection = http.client.HTTPConnection("127.0.0.1", ROUTER_PORT, timeout=5)
        stream_body = request_body("stream:disconnect", stream=True)
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=stream_body,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(stream_body)),
            },
        )
        response = connection.getresponse()
        require(response.status == 200, f"stream returned {response.status}")
        require(bool(response.readline()), "stream returned no first chunk")
        connection.close()
        wait_until(
            lambda: metric_value("vllm_router_admission_worker_slots_used") == 0,
            "stream disconnect leaked a slot",
            timeout=5,
        )
        results["stream_disconnect"] = "pass"
        reset_states(states, cluster)

        # One backend retry must use another healthy worker and drain both leases.
        status, _, _ = post("fail-once:retry")
        require(status == 200, f"retry returned {status}")
        retry_dispatches = [
            worker
            for worker, marker, _ in cluster.dispatches
            if marker == "fail-once:retry"
        ]
        require(len(retry_dispatches) == 2, f"retry dispatch count {retry_dispatches}")
        require(
            retry_dispatches[0] != retry_dispatches[1], "retry reused failed worker"
        )
        results["retry"] = {"workers": retry_dispatches}
        reset_states(states, cluster)

        # Health loss affects future dispatch; recovery returns the worker to service.
        states[0].healthy = False
        wait_until(
            lambda: any(
                labels.get("worker") == f"http://127.0.0.1:{WORKER_PORTS[0]}"
                and value == 0
                for labels, value in metric_samples(
                    get_text(METRICS_PORT, "/metrics"), "vllm_router_worker_health"
                )
            ),
            "Router did not observe worker loss",
            timeout=5,
        )
        for index in range(12):
            require(
                post(f"health-down:{index}")[0] == 200, "health-down request failed"
            )
        require(
            not any(
                marker.startswith("health-down:") for marker in states[0].snapshot()[2]
            ),
            "unhealthy worker received a new request",
        )
        states[0].healthy = True
        wait_until(
            lambda: any(
                labels.get("worker") == f"http://127.0.0.1:{WORKER_PORTS[0]}"
                and value == 1
                for labels, value in metric_samples(
                    get_text(METRICS_PORT, "/metrics"), "vllm_router_worker_health"
                )
            ),
            "Router did not observe worker recovery",
            timeout=5,
        )
        for index in range(8):
            require(post(f"health-up:{index}")[0] == 200, "health-up request failed")
        require(
            any(marker.startswith("health-up:") for marker in states[0].snapshot()[2]),
            "recovered worker did not re-enter service",
        )
        results["health_loss_recovery"] = "pass"
        wait_for_zero(states)

        metrics = get_text(METRICS_PORT, "/metrics")
        violations = sum(
            value
            for _, value in metric_samples(
                metrics, "vllm_router_admission_invariant_violations_total"
            )
        )
        require(violations == 0, f"admission invariant violations: {violations}")
        results["final"] = {
            "queue_depth": metric_value("vllm_router_admission_queue_depth_total"),
            "slots_used": metric_value("vllm_router_admission_worker_slots_used"),
            "invariant_violations": violations,
        }
        return results
    finally:
        cluster.block_event.set()
        stop_router(process)
        for server in servers:
            server.shutdown()
            server.server_close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        default="local/vllm-router:0.1.14-global-fifo-phase166",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/disk2/vllm-gfx906-build/phase166-router/cpu-contract"),
    )
    args = parser.parse_args()

    results = run_contract(args.image, args.output_dir)
    output = args.output_dir / "contract-results.json"
    output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"PASS: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
