#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exercise queue-aware Router selection and request-accounting lifecycle."""

from __future__ import annotations

import argparse
import http.client
import json
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

MODEL_ID = "mock-qwen35"
WORKER_PORTS = (18110, 18111, 18112, 18113)
ROUTER_PORT = 18120
METRICS_PORT = 29165
CONTAINER_NAME = "gfx906-router-phase165-contract"


@dataclass
class WorkerState:
    name: str
    base_running: int = 0
    base_waiting: int = 0
    metrics_enabled: bool = True
    active: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def queue_depth(self) -> tuple[int, int]:
        with self.lock:
            return self.base_running + self.active, self.base_waiting

    def begin(self) -> None:
        with self.lock:
            self.active += 1

    def end(self) -> None:
        with self.lock:
            self.active = max(0, self.active - 1)


class MockWorkerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], state: WorkerState) -> None:
        super().__init__(address, MockWorkerHandler)
        self.state = state


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
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        if self.path == "/v1/models":
            self._send_json(
                200,
                {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]},
            )
            return
        if self.path == "/metrics":
            if not self.server.state.metrics_enabled:
                self._send_json(503, {"error": "metrics disabled"})
                return
            running, waiting = self.server.state.queue_depth()
            body = (
                f'vllm:num_requests_running{{model_name="{MODEL_ID}"}} {running}\n'
                f'vllm:num_requests_waiting{{model_name="{MODEL_ID}"}} {waiting}\n'
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self._send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        marker = json.dumps(payload.get("messages", []))
        self.server.state.begin()
        try:
            if "MOCK_FAIL_500" in marker:
                self._send_json(500, {"error": "injected failure"})
                return
            delay_match = re.search(r"MOCK_DELAY_([0-9.]+)", marker)
            if delay_match:
                time.sleep(float(delay_match.group(1)))
            if payload.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                for index in range(20):
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
                    if index == 0 and "STREAM_ABORT_STALL" in marker:
                        time.sleep(3.0)
                    else:
                        time.sleep(0.1)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
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
            self.server.state.end()


def request_json(
    port: int, prompt: str, *, stream: bool = False, timeout: float = 40
) -> tuple[int, bytes]:
    body = json.dumps(
        {
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 8,
            "stream": stream,
        }
    ).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def get_text(port: int, path: str, timeout: float = 3) -> str:
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}{path}", timeout=timeout
    ) as response:
        return response.read().decode()


def metric_values(text: str, name: str) -> dict[str, float]:
    values: dict[str, float] = {}
    pattern = re.compile(rf"^{re.escape(name)}\{{([^}}]*)\}}\s+([0-9.eE+-]+)$")
    for line in text.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        labels = dict(re.findall(r'(\w+)="([^"]*)"', match.group(1)))
        key = labels.get("worker") or labels.get("reason") or "value"
        values[key] = float(match.group(2))
    return values


def wait_for_router(port: int, timeout: float = 20) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            if "healthy" in get_text(port, "/health", timeout=1).lower():
                return
        except Exception as error:  # noqa: BLE001
            last_error = error
        time.sleep(0.2)
    raise RuntimeError(f"router did not become healthy: {last_error}")


def start_router(
    image: str, timeout_seconds: int, log_path: Path
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
        "queue_aware",
        "--queue-metrics-interval-ms",
        "100",
        "--queue-metrics-timeout-ms",
        "50",
        "--queue-metrics-stale-ms",
        "500",
        "--request-timeout-secs",
        str(timeout_seconds),
        "--retry-max-retries",
        "1",
        "--health-check-interval-secs",
        "1",
        "--health-check-timeout-secs",
        "1",
        "--prometheus-host",
        "127.0.0.1",
        "--prometheus-port",
        str(METRICS_PORT),
    ]
    log_handle = log_path.open("ab")
    process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT)
    process.log_handle = log_handle  # type: ignore[attr-defined]
    wait_for_router(ROUTER_PORT)
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
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        log_handle = getattr(process, "log_handle", None)
        if log_handle is not None:
            log_handle.close()


def assert_inflight_zero() -> None:
    metrics = get_text(METRICS_PORT, "/metrics")
    values = metric_values(metrics, "vllm_router_worker_local_inflight")
    if values and any(value != 0 for value in values.values()):
        raise AssertionError(f"stranded in-flight counters: {values}")


def run_contract(image: str, log_path: Path) -> dict[str, Any]:
    states = [WorkerState(f"worker{index}") for index in range(4)]
    servers = [
        MockWorkerServer(("127.0.0.1", port), state)
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
        states[0].base_running, states[0].base_waiting = 8, 3
        states[1].base_running, states[1].base_waiting = 7, 0
        states[2].base_running, states[2].base_waiting = 8, 3
        states[3].base_running, states[3].base_waiting = 3, 0
        process = start_router(image, 30, log_path)
        time.sleep(0.5)

        status, body = request_json(ROUTER_PORT, "QUEUE_SELECTION")
        selected = json.loads(body)["choices"][0]["message"]["content"]
        if status != 200 or selected != "worker3":
            raise AssertionError(
                f"queue selection mismatch: status={status}, worker={selected}"
            )
        results["queue_selection"] = selected
        metrics = get_text(METRICS_PORT, "/metrics")
        dispatch_bounds = {
            float(value)
            for value in re.findall(
                r'^vllm_router_dispatch_duration_seconds_bucket\{[^}]*le="([^"]+)"[^}]*\}\s+',
                metrics,
                re.MULTILINE,
            )
            if value != "+Inf"
        }
        if not {0.002, 0.003}.issubset(dispatch_bounds):
            raise AssertionError(
                "fine dispatch histogram buckets are missing: "
                f"{sorted(dispatch_bounds)}"
            )
        results["dispatch_histogram_bounds_seconds"] = sorted(dispatch_bounds)

        for state in states:
            state.base_running = 0
            state.base_waiting = 0
        time.sleep(0.5)

        slow_result: dict[str, Any] = {}

        def run_slow() -> None:
            slow_status, slow_body = request_json(
                ROUTER_PORT, "MOCK_DELAY_12", timeout=20
            )
            slow_result["status"] = slow_status
            slow_result["worker"] = json.loads(slow_body)["choices"][0]["message"][
                "content"
            ]

        slow_thread = threading.Thread(target=run_slow)
        slow_thread.start()
        time.sleep(0.8)
        status, body = request_json(ROUTER_PORT, "FAST_WHILE_SLOW")
        fast_worker = json.loads(body)["choices"][0]["message"]["content"]
        metrics = get_text(METRICS_PORT, "/metrics")
        inflight = metric_values(metrics, "vllm_router_worker_local_inflight")
        if status != 200 or max(inflight.values(), default=0) != 1:
            raise AssertionError(f"least-inflight admission failed: {inflight}")
        time.sleep(9.7)
        metrics = get_text(METRICS_PORT, "/metrics")
        after_health_cycles = metric_values(
            metrics, "vllm_router_worker_local_inflight"
        )
        if max(after_health_cycles.values(), default=0) != 1:
            raise AssertionError(
                f"health checker reset active in-flight count: {after_health_cycles}"
            )
        slow_thread.join(timeout=5)
        if slow_thread.is_alive() or slow_result.get("status") != 200:
            raise AssertionError(f"slow lifecycle request failed: {slow_result}")
        if fast_worker == slow_result.get("worker"):
            raise AssertionError(
                "fast request reused the worker with an active slow request"
            )
        assert_inflight_zero()
        results["slow_worker"] = slow_result["worker"]
        results["fast_worker"] = fast_worker

        status, body = request_json(ROUTER_PORT, "STREAM_COMPLETE", stream=True)
        if status != 200 or b"data: [DONE]" not in body:
            raise AssertionError("stream completion was truncated")
        assert_inflight_zero()
        results["stream_complete"] = True

        connection = http.client.HTTPConnection("127.0.0.1", ROUTER_PORT, timeout=5)
        payload = json.dumps(
            {
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "STREAM_ABORT_STALL"}],
                "stream": True,
                "max_tokens": 8,
            }
        )
        connection.request(
            "POST",
            "/v1/chat/completions",
            payload,
            {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        response.read(96)
        connection.close()
        # The mock backend emits one chunk and then remains silent for three
        # seconds. Cleanup must follow the client-side body drop, not another
        # backend chunk.
        time.sleep(0.5)
        assert_inflight_zero()
        results["stream_abort_cleanup"] = True

        status, _ = request_json(ROUTER_PORT, "MOCK_FAIL_500")
        if status != 500:
            raise AssertionError(f"expected injected HTTP 500, got {status}")
        time.sleep(0.2)
        assert_inflight_zero()
        results["http_error_cleanup"] = True

        for state in states:
            state.metrics_enabled = False
        time.sleep(0.8)
        status, _ = request_json(ROUTER_PORT, "STALE_TELEMETRY_FALLBACK")
        metrics = get_text(METRICS_PORT, "/metrics")
        fallback = metric_values(metrics, "vllm_router_queue_fallback_total")
        if status != 200 or not fallback:
            raise AssertionError(f"telemetry fallback was not recorded: {fallback}")
        results["fallback_counters"] = fallback
        for state in states:
            state.metrics_enabled = True

        stop_router(process)
        process = start_router(image, 1, log_path)
        status, _ = request_json(ROUTER_PORT, "MOCK_DELAY_3", timeout=10)
        if status < 400:
            raise AssertionError(f"expected timeout status, got {status}")
        time.sleep(0.5)
        assert_inflight_zero()
        results["timeout_status"] = status
        results["timeout_cleanup"] = True
        return results
    finally:
        stop_router(process)
        for server in servers:
            server.shutdown()
            server.server_close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image", default="local/vllm-router:0.1.14-queue-aware-phase165"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--log", type=Path, default=Path("phase165-router-contract.log")
    )
    args = parser.parse_args()

    result = {
        "image": args.image,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "checks": run_contract(args.image, args.log),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
