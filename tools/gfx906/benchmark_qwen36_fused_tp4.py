#!/usr/bin/env python3
"""Measure one fixed-output Qwen3.6 TP4 request batch over OpenAI HTTP."""

import argparse
import json
import statistics
import threading
import time
import urllib.request


def request(
    base_url: str,
    model: str,
    index: int,
    max_tokens: int,
    results: dict[int, dict[str, float | int]],
) -> None:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Benchmark request {index}. Reply with exactly "
                    f"{max_tokens} short tokens."
                ),
            }
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_token = None
    usage: dict[str, int] = {}
    with urllib.request.urlopen(request, timeout=900) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or []
            if choices and choices[0].get("delta", {}).get("content"):
                first_token = first_token or time.perf_counter()
    finished = time.perf_counter()
    results[index] = {
        "latency_s": finished - started,
        "ttft_s": (first_token or finished) - started,
        "completion_tokens": usage.get("completion_tokens", 0),
    }


def run_batch(args: argparse.Namespace) -> dict[str, float | int]:
    results: dict[int, dict[str, float | int]] = {}
    threads = [
        threading.Thread(
            target=request,
            args=(args.base_url, args.model, index, args.max_tokens, results),
        )
        for index in range(args.concurrency)
    ]
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.perf_counter() - started
    if len(results) != args.concurrency:
        raise RuntimeError(f"incomplete result: {len(results)}/{args.concurrency}")
    tokens = sum(int(row["completion_tokens"]) for row in results.values())
    latencies = [float(row["latency_s"]) for row in results.values()]
    ttfts = [float(row["ttft_s"]) for row in results.values()]
    return {
        "concurrency": args.concurrency,
        "elapsed_s": round(elapsed, 4),
        "completion_tokens": tokens,
        "aggregate_completion_tok_s": round(tokens / elapsed, 4),
        "latency_p50_s": round(statistics.median(latencies), 4),
        "latency_max_s": round(max(latencies), 4),
        "ttft_p50_s": round(statistics.median(ttfts), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args()

    if args.warmup:
        warmup: dict[int, dict[str, float | int]] = {}
        request(args.base_url, args.model, -1, 32, warmup)
        print(json.dumps({"warmup": warmup[-1]}, indent=2), flush=True)
        return
    print(json.dumps(run_batch(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
