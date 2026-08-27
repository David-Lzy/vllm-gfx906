#!/usr/bin/env python3
"""Run the compact Qwen3.6 TP2x2 Router comparison workload."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--asset-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rounds", default=3, type=int)
    parser.add_argument("--timeout", default=900, type=float)
    return parser.parse_args()


def write_assets(fixture: Path, output_dir: Path, count: int) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(fixture) as source:
        image = source.convert("RGB").resize((256, 256))

    urls: list[str] = []
    for index in range(count):
        asset = image.copy()
        asset.putpixel(
            (index % 256, (index // 256) % 256),
            ((index * 17) % 256, (index * 31) % 256, (index * 47) % 256),
        )
        path = output_dir / f"asset-{index:03d}.png"
        asset.save(path, format="PNG", optimize=False)
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        urls.append("data:image/png;base64," + encoded)
    return urls


def request(endpoint: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    req = urllib.request.Request(
        endpoint + "/chat/completions",
        data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            response_body = json.loads(response.read())
            status = response.status
    except urllib.error.HTTPError as exc:
        response_body = {"error": exc.read().decode("utf-8", errors="replace")}
        status = exc.code
    except Exception as exc:  # noqa: BLE001 - persist transport evidence.
        response_body = {"error": repr(exc)}
        status = 0

    seconds = time.perf_counter() - started
    content = ""
    completion_tokens = 0
    if status == 200:
        content = str(
            response_body.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        completion_tokens = int(response_body.get("usage", {}).get("completion_tokens", 0))
    return {
        "status": status,
        "seconds": seconds,
        "completion_tokens": completion_tokens,
        "content": content,
        "body": response_body if status != 200 else None,
    }


def payload(
    model: str,
    text: str,
    *,
    max_tokens: int = 128,
    images: list[str] | None = None,
    json_mode: bool = False,
) -> dict[str, Any]:
    content: str | list[dict[str, Any]] = text
    if images:
        content = [{"type": "text", "text": text}]
        content.extend(
            {"type": "image_url", "image_url": {"url": image}}
            for image in images
        )

    result: dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": content}],
    }
    if max_tokens == 128:
        result.update({"min_tokens": 128, "ignore_eos": True})
    if json_mode:
        result["response_format"] = {"type": "json_object"}
    return result


def assert_response(sample: dict[str, Any], exact_tokens: int | None = None) -> None:
    if sample["status"] != 200:
        raise RuntimeError(f"HTTP {sample['status']}: {sample['body']}")
    if not sample["content"].strip():
        raise RuntimeError("empty completion")
    if exact_tokens is not None and sample["completion_tokens"] != exact_tokens:
        raise RuntimeError(
            f"expected {exact_tokens} completion tokens, got "
            f"{sample['completion_tokens']}"
        )


def summarize(samples: list[dict[str, Any]]) -> dict[str, float | int]:
    seconds = [sample["seconds"] for sample in samples]
    rates = [sample["completion_tokens"] / sample["seconds"] for sample in samples]
    ordered = sorted(seconds)
    return {
        "samples": len(samples),
        "median_completion_tok_s": statistics.median(rates),
        "mean_completion_tok_s": statistics.fmean(rates),
        "median_seconds": statistics.median(seconds),
        "p95_seconds": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
    }


def main() -> int:
    args = parse_args()
    if args.rounds < 1:
        raise SystemExit("--rounds must be positive")

    assets = write_assets(args.fixture, args.asset_dir, 64 + args.rounds * 48)
    cursor = 0

    def next_images(count: int) -> list[str]:
        nonlocal cursor
        selected = assets[cursor:cursor + count]
        cursor += count
        return selected

    result: dict[str, Any] = {"endpoint": args.endpoint, "model": args.model}
    smoke: dict[str, dict[str, Any]] = {}
    smoke_cases = {
        "text": payload(args.model, "Reply with one concise sentence about GPU inference.", max_tokens=32),
        "image1": payload(args.model, "Describe this image in one sentence.", max_tokens=48, images=next_images(1)),
        "image2": payload(args.model, "Describe these two images in one sentence.", max_tokens=48, images=next_images(2)),
    }
    for name, body in smoke_cases.items():
        sample = request(args.endpoint, body, args.timeout)
        assert_response(sample)
        smoke[name] = {key: sample[key] for key in ("status", "seconds", "completion_tokens", "content")}
    result["smoke"] = smoke

    json_samples: list[dict[str, Any]] = []
    for _ in range(3):
        sample = request(
            args.endpoint,
            payload(
                args.model,
                "Return exactly {\"status\":\"ok\"}.",
                max_tokens=32,
                json_mode=True,
            ),
            args.timeout,
        )
        assert_response(sample)
        if json.loads(sample["content"]).get("status") != "ok":
            raise RuntimeError(f"unexpected JSON result: {sample['content']}")
        json_samples.append({key: sample[key] for key in ("status", "seconds", "content")})
    result["json"] = json_samples

    c1_samples: list[dict[str, Any]] = []
    for _ in range(args.rounds):
        sample = request(
            args.endpoint,
            payload(args.model, "Write exactly 128 concise tokens about reliable GPU inference."),
            args.timeout,
        )
        assert_response(sample, exact_tokens=128)
        c1_samples.append(sample)
    result["c1_text"] = {"summary": summarize(c1_samples), "samples": c1_samples}

    c8_samples: list[dict[str, Any]] = []
    c8_body = payload(args.model, "Write exactly 128 concise tokens about reliable GPU inference.")
    for _ in range(args.rounds):
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            batch = list(executor.map(lambda _: request(args.endpoint, c8_body, args.timeout), range(8)))
        elapsed = time.perf_counter() - started
        for sample in batch:
            assert_response(sample, exact_tokens=128)
        c8_samples.append({
            "seconds": elapsed,
            "completion_tokens": sum(sample["completion_tokens"] for sample in batch),
            "requests": batch,
        })
    result["c8_text"] = {"summary": summarize(c8_samples), "samples": c8_samples}

    mixed_samples: list[dict[str, Any]] = []
    for _ in range(args.rounds):
        bodies = [
            payload(args.model, "Write exactly 128 concise tokens about reliable GPU inference.")
            for _ in range(8)
        ]
        bodies.extend(
            payload(args.model, "Describe this image in one sentence.", images=next_images(1))
            for _ in range(4)
        )
        bodies.extend(
            payload(args.model, "Describe these two images in one sentence.", images=next_images(2))
            for _ in range(4)
        )
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            batch = list(executor.map(lambda body: request(args.endpoint, body, args.timeout), bodies))
        elapsed = time.perf_counter() - started
        for index, sample in enumerate(batch):
            assert_response(sample, exact_tokens=128 if index < 8 else None)
        mixed_samples.append({
            "seconds": elapsed,
            "completion_tokens": sum(sample["completion_tokens"] for sample in batch),
            "requests": batch,
        })
    result["mixed_c16"] = {"summary": summarize(mixed_samples), "samples": mixed_samples}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "c1": result["c1_text"]["summary"]["median_completion_tok_s"],
        "c8": result["c8_text"]["summary"]["median_completion_tok_s"],
        "mixed_c16": result["mixed_c16"]["summary"]["median_completion_tok_s"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
