#!/usr/bin/env python3
"""Screen Qwen GDN output-norm reshape elision at exact rank-local shapes."""

from __future__ import annotations

import argparse
import json
import math
import statistics

import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNormGated


SHAPES = {
    "qwen35_tp1_c1": (1, 32, 128),
    "qwen35_tp1_c8": (8, 32, 128),
    "qwen27_tp4_c1": (1, 12, 128),
    "qwen27_tp4_c8": (8, 12, 128),
}


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)]


def time_call(fn, repeats: int) -> list[float]:
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def summarize(
    samples: list[float], include_samples: bool
) -> dict[str, float | int | list[float]]:
    summary: dict[str, float | int | list[float]] = {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "p95_ms": percentile(samples, 0.95),
        "sample_count": len(samples),
    }
    if include_samples:
        summary["samples_ms"] = samples
    return summary


def run_shape(
    label: str,
    shape: tuple[int, int, int],
    warmups: int,
    repeats: int,
    atol: float,
    rtol: float,
    device: str,
    include_samples: bool,
) -> dict[str, object]:
    tokens, heads, head_dim = shape
    norm = RMSNormGated(
        head_dim,
        eps=1e-6,
        group_size=None,
        norm_before_gate=True,
        activation="silu",
        device=torch.device(device),
        dtype=torch.float16,
    )
    with torch.no_grad():
        norm.weight.copy_(torch.randn_like(norm.weight))
    output = torch.randn(shape, device=device, dtype=torch.float16)
    gate = torch.randn_like(output)

    def control() -> torch.Tensor:
        normalized = norm(
            output.reshape(-1, head_dim), gate.reshape(-1, head_dim)
        ).reshape(tokens, heads, head_dim)
        return normalized.flatten(-2)

    def candidate() -> torch.Tensor:
        return norm.forward_native(output, gate).reshape(tokens, -1)

    for _ in range(warmups):
        control()
        candidate()
    torch.cuda.synchronize()
    control_output = control()
    candidate_output = candidate()
    torch.cuda.synchronize()
    max_abs_error = float(
        (control_output.float() - candidate_output.float()).abs().max().detach()
    )
    allclose = bool(torch.allclose(control_output, candidate_output, atol=atol, rtol=rtol))
    if not allclose:
        raise RuntimeError(f"{label} mismatch: max_abs_error={max_abs_error}")

    control_samples: list[float] = []
    candidate_samples: list[float] = []
    for _ in range(repeats):
        control_samples.extend(time_call(control, 1))
        candidate_samples.extend(time_call(candidate, 1))
    control_summary = summarize(control_samples, include_samples)
    candidate_summary = summarize(candidate_samples, include_samples)
    improvement_pct = (
        (float(control_summary["median_ms"]) - float(candidate_summary["median_ms"]))
        / float(control_summary["median_ms"])
        * 100
    )
    return {
        "shape": {"tokens": tokens, "heads": heads, "head_dim": head_dim},
        "numerical": {"allclose": allclose, "max_abs_error": max_abs_error},
        "control": control_summary,
        "candidate": candidate_summary,
        "candidate_median_improvement_pct": improvement_pct,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-index", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--atol", type=float, default=0.005)
    parser.add_argument("--rtol", type=float, default=0.005)
    parser.add_argument("--include-samples", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not 0 <= args.device_index < torch.cuda.device_count():
        raise RuntimeError(f"ROCm device index {args.device_index} is unavailable")

    torch.manual_seed(142)
    torch.cuda.set_device(args.device_index)
    device = f"cuda:{args.device_index}"
    # RMSNormGated is a vLLM CustomOp and therefore needs the normal runtime
    # configuration context even in this direct, server-free microbenchmark.
    with set_current_vllm_config(VllmConfig()):
        results = {
            label: run_shape(
                label,
                shape,
                args.warmups,
                args.repeats,
                args.atol,
                args.rtol,
                device,
                args.include_samples,
            )
            for label, shape in SHAPES.items()
        }
    device_props = torch.cuda.get_device_properties(args.device_index)
    print(
        json.dumps(
            {
                "device": {
                    "name": device_props.name,
                    "multi_processor_count": device_props.multi_processor_count,
                },
                "results": results,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
