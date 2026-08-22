#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure real Qwen3.5 compressed-tensors GPTQ shapes on gfx906.

The benchmark reads an existing model snapshot but never writes model data. It
reproduces the explicit gfx906 GPTQ adapter's packing before timing the stable
``_C::gptq_gemm`` operator. Results are JSONL so a future guarded HIP candidate
can be compared without retaining tensors or checkpoints in the repository.
"""

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
)
from vllm.scalar_type import scalar_types


DEFAULT_SNAPSHOT = Path(
    "/root/.cache/huggingface/hub/"
    "models--cyankiwi--Qwen3.5-9B-AWQ-4bit/snapshots/"
    "156edc4bbeb8d1910ee7be9196bafaf1bc052156"
)


@dataclass(frozen=True)
class ShapeSpec:
    name: str
    prefixes: tuple[str, ...]


LAYER = "model.language_model.layers.3"
SPECS = (
    ShapeSpec("mlp_gate_up", (f"{LAYER}.mlp.gate_proj", f"{LAYER}.mlp.up_proj")),
    ShapeSpec("mlp_down", (f"{LAYER}.mlp.down_proj",)),
    ShapeSpec("mlp_gate", (f"{LAYER}.mlp.gate_proj",)),
    ShapeSpec(
        "self_attn_qkv",
        (
            f"{LAYER}.self_attn.q_proj",
            f"{LAYER}.self_attn.k_proj",
            f"{LAYER}.self_attn.v_proj",
        ),
    ),
    ShapeSpec("self_attn_o", (f"{LAYER}.self_attn.o_proj",)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/mnt/disk2/vllm-gfx906-build/phase-36/results/gptq-shapes.jsonl"
        ),
    )
    parser.add_argument("--m", type=int, nargs="+", default=(1, 8, 27, 190))
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument(
        "--spec",
        choices=tuple(spec.name for spec in SPECS),
        action="append",
        help="Restrict the run to a named shape; repeat to select several.",
    )
    return parser.parse_args()


def load_tensor(snapshot: Path, weight_map: dict[str, str], key: str) -> torch.Tensor:
    filename = weight_map[key]
    with safe_open(str(snapshot / filename), framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def make_gptq_inputs(
    snapshot: Path, weight_map: dict[str, str], spec: ShapeSpec
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
    qweights: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []
    for prefix in spec.prefixes:
        packed = load_tensor(snapshot, weight_map, f"{prefix}.weight_packed")
        scale = load_tensor(snapshot, weight_map, f"{prefix}.weight_scale")
        if packed.dtype != torch.int32 or scale.dtype not in (
            torch.float16,
            torch.bfloat16,
        ):
            raise ValueError(f"Unexpected compressed-tensors layout for {prefix}")
        qweights.append(packed.t().contiguous())
        scales.append(scale.t().contiguous())

    qweight = torch.cat(qweights, dim=1).to("cuda")
    scales_gpu = torch.cat(scales, dim=1).to(device="cuda", dtype=torch.float16)
    size_k = qweight.shape[0] * 8
    size_n = qweight.shape[1]
    groups = scales_gpu.shape[0]
    if size_k % groups != 0 or size_k // groups != 32:
        raise ValueError(
            f"Expected group-size 32 for {spec.name}, got K={size_k}, groups={groups}"
        )

    empty_g_idx = torch.empty(0, dtype=torch.int32, device="cuda")
    ops.gptq_shuffle(qweight, empty_g_idx, 4)
    zeros = torch.full(
        (groups, size_n), scalar_types.uint4.bias, dtype=torch.int32, device="cuda"
    )
    qzeros = pack_quantized_values_into_int32(
        zeros, scalar_types.uint4, packed_dim=1
    ).contiguous()
    return qweight, qzeros, scales_gpu, empty_g_idx, size_k, size_n, groups


def measure(
    activation: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    g_idx: torch.Tensor,
    iterations: int,
    rounds: int,
) -> tuple[float, float, torch.Tensor]:
    def run() -> torch.Tensor:
        return ops.gptq_gemm(
            activation, qweight, qzeros, scales, g_idx, True, True, 4
        )

    for _ in range(5):
        run()
    torch.cuda.synchronize()
    samples_ms: list[float] = []
    output = run()
    for _ in range(rounds):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            output = run()
        torch.cuda.synchronize()
        samples_ms.append((time.perf_counter_ns() - start) / iterations / 1e6)
    return statistics.median(samples_ms), max(samples_ms), output


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a ROCm/CUDA device")
    index_path = args.snapshot / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    selected = [spec for spec in SPECS if not args.spec or spec.name in args.spec]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as output_file:
        for spec in selected:
            qweight, qzeros, scales, g_idx, size_k, size_n, groups = make_gptq_inputs(
                args.snapshot, weight_map, spec
            )
            for size_m in args.m:
                activation = torch.randn(
                    (size_m, size_k), device="cuda", dtype=torch.float16
                )
                median_ms, p95_ms, result = measure(
                    activation,
                    qweight,
                    qzeros,
                    scales,
                    g_idx,
                    args.iterations,
                    args.rounds,
                )
                repeat = ops.gptq_gemm(
                    activation, qweight, qzeros, scales, g_idx, True, True, 4
                )
                repeat_max_abs_delta = (result - repeat).abs().max().item()
                record = {
                    "operator": "_C::gptq_gemm",
                    "spec": spec.name,
                    "m": size_m,
                    "n": size_n,
                    "k": size_k,
                    "groups": groups,
                    "group_size": size_k // groups,
                    "k_split_blocks": size_k // 128,
                    "median_ms": median_ms,
                    "p95_ms": p95_ms,
                    # K-split atomic accumulation is not bitwise deterministic.
                    # This is a repeat delta, not correctness against a separate
                    # dequantized reference.
                    "repeat_max_abs_delta": repeat_max_abs_delta,
                }
                print(json.dumps(record, sort_keys=True), flush=True)
                output_file.write(json.dumps(record, sort_keys=True) + "\n")
            del qweight, qzeros, scales, g_idx
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
