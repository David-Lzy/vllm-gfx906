# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark the split-KV decode path at the Qwen3.6/3.8 TP2 shape.

Run this only inside a disposable ROCm vLLM container. It exercises the exact
per-rank decode geometry used by the Qwen 27B control: 12 query heads, 2 KV
heads, 256-wide heads, and a 784-token physical cache page. The JSON result is
designed for phase scripts to save alongside an end-to-end measurement.
"""

from __future__ import annotations

import argparse
import json
import math

import torch

from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
    paged_attention_2d_splitkv_decode,
)


def parse_splits(value: str) -> list[int]:
    splits = [int(item) for item in value.split(",") if item]
    if not splits or any(split < 1 for split in splits):
        raise argparse.ArgumentTypeError("splits must be positive integers")
    return splits


def to_vllm_kv_cache(
    key_cache: torch.Tensor, value_cache: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    num_blocks, block_size, num_kv_heads, head_size = key_cache.shape
    x = 8
    key_cache = (
        key_cache.view(num_blocks, block_size, num_kv_heads, head_size // x, x)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    value_cache = value_cache.permute(0, 2, 3, 1).contiguous()
    return key_cache, value_cache


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=32780)
    parser.add_argument("--physical-block-size", type=int, default=784)
    parser.add_argument("--query-heads", type=int, default=12)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-size", type=int, default=256)
    parser.add_argument("--splits", type=parse_splits, default=parse_splits("1,4,8,14,16,20,24,28,32"))
    parser.add_argument("--warmups", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--atol", type=float, default=0.03)
    parser.add_argument("--rtol", type=float, default=0.03)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm device is required")

    torch.manual_seed(0)
    torch.set_default_device("cuda")
    dtype = torch.float16
    batch_size = 1
    num_blocks = math.ceil(args.seq_len / args.physical_block_size)
    scale = args.head_size**-0.5
    query = torch.randn(batch_size, args.query_heads, args.head_size, dtype=dtype)
    dense_key_cache = torch.randn(
        num_blocks,
        args.physical_block_size,
        args.kv_heads,
        args.head_size,
        dtype=dtype,
    )
    dense_value_cache = torch.randn_like(dense_key_cache)
    key_cache, value_cache = to_vllm_kv_cache(dense_key_cache, dense_value_cache)
    block_tables = torch.arange(num_blocks, dtype=torch.int32).view(batch_size, -1)
    seq_lens = torch.tensor([args.seq_len], dtype=torch.int32)

    def run_once(splits: int) -> torch.Tensor:
        output = torch.empty_like(query)
        kwargs: dict[str, torch.Tensor] = {}
        if splits > 1:
            kwargs["mid_out"] = torch.empty(
                (batch_size, args.query_heads, splits, args.head_size),
                dtype=torch.float32,
            )
            kwargs["mid_lse"] = torch.empty(
                (batch_size, args.query_heads, splits), dtype=torch.float32
            )
        return paged_attention_2d_splitkv_decode(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
            scale=scale,
            output=output,
            actual_max_splits=splits,
            max_seq_len=args.seq_len,
            max_num_splits=max(args.splits),
            **kwargs,
        )

    reference = run_once(1)
    torch.cuda.synchronize()
    measurements: list[dict[str, object]] = []
    for splits in args.splits:
        for _ in range(args.warmups):
            run_once(splits)
        torch.cuda.synchronize()

        elapsed_ms: list[float] = []
        output = reference
        for _ in range(args.repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = run_once(splits)
            end.record()
            end.synchronize()
            elapsed_ms.append(start.elapsed_time(end))

        max_abs_error = float((output.float() - reference.float()).abs().max().item())
        close = bool(torch.allclose(output, reference, atol=args.atol, rtol=args.rtol))
        measurements.append(
            {
                "splits": splits,
                "median_ms": percentile(elapsed_ms, 0.5),
                "p95_ms": percentile(elapsed_ms, 0.95),
                "mean_ms": sum(elapsed_ms) / len(elapsed_ms),
                "max_abs_error": max_abs_error,
                "allclose_to_nonsplit": close,
            }
        )

    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    print(
        json.dumps(
            {
                "shape": {
                    "seq_len": args.seq_len,
                    "physical_block_size": args.physical_block_size,
                    "query_heads": args.query_heads,
                    "kv_heads": args.kv_heads,
                    "head_size": args.head_size,
                    "dtype": str(dtype),
                },
                "device": {
                    "name": device.name,
                    "multi_processor_count": device.multi_processor_count,
                    "warp_size": device.warp_size,
                },
                "warmups": args.warmups,
                "repeats": args.repeats,
                "measurements": measurements,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
