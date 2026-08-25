#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Create a copy-on-write INT8 embedding/lm-head Qwen3.5 checkpoint.

Only the two selected safetensor shards, model config, and weight index are
materialized. Every other source file is read-only through a symlink, keeping
the experiment small and reversible.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from compressed_tensors.compressors.pack_quantized.base import pack_to_int32
from safetensors import safe_open
from safetensors.torch import save_file

BITS = 8
GROUP_SIZE = 128
QMAX = 127
TARGETS = {
    "lm_head.weight": "re:.*lm_head$",
    "model.language_model.embed_tokens.weight": "re:.*embed_tokens$",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    if weight.ndim != 2:
        raise ValueError(f"expected rank-2 weight, got {tuple(weight.shape)}")
    out_features, in_features = weight.shape
    if in_features % GROUP_SIZE:
        raise ValueError(f"input width {in_features} is not divisible by {GROUP_SIZE}")

    reference = weight.to(torch.float32)
    groups = reference.reshape(out_features, -1, GROUP_SIZE)
    scale = torch.clamp(groups.abs().amax(dim=-1, keepdim=True) / QMAX, min=1e-10)
    values = torch.clamp(torch.round(groups / scale), -QMAX - 1, QMAX).to(torch.int8)
    reconstructed = values.to(torch.float32) * scale
    relative_error = ((reconstructed - groups).norm() / groups.norm()).item()
    packed = pack_to_int32(
        values.reshape(out_features, in_features), BITS, packed_dim=1
    ).contiguous()
    return packed, scale.squeeze(-1).to(torch.float16).contiguous(), relative_error


def copy_as_symlinks(source: Path, destination: Path, rebuilt: set[str]) -> None:
    destination.mkdir(mode=0o750)
    for entry in source.iterdir():
        if entry.name in rebuilt or entry.name in {
            "config.json",
            "model.safetensors.index.json",
        }:
            continue
        os.symlink(entry.resolve(), destination / entry.name)


def load_shard(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {"format": "pt"})
        for name in handle.keys():
            tensors[name] = handle.get_tensor(name)
    return tensors, metadata


def update_config(config: dict[str, Any]) -> None:
    quant_config = config["quantization_config"]
    groups = quant_config["config_groups"]
    if "group_0" not in groups:
        raise ValueError("expected source W4A16 config group_0")

    quant_config["ignore"] = [
        name for name in quant_config.get("ignore", []) if name != "lm_head"
    ]
    for name, target in TARGETS.items():
        group = copy.deepcopy(groups["group_0"])
        group["targets"] = [target]
        weights = group["weights"]
        weights["num_bits"] = BITS
        weights["group_size"] = GROUP_SIZE
        weights["symmetric"] = True
        weights["type"] = "int"
        groups[f"someai_{name.replace('.', '_')}_int8"] = group


def write_index(
    source_index: dict[str, Any], destination: Path, target_shards: dict[str, str]
) -> None:
    index = copy.deepcopy(source_index)
    weight_map = index["weight_map"]
    for name, shard in target_shards.items():
        del weight_map[name]
        stem = name.removesuffix(".weight")
        for suffix in ("weight_packed", "weight_scale", "weight_shape"):
            weight_map[f"{stem}.{suffix}"] = shard
    (destination / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def materialize(
    source: Path, destination: Path, max_relative_error: float
) -> dict[str, Any]:
    source_index_path = source / "model.safetensors.index.json"
    source_config_path = source / "config.json"
    if not source_index_path.is_file() or not source_config_path.is_file():
        raise FileNotFoundError("source must contain config.json and safetensors index")
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")

    source_index = json.loads(source_index_path.read_text(encoding="utf-8"))
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    weight_map = source_index["weight_map"]
    missing = sorted(set(TARGETS) - set(weight_map))
    if missing:
        raise KeyError(f"source checkpoint missing target weights: {missing}")

    target_shards = {name: weight_map[name] for name in TARGETS}
    rebuilt_shards = set(target_shards.values())
    copy_as_symlinks(source, destination, rebuilt_shards)
    results: dict[str, Any] = {}

    for shard in sorted(rebuilt_shards):
        tensors, metadata = load_shard(source / shard)
        for name, mapped_shard in target_shards.items():
            if mapped_shard != shard:
                continue
            weight = tensors.pop(name)
            packed, scale, relative_error = quantize(weight)
            if relative_error > max_relative_error:
                raise ValueError(
                    f"{name} relative error {relative_error:.6f} exceeds "
                    f"{max_relative_error:.6f}"
                )
            stem = name.removesuffix(".weight")
            tensors[f"{stem}.weight_packed"] = packed
            tensors[f"{stem}.weight_scale"] = scale
            tensors[f"{stem}.weight_shape"] = torch.tensor(
                list(weight.shape), dtype=torch.int64
            )
            results[name] = {
                "source_dtype": str(weight.dtype),
                "shape": list(weight.shape),
                "relative_error": relative_error,
                "source_bytes": weight.numel() * weight.element_size(),
                "packed_bytes": packed.numel() * packed.element_size(),
                "scale_bytes": scale.numel() * scale.element_size(),
            }
        save_file(tensors, destination / shard, metadata=metadata)

    output_config = copy.deepcopy(source_config)
    update_config(output_config)
    (destination / "config.json").write_text(
        json.dumps(output_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_index(source_index, destination, target_shards)

    materialized = [
        destination / "config.json",
        destination / "model.safetensors.index.json",
        *(destination / shard for shard in sorted(rebuilt_shards)),
    ]
    return {
        "source": str(source),
        "destination": str(destination),
        "bits": BITS,
        "group_size": GROUP_SIZE,
        "max_relative_error": max_relative_error,
        "targets": results,
        "materialized_files": {path.name: sha256(path) for path in materialized},
        "symlinked_files": sorted(
            entry.name for entry in destination.iterdir() if entry.is_symlink()
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--max-relative-error", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = materialize(
        args.source.resolve(), args.destination.resolve(), args.max_relative_error
    )
    (args.destination / "conversion-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
