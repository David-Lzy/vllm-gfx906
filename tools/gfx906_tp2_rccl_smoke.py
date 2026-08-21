# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Minimal two-GPU RCCL smoke test for the gfx906 Phase 22 runtime."""

import os

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")

    torch.accelerator.set_device_index(device)
    dist.init_process_group(backend="nccl")
    value = torch.tensor([local_rank + 1], dtype=torch.float32, device=device)
    dist.all_reduce(value)
    torch.accelerator.synchronize(device)

    expected = world_size * (world_size + 1) / 2
    assert value.item() == expected, (local_rank, value.item(), expected)
    print(f"rank={local_rank} value={value.item()} world_size={world_size}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
