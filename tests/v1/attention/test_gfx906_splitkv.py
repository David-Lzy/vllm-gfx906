# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.v1.attention.ops import chunked_prefill_paged_decode as paged_decode


@pytest.mark.parametrize(
    ("physical_block_size", "expected_logical_block_size"),
    [(784, 16), (512, 32), (30, 2), (7, 1)],
)
def test_gfx906_splitkv_uses_compatible_logical_page_sizes(
    physical_block_size: int, expected_logical_block_size: int
) -> None:
    assert (
        paged_decode._choose_gfx906_splitkv_block_size(physical_block_size)
        == expected_logical_block_size
    )


def test_gfx906_splitkv_qwen27_uses_eight_row_tile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_ROCM_GFX906_SPLITKV_QUERY_ROWS", "8")
    assert paged_decode._get_gfx906_splitkv_query_rows(48, 8, 256) == 8


def test_gfx906_splitkv_keeps_default_tile_for_other_geometries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_ROCM_GFX906_SPLITKV_QUERY_ROWS", "8")
    monkeypatch.setattr(
        paged_decode.triton, "next_power_of_2", lambda _: 8, raising=False
    )
    assert paged_decode._get_gfx906_splitkv_query_rows(32, 8, 256) == 16


def test_gfx906_splitkv_split_count_scales_for_long_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        paged_decode.torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=10),
    )
    monkeypatch.setattr(
        paged_decode.torch.accelerator, "current_device_index", lambda: 0
    )
    assert paged_decode._num_gfx906_splitkv_splits(1, 2, 32780, 16) == 9


def test_gfx906_splitkv_avoids_extra_splits_when_batch_is_saturated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        paged_decode.torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=10),
    )
    monkeypatch.setattr(
        paged_decode.torch.accelerator, "current_device_index", lambda: 0
    )
    assert paged_decode._num_gfx906_splitkv_splits(8, 2, 32780, 16) == 1
