# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

import vllm.v1.worker.gpu_model_runner as model_runner

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    ("enabled", "is_gfx906", "tp_size", "expected_blocking"),
    [
        (False, True, 1, True),
        (True, False, 1, True),
        (True, True, 2, True),
        (True, True, 1, False),
    ],
)
def test_gfx906_tp1_spin_event_gate(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    is_gfx906: bool,
    tp_size: int,
    expected_blocking: bool,
):
    captured: dict[str, bool] = {}

    monkeypatch.setattr(
        model_runner.envs, "VLLM_GFX906_TP1_SPIN_EVENTS", enabled
    )
    monkeypatch.setattr(model_runner, "_is_gfx906", lambda: is_gfx906)
    monkeypatch.setattr(
        model_runner, "get_tp_group", lambda: SimpleNamespace(world_size=tp_size)
    )
    monkeypatch.setattr(
        model_runner.torch.cuda,
        "Event",
        lambda *, blocking: captured.setdefault("blocking", blocking),
    )

    model_runner._make_async_scheduling_event()

    assert captured["blocking"] is expected_blocking
