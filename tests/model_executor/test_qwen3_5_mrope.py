# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


def test_qwen3_5_mrope_position_contract():
    from vllm.model_executor.models.interfaces import SupportsMRoPE
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLMBase

    assert SupportsMRoPE in Qwen3_5ForCausalLMBase.__bases__

    input_tokens = [17, 23, 42, 99]
    positions, delta = Qwen3_5ForCausalLMBase.get_mrope_input_positions(
        None, input_tokens, []
    )

    expected = torch.tensor([[0, 1, 2, 3]]).expand(3, -1)
    assert positions.dtype == torch.long
    assert torch.equal(positions, expected)
    assert delta == 0
