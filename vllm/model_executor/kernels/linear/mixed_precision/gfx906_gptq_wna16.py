# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""gfx906 GPTQ-compatible W4A16 kernel adapter.

The legacy MI50 path repacks compressed-tensors WNA16 weights for the stable
ROCm GPTQ operator. It is selectable only through ``gfx906_gptq`` so generic
ROCm kernels remain the default for every other device and checkpoint format.
"""

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
)
from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig


class Gfx906GPTQWNA16LinearKernel(MPLinearKernel):
    """Use the stable ROCm GPTQ operator for supported gfx906 WNA16 layers."""

    SUPPORTED_QUANT_TYPES = [
        scalar_types.uint4,
        scalar_types.uint4b8,
        scalar_types.uint8,
        scalar_types.uint8b128,
    ]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "Gfx906GPTQWNA16LinearKernel is only enabled on gfx906"
        # Importing rocm.py probes the active device, so defer it until a ROCm
        # kernel is actually being considered rather than at package import time.
        from vllm.platforms.rocm import on_gfx906

        if not on_gfx906():
            return False, "Gfx906GPTQWNA16LinearKernel is only enabled on gfx906"
        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return False, f"Unsupported quant type {c.weight_type}"
        if c.has_g_idx:
            return False, "Activation reordering is not supported"
        if c.partition_weight_shape[1] % (32 // c.weight_type.size_bits) != 0:
            return False, "Output features must be divisible by the pack factor"

        group_size = c.group_size if c.group_size != -1 else c.partition_weight_shape[0]
        if group_size not in (32, 64, 128, 256, c.partition_weight_shape[0]):
            return False, f"Unsupported group size {c.group_size}"
        if c.partition_weight_shape[0] % group_size != 0:
            return False, "Input features must be divisible by group size"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        c = self.config

        def transform_w_q(x: BasevLLMParameter) -> BasevLLMParameter:
            permute_param_layout_(x, input_dim=1, output_dim=0, packed_dim=1)
            x.data = x.data.t().contiguous()
            empty_g_idx = torch.empty(0, dtype=torch.int32, device=x.data.device)
            ops.gptq_shuffle(x.data, empty_g_idx, c.weight_type.size_bits)
            return x

        def transform_w_s(x: BasevLLMParameter) -> BasevLLMParameter:
            permute_param_layout_(x, input_dim=1, output_dim=0)
            x.data = x.data.t().contiguous()
            return x

        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)

        if c.zero_points:
            w_zp_name = self.w_zp_name
            assert w_zp_name is not None
            qzeros = getattr(layer, w_zp_name, None)
            if qzeros is not None:
                replace_parameter(
                    layer,
                    w_zp_name,
                    torch.nn.Parameter(
                        qzeros.data.t().contiguous(), requires_grad=False
                    ),
                )
        else:
            self.w_zp_name = "qzeros"
            group_size = (
                c.group_size if c.group_size != -1 else c.partition_weight_shape[0]
            )
            groups = c.partition_weight_shape[0] // group_size
            out_features = c.partition_weight_shape[1]
            zeros = torch.full(
                (groups, out_features),
                c.weight_type.bias,
                dtype=torch.int32,
                device=getattr(layer, self.w_q_name).device,
            )
            packed_zeros = pack_quantized_values_into_int32(
                zeros, c.weight_type, packed_dim=1
            )
            setattr(
                layer,
                self.w_zp_name,
                torch.nn.Parameter(packed_zeros.contiguous(), requires_grad=False),
            )

        self.w_gidx_name = "g_idx"
        setattr(
            layer,
            self.w_gidx_name,
            torch.nn.Parameter(
                torch.empty(
                    0,
                    dtype=torch.int32,
                    device=getattr(layer, self.w_q_name).device,
                ),
                requires_grad=False,
            ),
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config
        orig_dtype = x.dtype
        x_2d = x.reshape(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (c.partition_weight_shape[1],)

        w_q, w_s, w_zp, w_g_idx = self._get_weight_params(layer)
        if x_2d.dtype != torch.float16:
            x_2d = x_2d.to(torch.float16)
        if w_s.dtype != torch.float16:
            w_s = w_s.to(torch.float16)

        assert w_zp is not None
        assert w_g_idx is not None
        output = ops.gptq_gemm(
            x_2d.contiguous(),
            w_q,
            w_zp,
            w_s,
            w_g_idx,
            True,
            True,
            c.weight_type.size_bits,
        )
        if output.dtype != orig_dtype:
            output = output.to(orig_dtype)
        if bias is not None:
            output.add_(bias)
        return output.reshape(out_shape)
