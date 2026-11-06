# SPDX-License-Identifier: Apache-2.0
"""Keep unquantized Linear modules and their existing FlagGems mm backend."""

import torch

from vllm.model_executor.layers.linear import UnquantizedLinearMethod


@torch.library.custom_op("vllm_fl::bf16_linear", mutates_args=())
def bf16_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    import flag_gems

    if x.shape[0] == 0:
        return x.new_empty((0, weight.shape[0]))
    # Use the public original operator entry; tuning lives in its existing
    # Iluvatar implementation, including unsupported-shape handling.
    return flag_gems.mm(x, weight.t())


@bf16_linear.register_fake
def _fake(x, weight):
    return x.new_empty((x.shape[0], weight.shape[0]))


class GemsBF16LinearMethod(UnquantizedLinearMethod):
    def apply(self, layer, x, bias=None):
        if (
            bias is not None
            or x.ndim != 2
            or x.device.type != "cuda"
            or x.dtype != torch.bfloat16
            or not x.is_contiguous()
            or layer.weight.dtype != x.dtype
            or not layer.weight.is_contiguous()
        ):
            return super().apply(layer, x, bias)
        return bf16_linear(x, layer.weight)
