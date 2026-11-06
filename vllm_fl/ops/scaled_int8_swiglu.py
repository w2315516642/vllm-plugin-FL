# Copyright (c) 2026 BAAI. All rights reserved.
"""Dispatch boundary for INT8 scaled projection with a BF16 SwiGLU epilogue."""

import torch

from vllm_fl.dispatch import CachedOp

_dispatch = CachedOp("scaled_int8_swiglu")


@torch.library.custom_op("vllm_fl::scaled_int8_swiglu", mutates_args=())
def scaled_int8_swiglu(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale_x: torch.Tensor,
    scale_w: torch.Tensor,
) -> torch.Tensor:
    return _dispatch(x, weight, scale_x, scale_w)


@scaled_int8_swiglu.register_fake
def _fake(x, weight, scale_x, scale_w):
    return torch.empty(
        (x.shape[0], weight.shape[1] // 2), device=x.device, dtype=torch.bfloat16
    )
