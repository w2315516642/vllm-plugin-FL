# Copyright (c) 2026 BAAI. All rights reserved.
"""Opaque dispatch boundary for fused, bias-free BF16 linear + SwiGLU."""

import torch

from vllm_fl.dispatch import CachedOp

_dispatch = CachedOp("linear_swiglu")


@torch.library.custom_op("vllm_fl::linear_swiglu", mutates_args=())
def linear_swiglu(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _dispatch(x, weight)


@linear_swiglu.register_fake
def _fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0] // 2))
