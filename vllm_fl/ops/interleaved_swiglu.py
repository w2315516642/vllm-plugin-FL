# Copyright (c) 2026 BAAI. All rights reserved.
"""Explicit layout boundaries; ordinary split-half SwiGLU is unchanged."""

import torch

from vllm_fl.dispatch import CachedOp

_activation = CachedOp("interleaved_swiglu")
_fused = CachedOp("scaled_int8_swiglu_interleaved")


@torch.library.custom_op("vllm_fl::interleaved_swiglu", mutates_args=())
def interleaved_swiglu(x: torch.Tensor) -> torch.Tensor:
    return _activation(x)


@interleaved_swiglu.register_fake
def _activation_fake(x):
    return torch.empty(
        (*x.shape[:-1], x.shape[-1] // 2), device=x.device, dtype=x.dtype
    )


@torch.library.custom_op("vllm_fl::scaled_int8_swiglu_interleaved", mutates_args=())
def scaled_int8_swiglu_interleaved(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale_x: torch.Tensor,
    scale_w: torch.Tensor,
) -> torch.Tensor:
    return _fused(x, weight, scale_x, scale_w)


@scaled_int8_swiglu_interleaved.register_fake
def _fused_fake(x, weight, scale_x, scale_w):
    return torch.empty(
        (x.shape[0], weight.shape[1] // 2), device=x.device, dtype=torch.bfloat16
    )
