# Copyright (c) 2026 BAAI. All rights reserved.
"""Opaque quantized graph edges backed by public FlagGems Triton operators."""

import torch


@torch.library.custom_op("vllm_fl::scaled_int8_linear", mutates_args=())
def scaled_int8_linear(
    q: torch.Tensor,
    weight: torch.Tensor,
    scale_x: torch.Tensor,
    scale_w: torch.Tensor,
    kind: str,
) -> torch.Tensor:
    from flag_gems import scaled_int8_linear as implementation

    return implementation(q, weight, scale_x, scale_w, kind=kind)


@scaled_int8_linear.register_fake
def _scaled_fake(q, weight, scale_x, scale_w, kind):
    return torch.empty(
        (q.shape[0], weight.shape[1]), device=q.device, dtype=torch.bfloat16
    )


@torch.library.custom_op("vllm_fl::dynamic_int8_linear", mutates_args=())
def dynamic_int8_linear(
    x: torch.Tensor, weight: torch.Tensor, scale_w: torch.Tensor, kind: str
) -> torch.Tensor:
    from flag_gems import dynamic_int8_linear as implementation

    return implementation(x, weight, scale_w, kind=kind)


@dynamic_int8_linear.register_fake
def _dynamic_fake(x, weight, scale_w, kind):
    return torch.empty(
        (x.shape[0], weight.shape[1]), device=x.device, dtype=torch.bfloat16
    )
