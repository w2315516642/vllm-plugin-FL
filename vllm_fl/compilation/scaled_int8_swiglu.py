# Copyright (c) 2026 BAAI. All rights reserved.
"""Semantic matching for the W8A8 projection followed by native SwiGLU."""

import functools

import torch
import torch.nn.functional as F
from torch._inductor import pattern_matcher as pm

from vllm.compilation.passes.inductor_pass import enable_fake_mode

from vllm_fl.ops.interleaved_swiglu import (
    interleaved_swiglu,
    scaled_int8_swiglu_interleaved,
)
from vllm_fl.ops.scaled_int8_swiglu import scaled_int8_swiglu
from vllm_fl.quantization.w8a8.ffn import _scaled_mm


def _pattern(x, weight, scale_x, scale_w):
    projection = _scaled_mm(x, weight, scale_x, scale_w, torch.bfloat16)
    n = weight.shape[1] // 2
    return F.silu(projection[..., :n]) * projection[..., n:]


def _replacement(x, weight, scale_x, scale_w):
    return scaled_int8_swiglu(x, weight, scale_x, scale_w)


def _pattern_interleaved(x, weight, scale_x, scale_w):
    return interleaved_swiglu(_scaled_mm(x, weight, scale_x, scale_w, torch.bfloat16))


def _replacement_interleaved(x, weight, scale_x, scale_w):
    return scaled_int8_swiglu_interleaved(x, weight, scale_x, scale_w)


def _supported(match, *, min_tokens):
    tensors = [
        match.kwargs[name].meta.get("val")
        for name in ("x", "weight", "scale_x", "scale_w")
    ]
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        return False
    x, weight, scale_x, scale_w = tensors
    if x.ndim != 2 or weight.ndim != 2:
        return False
    # The pass selects the dynamic token compile range. An independently
    # fixed-size projection inside that graph must still obey its own M.
    if type(x.shape[0]) is int and x.shape[0] < min_tokens:
        return False
    k, columns = weight.shape
    return (
        type(k) is int
        and type(columns) is int
        and 0 < k <= 131071
        and columns > 0
        and columns % 2 == 0
        and x.shape[1] == k
        and x.dtype == weight.dtype == torch.int8
        and x.device.type == "cuda"
        and all(t.device == x.device for t in tensors)
        and x.is_contiguous()
        and weight.t().is_contiguous()
        and scale_x.dtype == scale_w.dtype == torch.float32
        and scale_x.shape == (x.shape[0], 1)
        and scale_w.shape == (columns,)
        and scale_x.is_contiguous()
        and scale_w.is_contiguous()
    )


@functools.lru_cache(maxsize=2)
@enable_fake_mode
def matcher(min_tokens=129):
    patterns = pm.PatternMatcherPass(pass_name="fl_scaled_int8_swiglu")
    pm.register_replacement(
        _pattern,
        _replacement,
        [
            torch.empty((3, 32), dtype=torch.int8, device="cuda"),
            torch.empty((96, 32), dtype=torch.int8, device="cuda").t(),
            torch.empty((3, 1), dtype=torch.float32, device="cuda"),
            torch.empty((96,), dtype=torch.float32, device="cuda"),
        ],
        pm.fwd_only,
        patterns,
        extra_check=functools.partial(_supported, min_tokens=min_tokens),
    )
    pm.register_replacement(
        _pattern_interleaved,
        _replacement_interleaved,
        [
            torch.empty((3, 32), dtype=torch.int8, device="cuda"),
            torch.empty((96, 32), dtype=torch.int8, device="cuda").t(),
            torch.empty((3, 1), dtype=torch.float32, device="cuda"),
            torch.empty((96,), dtype=torch.float32, device="cuda"),
        ],
        pm.fwd_only,
        patterns,
        extra_check=functools.partial(_supported, min_tokens=min_tokens),
    )
    return patterns


def source_uuid(pass_type):
    return pass_type.hash_source(
        _pattern,
        _replacement,
        _pattern_interleaved,
        _replacement_interleaved,
        _supported,
        matcher.__wrapped__,
    )
