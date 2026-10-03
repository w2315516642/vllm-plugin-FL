# Copyright (c) 2026 BAAI. All rights reserved.
"""Opt-in semantic fusion, without patching models or bypassing large-M graphs.

Enable with additional_config={"linear_swiglu_fusion": {"max_tokens": 64}}.
The token limit is an execution policy, not an operator shape restriction.
Other compile ranges retain the original linear and activation graph.
"""

import functools

import torch
import torch.nn.functional as F
from torch._inductor import pattern_matcher as pm

from vllm.compilation.passes.inductor_pass import (
    InductorPass,
    enable_fake_mode,
    get_pass_context,
)
from vllm.logger import init_logger

from vllm_fl.ops.linear_swiglu import linear_swiglu

logger = init_logger(__name__)
_PASS_KEY = "post_grad_custom_post_pass"


def _pattern(x, weight):
    projection = F.linear(x, weight)
    n = weight.shape[0] // 2
    return F.silu(projection[..., :n]) * projection[..., n:]


def _replacement(x, weight):
    return linear_swiglu(x, weight)


def _supported(match):
    x = match.kwargs["x"].meta.get("val")
    weight = match.kwargs["weight"].meta.get("val")
    if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor):
        return False
    return (
        x.ndim == weight.ndim == 2
        and x.dtype == weight.dtype == torch.bfloat16
        and x.device.type == "cuda"
        and x.device == weight.device
        and type(weight.shape[0]) is int
        and type(weight.shape[1]) is int
        and weight.shape[0] > 0
        and weight.shape[0] % 2 == 0
        and weight.shape[1] > 0
        and x.is_contiguous()
        and weight.is_contiguous()
    )


@functools.lru_cache(maxsize=1)
@enable_fake_mode
def _matcher():
    patterns = pm.PatternMatcherPass(pass_name="fl_linear_swiglu")
    # Examples specify structure/dtype, not the dimensions accepted by a match.
    pm.register_replacement(
        _pattern,
        _replacement,
        [
            torch.empty((3, 32), dtype=torch.bfloat16, device="cuda"),
            torch.empty((96, 32), dtype=torch.bfloat16, device="cuda"),
        ],
        pm.fwd_only,
        patterns,
        extra_check=_supported,
    )
    return patterns


class LinearSwiGLUPass(InductorPass):
    """Compose with an existing pass and fuse only selected compile ranges."""

    def __init__(self, max_tokens: int, previous=None):
        self.max_tokens = max_tokens
        self.previous = previous
        self.matched_count = 0

    def __call__(self, graph):
        compile_range = get_pass_context().compile_range
        if self.previous is not None and self.previous.is_applicable_for_range(
            compile_range
        ):
            self.previous(graph)
        self.matched_count = 0
        if compile_range.start < 1 or compile_range.end > self.max_tokens:
            return
        self.matched_count = _matcher().apply(graph)
        if self.matched_count:
            logger.debug(
                "Fused %d linear/SwiGLU patterns in range %s",
                self.matched_count,
                compile_range,
            )

    def uuid(self):
        return self.hash_dict(
            {
                "source": self.hash_source(
                    type(self), _pattern, _replacement, _supported, _matcher.__wrapped__
                ),
                "max_tokens": self.max_tokens,
                "previous": self.previous.uuid() if self.previous is not None else None,
            }
        )


def configure_linear_swiglu_fusion(config, vendor_name):
    """Keep this unproven performance option off unless explicitly requested.

    Shape support is independent of MiniCPM or any model class. Only the
    validated MetaX BF16 inference stack is enabled here; unsupported backends
    and configurations retain their existing graph and dispatch policy.
    """
    from vllm.config import CompilationMode

    from vllm_fl.dispatch import is_dump_enabled
    from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend
    from vllm_fl.utils import is_oot_enabled, use_flaggems_op

    extra = config.additional_config
    if not isinstance(extra, dict):
        return False
    settings = extra.get("linear_swiglu_fusion")
    if settings is None or settings is False:
        return False
    if settings is True:
        settings = {}
    if not isinstance(settings, dict) or set(settings) - {"max_tokens"}:
        raise ValueError("linear_swiglu_fusion accepts only max_tokens")
    max_tokens = settings.get("max_tokens", 64)
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("linear_swiglu_fusion.max_tokens must be a positive integer")
    cc, mc = config.compilation_config, config.model_config
    if (
        vendor_name != "metax"
        or mc is None
        or mc.dtype != torch.bfloat16
        or mc.enforce_eager
        or cc.mode != CompilationMode.VLLM_COMPILE
        or config.lora_config is not None
        or config.quant_config is not None
        or config.speculative_config is not None
        or not is_oot_enabled()
        or not use_flaggems_op("linear_swiglu")
        or is_dump_enabled()
        or not FlagGemsBackend().linear_swiglu_is_available()
    ):
        logger.info("Linear/SwiGLU fusion left disabled for this configuration")
        return False
    previous = cc.inductor_compile_config.get(_PASS_KEY)
    if isinstance(previous, LinearSwiGLUPass):
        if previous.max_tokens != max_tokens:
            raise ValueError(
                "linear_swiglu_fusion configured twice with different limits"
            )
        return True
    if previous is not None and not isinstance(previous, InductorPass):
        raise ValueError("Existing post-grad pass must be an InductorPass")
    cc.inductor_compile_config[_PASS_KEY] = LinearSwiGLUPass(max_tokens, previous)
    # Preserve user/other-pass endpoints. vLLM adds the scheduler's upper bound.
    cc.compile_ranges_endpoints = sorted(
        set([*(cc.compile_ranges_endpoints or []), max_tokens])
    )
    logger.info(
        "Enabled linear/SwiGLU graph fusion for token ranges up to %d", max_tokens
    )
    return True
