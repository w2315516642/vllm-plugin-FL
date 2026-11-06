# SPDX-License-Identifier: Apache-2.0
"""Install supported per-model components once, before compilation/capture."""

import torch

from .config import feature_enabled


def install_model_optimizations(model, config, vendor):
    counts = {"bf16_down": 0, "triton_attention": 0}
    if vendor != "iluvatar":
        return counts
    from vllm_fl.utils import is_oot_enabled, use_flaggems_op

    if not is_oot_enabled():
        return counts
    tune_linear = feature_enabled(config, "bf16_linear_tuning") and use_flaggems_op(
        "mm"
    )
    tune_attention = feature_enabled(
        config, "triton_attention_tuning"
    ) and use_flaggems_op("attention_backend")
    if not tune_linear and not tune_attention:
        return counts
    from vllm.model_executor.layers.linear import (
        RowParallelLinear,
        UnquantizedLinearMethod,
    )
    from vllm.model_executor.models.llama import LlamaMLP
    from vllm.v1.attention.backends.triton_attn import (
        AttentionType,
        TritonAttentionImpl,
    )

    from .linear import GemsBF16LinearMethod
    from vllm_fl.attention.iluvatar.impl import TunedTritonAttentionImpl

    mc, pc = config.model_config, config.parallel_config
    if (
        mc.dtype != torch.bfloat16
        or mc.runner_type != "generate"
        or mc.is_encoder_decoder
        or mc.multimodal_config is not None
        or config.quant_config is not None
        or config.lora_config is not None
        or config.speculative_config is not None
        or config.kv_transfer_config is not None
        or pc.tensor_parallel_size != 1
        or pc.pipeline_parallel_size != 1
        or pc.data_parallel_size != 1
        or pc.decode_context_parallel_size != 1
        or pc.prefill_context_parallel_size != 1
        or pc.use_ubatching
        or config.offload_config.uva.cpu_offload_gb > 0
        or config.offload_config.prefetch.offload_group_size > 0
    ):
        return counts
    for module in model.modules():
        if tune_linear and type(module) is LlamaMLP:
            down = module.down_proj
            if (
                type(down) is RowParallelLinear
                and type(down.quant_method) is UnquantizedLinearMethod
                and down.bias is None
                and down.tp_size == 1
                and down.tp_rank == 0
                and down.return_bias
                and down.input_is_parallel
                and down.weight.ndim == 2
                and down.weight.dtype == torch.bfloat16
                and down.weight.device.type == "cuda"
                and down.weight.is_contiguous()
            ):
                down.quant_method = GemsBF16LinearMethod()
                counts["bf16_down"] += 1
        impl = getattr(module, "impl", None)
        if (
            tune_attention
            and type(impl) is TritonAttentionImpl
            and impl.attn_type == AttentionType.DECODER
        ):
            # Backend class/enum, metadata builder and KV-cache update methods
            # stay unchanged. No registry/policy or global function replacement.
            module.impl = TunedTritonAttentionImpl.from_impl(impl)
            counts["triton_attention"] += 1
    return counts
