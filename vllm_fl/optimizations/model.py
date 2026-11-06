# SPDX-License-Identifier: Apache-2.0
"""Install model-local BF16 Linear and selected Attention implementation extensions."""

import torch

from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from .config import feature_enabled


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
