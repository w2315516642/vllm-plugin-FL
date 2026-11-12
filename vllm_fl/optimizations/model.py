# SPDX-License-Identifier: Apache-2.0
"""Install model-local BF16 Linear and selected Attention implementation extensions."""

import torch

from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from .config import feature_enabled, supports_local_dense


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

    if not supports_local_dense(config, allow_stock_compile=True):
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


def configure_model(model, config, vendor):
    """Install paired components after loading; return tail and Graph policies."""
    from vllm.logger import init_logger
    from ..quantization.w8a8.ffn import install_ffn_w8a8, install_gate_norm_quant
    from ..quantization.w8a8.projections import (
        install_ffn_down_w8a8,
        install_attention_o_w8a8,
        install_qkv_w8a8,
    )
    from ..worker.prefill_tail import make_prefill_tail

    logger = init_logger(__name__)
    components = (
        (install_ffn_w8a8, "Enabled gate/up W8A8 for %d FFNs"),
        (install_ffn_down_w8a8, "Enabled down W8A8 for %d FFNs"),
        (install_attention_o_w8a8, "Enabled O projection W8A8 for %d attention layers"),
        (install_qkv_w8a8, "Enabled paired RMSNorm/QKV W8A8 for %d layers"),
        (install_gate_norm_quant, "Enabled paired gate Norm/quant for %d layers"),
    )
    for install, message in components:
        count = install(model, config, vendor)
        if count:
            logger.info(message, count)

    optimized = install_model_optimizations(model, config, vendor)
    if any(optimized.values()):
        logger.info("Installed model-local optimizations: %s", optimized)
    tail = (
        make_prefill_tail(model, config)
        if feature_enabled(config, "prefill_tail")
        else None
    )
    if tail is not None:
        logger.info("Enabled demand-driven final-layer prefill computation")
    graphs = None
    if optimized["triton_attention"] and feature_enabled(config, "quest_attention"):
        from ..attention.iluvatar._rope import install

        graphs = install(model, config)
        if graphs is not None:
            logger.info(
                "Enabled model-local Quest, RoPE/KV fusion and page-capacity Graphs"
            )
    return tail, graphs
